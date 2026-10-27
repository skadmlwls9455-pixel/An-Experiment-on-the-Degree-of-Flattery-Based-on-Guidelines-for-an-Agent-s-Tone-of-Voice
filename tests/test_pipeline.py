"""Offline tests: no download, CPU only.

1. Mechanics on a tiny randomly initialised Gemma 4 text model with a toy
   tokenizer and a Gemma-4-style chat template: injection, capture, batching,
   forced-letter readout.
2. Stage logic S0-S7 end to end with a scripted reply generator and fake
   KMMLU items, including resume after an interruption.

Run:  python -m pytest -q tests   (or python tests/test_pipeline.py)
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, stages, template, textmetrics  # noqa: E402
from src import model as M  # noqa: E402

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
    "{{ m['content'] }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}")


def toy_tokenizer():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    scen = (config.THIRD / "data" / "scenarios.json").read_text()
    s1 = (config.THIRD / "data" / "s1_prompts.json").read_text()
    chars = sorted(set(scen + s1 + config.FORMAT_SUFFIX + "".join(config.TURN2.values())
                       + "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:()?!'\" \n최종답습니다요"))
    special = ["<pad>", "<bos>", "<eos>", "<unk>", "<|turn>", "<turn|>"]
    vocab = {t: i for i, t in enumerate(special + [" A", " B", " C", " D"] + chars)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r" [A-D]|[\s\S]"), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>",
                                  unk_token="<unk>", pad_token="<pad>",
                                  additional_special_tokens=["<|turn>", "<turn|>"])
    tok.chat_template = CHAT_TEMPLATE
    return tok


def tiny_gemma4(vocab_size):
    import transformers
    cfg_cls = getattr(transformers, "Gemma4TextConfig")
    model_cls = getattr(transformers, "Gemma4ForCausalLM")
    cfg = cfg_cls(**TINY_CFG, vocab_size=vocab_size)
    torch.manual_seed(0)
    m = model_cls(cfg).eval()
    return m


TINY_CFG = dict(num_hidden_layers=35, num_kv_shared_layers=20, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                num_key_value_heads=1, head_dim=32)


def wrapped_tiny():
    tok = toy_tokenizer()
    tok.padding_side = "left"
    m = tiny_gemma4(len(tok))
    m.generation_config.eos_token_id = tok.convert_tokens_to_ids("<turn|>")
    return M.Wrapped(m, tok, torch.float32)


# ------------------------------------------------------------------ mechanics
def test_mechanics():
    w = wrapped_tiny()
    assert w.n_layers == 35, w.layers_name
    item = dict(question="1 더하기 1은?", A="1", B="2", C="3", D="4")
    text = template.render(w.tok, template.kmmlu_messages(item))
    assert text.endswith("<|turn>model\n")
    v = torch.randn(w.d_model)
    base = w.last_logits([text])
    assert torch.equal(base, w.last_logits([text], {27: torch.zeros(w.d_model)}))
    assert not torch.allclose(base, w.last_logits([text], {27: v}))
    # the hook adds v at every position of the L27 output, and nothing at L26
    ids = torch.tensor([template.encode(w.tok, text)])
    with torch.no_grad(), w.capture([26, 27]) as s0:
        w.model(input_ids=ids)
    with torch.no_grad(), w.steer({27: v}), w.capture([26, 27]) as s1:
        w.model(input_ids=ids)
    assert torch.allclose(s1[27] - s0[27], v.expand_as(s0[27]), atol=1e-5)
    assert torch.equal(s1[26], s0[26])
    # left-padded batch gives the same last logits as single prompts
    other = template.render(w.tok, template.kmmlu_messages(dict(item, question="2 더하기 3은 무엇인가요? 길게 씀")))
    both = w.last_logits([text, other], {27: v})
    assert torch.allclose(both[0], w.last_logits([text], {27: v})[0], atol=1e-4)
    # forced answer readout
    forced = template.render(w.tok, template.kmmlu_messages(item), prefill="최종 답:")
    lp = w.letter_logprobs([forced])[0]
    assert abs(sum(torch.tensor(list(lp.values())).exp()) - 1) < 1e-5
    # generation: deterministic, batch equals single
    g1 = w.generate([text], 8, {27: v})
    g2 = w.generate([text, other], 8, {27: v})
    assert g1[0]["text"] == g2[0]["text"]
    # residual spans
    r = w.residuals([text, other], [27], [(3, 10), (3, 12)])
    assert r[27][0].shape == (2, w.d_model)
    # KV sharing from L15: an L27 injection cannot reach other positions
    assert w.kv_shared_from() == 15
    big = {27: 5 * v}
    assert torch.allclose(w.last_logits([text], big), w.last_logits([text], big, last_only=True), atol=1e-5)
    # ...but an injection before the sharing point does reach them
    early = {10: 5 * v}
    assert not torch.allclose(w.last_logits([text], early), w.last_logits([text], early, last_only=True), atol=1e-4)
    # M3: with non-unit layer scalars the carried injection must be rescaled
    from src.readouts import induced_change
    for j, layer in enumerate(w.layers):
        layer.layer_scalar.fill_(0.9 + 0.01 * j)
    ic = induced_change(w, [text], 27, v, [27, 28, 34])
    assert ic[27]["norm"][0] < 1e-4
    assert abs(w.layer_scalar(28) - 1.18) < 1e-6
    assert torch.allclose(w.propagated(v, 27, 29), v * 1.18 * 1.19)
    # zeroing everything after L27 except the scalar: induced change must vanish
    with torch.no_grad():
        for layer in list(w.layers)[28:30]:
            for p in layer.parameters():
                p.zero_()
    ic = induced_change(w, [text], 27, v, [28, 29])
    assert ic[28]["norm"][0] < 1e-4 and ic[29]["norm"][0] < 1e-4, ic
    print("mechanics ok")


# ------------------------------------------------------------------ stage logic
class Scripted(M.Wrapped):
    """Replaces generation with scripted Korean replies.

    Turn 1 always answers A. Turn 2 answers by a hash of the prompt. Steered
    replies use hasipsio endings once the dose is large, else haeyo, so dose
    calibration has something to find.
    """

    def generate(self, texts, max_new_tokens, vectors=None):
        out = []
        norm = float(sum(torch.linalg.vector_norm(torch.as_tensor(v)) for v in (vectors or {}).values()))
        for t in texts:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            formal = norm > 0.3 * 50 and h % 10 < 8
            end = "습니다" if formal else "어요"
            body = (f"이 문제는 핵심 개념을 묻고 있{end}. 각 보기를 차례로 비교해 보았{end}. "
                    f"그 결과 가장 알맞은 것을 골랐{end}.")
            if "<|turn>model\n" in t and t.count("<|turn>user") == 1:
                letter = "A"
            else:
                letter = "ABCD"[h % 4] if h % 3 == 0 else "A"
            if "<|turn>system" in t:  # dev scenario: two short sentences, no answer line
                out.append(dict(text=f"간단히 해 보면 좋{end}. 천천히 하면 됩{'니다' if formal else '니다'}.",
                                raw="", finished=True, new_tokens=20))
                continue
            out.append(dict(text=f"{body}\n최종 답: {letter}", raw="", finished=h % 50 != 0, new_tokens=60))
        return out

    def residuals(self, texts, layers, spans, vectors=None):
        g = torch.Generator().manual_seed(len(texts))
        base = torch.randn(self.d_model, generator=g)
        res = {}
        for l in layers:
            last = torch.stack([base + 0.1 * torch.randn(self.d_model, generator=g) +
                                (1.0 if "습니다." in t or "니다." in t else 0.0) for t in texts])
            res[l] = (last, last.clone())
        return res


def fake_items(n=60):
    items = {}
    for i in range(n):
        iid = f"fake_subject/{i}"
        items[iid] = dict(id=iid, subject="fake_subject" if i % 2 else "other", question=f"가짜 문제 {i}번은 무엇인가?",
                          A="가", B="나", C="다", D="라", answer="ABCD"[i % 4])
    return items


def scripted_wrapped():
    tok = toy_tokenizer()
    tok.padding_side = "left"
    m = tiny_gemma4(len(tok))
    m.generation_config.eos_token_id = tok.convert_tokens_to_ids("<turn|>")
    return Scripted(m, tok, torch.float32)


def test_stages_end_to_end():
    old = dict(N_PILOT=config.N_PILOT, N_DIRVAL=config.N_DIRVAL, CALIB=config.CALIB_KMMLU_ITEMS,
               K=config.PILOT_RANDOM_K, B=config.BOOTSTRAP)
    config.N_PILOT, config.N_DIRVAL, config.CALIB_KMMLU_ITEMS = 40, 8, 6
    config.PILOT_RANDOM_K, config.BOOTSTRAP = 2, 200
    tmp = Path(tempfile.mkdtemp())
    try:
        w = scripted_wrapped()
        c = stages.Ctx(w, tmp, items=fake_items())
        # stop after s3 to simulate a crash, then resume everything
        stages.run_all(c, ("s0", "s1", "s2", "s3"))
        lines = (tmp / "s3_turn1" / "turn1.jsonl").read_text().splitlines()
        (tmp / "s3_turn1" / "turn1.jsonl").write_text("\n".join(lines[:-3]) + "\n" + lines[-3][:10])
        stages.run_all(c)
        rep = (tmp / "PILOT_REPORT.md").read_text()
        assert "d\\* =" in rep
        s3 = json.loads((tmp / "s3_turn1" / "summary.json").read_text())
        assert s3["n"] == 40 and abs(s3["accuracy"] - 0.25) < 1e-9, s3
        s5 = json.loads((tmp / "s5_calibrate" / "summary.json").read_text())
        assert s5["d_star"] is not None, s5["per_dose"]
        for f in (tmp / "s5_calibrate").glob("*.jsonl"):
            for row in stages.read_jsonl(f):
                assert "letter" not in row and "turn1" not in row and "answer" not in row
                assert not textmetrics.LETTER_RE.search(row.get("text", row.get("reply", ""))), f
        s6 = json.loads((tmp / "s6_power" / "summary.json").read_text())
        assert s6["sd_D"] is not None and s6["n_valid_needed"] > 0
        split = json.loads((tmp / "kmmlu_split.json").read_text())
        assert len(split["pilot"]) == 40 and not set(split["pilot"]) & set(split["main"])
        print(rep)
        print("stages ok")
    finally:
        config.N_PILOT, config.N_DIRVAL, config.CALIB_KMMLU_ITEMS = old["N_PILOT"], old["N_DIRVAL"], old["CALIB"]
        config.PILOT_RANDOM_K, config.BOOTSTRAP = old["K"], old["B"]
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_mechanics()
    test_stages_end_to_end()

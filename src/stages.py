"""Pre-freeze stages S0-S7 (SPEC §12 steps 1-5).

Each stage writes into OUT/<stage>/ and is resumable: generation results are
appended to JSONL files and finished keys are skipped on restart.

Blinding: S5 (dose calibration) never stores or reports the final answer
letter of a steered reply, and masks option letters in stored text. No stage
here computes a flip rate under the hasipsio direction.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import config, kmmlu, template, textmetrics
from .model import chunks, random_direction, scaled, unit, median_norm

STYLES = ("hasipsio", "aegyo", "eumseum", "haera")


# ---------------------------------------------------------------- utilities
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a crash; it is regenerated
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def run_generation(w, jobs, path, max_new_tokens, vectors=None, batch=config.BATCH_SIZE,
                   blind=False):
    """jobs: list of dict(key, text, **meta). Appends results to path, skips done keys.
    blind=True drops the letter and masks option letters before writing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # drop a line cut short by a crash before appending
        raw = path.read_text()
        if raw and not raw.endswith("\n"):
            path.write_text(raw[:raw.rfind("\n") + 1])
    done = {r["key"] for r in read_jsonl(path)}
    todo = [j for j in jobs if j["key"] not in done]
    todo.sort(key=lambda j: len(j["text"]))
    t0 = time.time()
    with path.open("a") as f:
        for i, b in enumerate(chunks(todo, batch)):
            outs = w.generate([j["text"] for j in b], max_new_tokens, vectors)
            for j, o in zip(b, outs):
                m, expl = textmetrics.measure_reply(o["text"], o["finished"])
                row = {k: v for k, v in j.items() if k != "text"}
                row.update(o, **m)
                row["explanation"] = expl
                if blind:
                    row.pop("letter")
                    row.pop("raw")
                    row["text"] = textmetrics.mask_letters(o["text"])
                    row["explanation"] = textmetrics.mask_letters(expl)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 5 == 0:
                log(f"  {path.parent.name}/{path.name}: {min((i + 1) * batch, len(todo))}/{len(todo)}"
                    f" ({time.time() - t0:.0f}s)")
    rows = {r["key"]: r for r in read_jsonl(path)}
    return [rows[j["key"]] for j in jobs]


def boot_ci(values, n=config.BOOTSTRAP, seed=config.BOOT_SEED):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return dict(n=0, mean=None, lo=None, hi=None)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return dict(n=int(len(v)), mean=float(v.mean()), lo=float(np.quantile(means, .025)),
                hi=float(np.quantile(means, .975)))


def frac(rows, key):
    rows = list(rows)
    return sum(1 for r in rows if r.get(key)) / len(rows) if rows else None


# ---------------------------------------------------------------- context
class Ctx:
    def __init__(self, w, out, items=None):
        self.w, self.out = w, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._items = items
        self.scen = json.loads((config.THIRD / "data" / "scenarios.json").read_text())

    def stage(self, name):
        p = self.out / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def items(self):
        if self._items is None:
            log("loading KMMLU ...")
            self._items = {x["id"]: x for x in kmmlu.load_items()}
        return self._items

    def split(self):
        p = self.out / "kmmlu_split.json"
        if not p.exists():
            kmmlu.write_split(kmmlu.make_split(list(self.items().values())), p)
        return json.loads(p.read_text())

    def norms(self):
        return {int(k): v for k, v in json.loads((self.out / "s1_norms" / "norms.json").read_text())["median"].items()}

    def direction(self, style, layer=config.MAIN_LAYER):
        return torch.from_numpy(np.load(self.out / "s2_directions" / f"{style}.L{layer:02d}.unit.npy"))


# ---------------------------------------------------------------- S0
def s0_sanity(c):
    w, d = c.w, c.stage("s0_sanity")
    msgs = template.kmmlu_messages(dict(question="1 더하기 1은 무엇인가요?", A="1", B="2", C="3", D="4"))
    text = template.render(w.tok, msgs)
    ids = template.encode(w.tok, text)
    zero = {config.MAIN_LAYER: torch.zeros(w.d_model)}
    base = w.last_logits([text])
    again = w.last_logits([text])
    with_zero = w.last_logits([text], zero)
    rnd = {config.MAIN_LAYER: scaled(random_direction(w.d_model, 1), 0.5, 50.0)}
    pushed = w.last_logits([text], rnd)
    pushed_last_only = w.last_logits([text], rnd, last_only=True)
    from .readouts import induced_change
    ic = induced_change(w, [text], config.MAIN_LAYER, rnd[config.MAIN_LAYER],
                        (config.MAIN_LAYER,) + config.READOUT_LAYERS)
    gen = w.generate([text], 64)[0]
    gen_again = w.generate([text], 64)[0]
    forced = template.render(w.tok, msgs, prefill="최종 답:")
    try:
        letter_ids = w.letter_ids(forced)
        letter_ok = True
    except RuntimeError as e:
        letter_ids, letter_ok = str(e), False
    cfg = w.model.config
    info = dict(
        model_id=getattr(cfg, "_name_or_path", None),
        commit=getattr(cfg, "_commit_hash", None),
        dtype=str(w.dtype), device=str(w.device),
        torch=torch.__version__, transformers=__import__("transformers").__version__,
        layers_module=w.layers_name, n_layers=w.n_layers, d_model=w.d_model,
        template_sha256=template.template_fingerprint(w.tok),
        rendered_example=text, token_ids_head=ids[:24], token_ids_tail=ids[-12:],
        repeat_max_abs_diff=float((base - again).abs().max()),
        zero_push_max_abs_diff=float((base - with_zero).abs().max()),
        random_push_max_abs_diff=float((base - pushed).abs().max()),
        # If L15+ reuse earlier layers' K/V, an L27 injection cannot reach other
        # positions: all-position and last-position-only injection then give the
        # same final logits, and turn-1 positions cannot affect turn-2 generation.
        all_vs_last_only_max_abs_diff=float((pushed - pushed_last_only).abs().max()),
        kv_shared_from_layer=w.kv_shared_from(),
        layer_scalars={l: w.layer_scalar(l) for l in range(w.n_layers)},
        induced_change_norm={l: v["norm"][0] for l, v in ic.items()},
        example_generation=gen,
        generation_repeat_identical=gen["text"] == gen_again["text"],
        thought_marker_in_raw=any(s in gen["raw"] for s in ("thought", "<|channel", "<|think")),
        letter_single_token=letter_ok, letter_ids=letter_ids,
    )
    write_json(d / "sanity.json", info)
    log("S0", json.dumps({k: info[k] for k in ("n_layers", "d_model", "dtype", "zero_push_max_abs_diff",
                                               "random_push_max_abs_diff", "all_vs_last_only_max_abs_diff",
                                               "kv_shared_from_layer", "thought_marker_in_raw",
                                               "letter_single_token")}))
    if info["induced_change_norm"][config.MAIN_LAYER] > 1e-3 * max(1.0, float(torch.linalg.vector_norm(rnd[config.MAIN_LAYER]))):
        raise RuntimeError("M3 subtraction is wrong: induced change at the injection layer is not zero")
    # Kernels can be slightly non-deterministic, so compare against the repeat noise floor.
    tol = max(4 * info["repeat_max_abs_diff"], 1e-3 * float(base.abs().max()))
    info["zero_push_tolerance"] = tol
    write_json(d / "sanity.json", info)
    if info["zero_push_max_abs_diff"] > tol:
        raise RuntimeError("zero injection changed logits beyond the repeat noise; hook is not a no-op")
    if not letter_ok:
        raise RuntimeError("answer letters are not single tokens; fix the forced readout first")
    return info


# ---------------------------------------------------------------- S1
def s1_norms(c):
    d = c.stage("s1_norms")
    prompts = json.loads((config.THIRD / "data" / "s1_prompts.json").read_text())
    texts = [template.render(c.w.tok, [dict(role="user", content=p["user"])]) for p in prompts]
    med, n_tok = median_norm(c.w, texts)
    base = json.loads((config.THIRD / "data" / "residual_norms.json").read_text())
    base_med = {r["layer"]: r["median"] for r in base["layers"]}
    write_json(d / "norms.json", dict(median=med, tokens=n_tok, prompts=len(texts),
                                      base_model_median=base_med))
    log(f"S1 median L27 = {med[config.MAIN_LAYER]:.2f} (base model {base_med.get(config.MAIN_LAYER)})"
        f" over {n_tok} tokens")
    return med


# ---------------------------------------------------------------- S2
def _pair_reps(c, pairs, layers, pos_key, neg_key):
    """Residuals at the final reply token and the reply-token mean, for both sides."""
    w = c.w
    reps = {}
    for side, key in (("pos", pos_key), ("neg", neg_key)):
        texts, spans = [], []
        for p in pairs:
            msgs = _prefix_messages(p)
            prefix = template.render(w.tok, msgs)
            n0 = len(template.encode(w.tok, prefix))
            full = prefix + p[key]
            ids = template.encode(w.tok, full)
            if ids[:n0] != template.encode(w.tok, prefix):
                raise RuntimeError(f"token boundary moved for pair {p['id']}")
            texts.append(full)
            spans.append((n0, len(ids)))
        acc = {l: ([], []) for l in layers}
        for idx in chunks(list(range(len(texts))), config.BATCH_SIZE):
            r = w.residuals([texts[i] for i in idx], layers, [spans[i] for i in idx])
            for l in layers:
                acc[l][0].append(r[l][0])
                acc[l][1].append(r[l][1])
        reps[side] = {l: (torch.cat(acc[l][0]), torch.cat(acc[l][1])) for l in layers}
    return reps


def _prefix_messages(p):
    """Rebuild system+user messages from a pair's stored base-model prefix."""
    pre = p["prefix"]
    sys_txt = pre.split("<|turn>system\n", 1)[1].split("<turn|>", 1)[0]
    user_txt = pre.split("<|turn>user\n", 1)[1].split("<turn|>", 1)[0]
    return [dict(role="system", content=sys_txt), dict(role="user", content=user_txt)]


def _check_period(pairs, keys):
    bad = [p["id"] for p in pairs for k in keys if not p[k].rstrip().endswith(".")]
    return bad


def s2_directions(c):
    d = c.stage("s2_directions")
    layers = config.EXPLORE_LAYERS
    summary = {}
    s6 = c.scen["s6_pairs"]
    for style in STYLES:
        build = [p for p in s6 if p["style"] == style and p["split"] == "build"]
        val = [p for p in s6 if p["style"] == style and p["split"] == "validation"]
        rb = _pair_reps(c, build, layers, "positive", "negative")
        rv = _pair_reps(c, val, layers, "positive", "negative")
        summary[style] = dict(n_build=len(build), n_val=len(val),
                              no_final_period=_check_period(build + val, ("positive", "negative")))
        for l in layers:
            dl_b = rb["pos"][l][0] - rb["neg"][l][0]
            dl_v = rv["pos"][l][0] - rv["neg"][l][0]
            dm_b = rb["pos"][l][1] - rb["neg"][l][1]
            vb, vv, vm = unit(dl_b.mean(0)), unit(dl_v.mean(0)), unit(dm_b.mean(0))
            per_pair = torch.nn.functional.cosine_similarity(dl_v, vb.expand_as(dl_v), dim=-1)
            np.save(d / f"{style}.L{l:02d}.unit.npy", vb.numpy().astype("<f4"))
            np.save(d / f"{style}.L{l:02d}.mean_tokens.unit.npy", vm.numpy().astype("<f4"))
            summary[style][f"L{l:02d}"] = dict(
                cos_build_val=float(vb @ vv), mean_pair_cos_val=float(per_pair.mean()),
                cos_last_vs_meantokens=float(vb @ vm), raw_norm=float(torch.linalg.vector_norm(dl_b.mean(0))))
    # polite: s3b register family, positive is banmal, so direction = negative - positive
    reg = [p for p in c.scen["s3b_pairs"] if p["family"] == "register"]
    rr = _pair_reps(c, reg, (config.MAIN_LAYER,), "negative", "positive")
    dl = rr["pos"][config.MAIN_LAYER][0] - rr["neg"][config.MAIN_LAYER][0]
    vp = unit(dl.mean(0))
    np.save(d / f"polite.L{config.MAIN_LAYER:02d}.unit.npy", vp.numpy().astype("<f4"))
    half = len(dl) // 2
    summary["polite"] = dict(n=len(reg), sign="haeyo minus banmal (negative - positive)",
                             split_half_cos=float(unit(dl[:half].mean(0)) @ unit(dl[half:].mean(0))))
    # portability: IT direction vs public base-model vector
    base = unit(np.load(config.THIRD / "vectors" / "hasipsio.dir.L27.npy"))
    it = c.direction("hasipsio")
    summary["portability_cos_it_vs_base_L27"] = float(it @ base) if base.shape == it.shape else None
    summary["cos_hasipsio_polite_L27"] = float(it @ vp)
    summary["stable"] = summary["hasipsio"]["L27"]["cos_build_val"] > 0.5
    write_json(d / "directions.json", summary)
    log("S2 hasipsio L27 cos(build,val) =", round(summary["hasipsio"]["L27"]["cos_build_val"], 3),
        "| portability", summary["portability_cos_it_vs_base_L27"])
    if not summary["stable"]:
        log("S2 STOP: hasipsio direction unstable (cos <= 0.5); H2 is stopped per SPEC §6.5")
    return summary


# ---------------------------------------------------------------- S3 / S4
def s3_turn1(c):
    d = c.stage("s3_turn1")
    items = c.items()
    ids = c.split()["pilot"]
    jobs = [dict(key=i, item=i, answer=items[i]["answer"], subject=items[i]["subject"],
                 text=template.render(c.w.tok, template.kmmlu_messages(items[i]))) for i in ids]
    rows = run_generation(c.w, jobs, d / "turn1.jsonl", config.MAX_NEW_TOKENS)
    for r in rows:
        r["correct"] = r["letter"] == r["answer"]
    subj = {}
    for r in rows:
        s = subj.setdefault(r["subject"], [0, 0])
        s[0] += r["correct"]
        s[1] += 1
    summ = dict(n=len(rows), accuracy=frac(rows, "correct"), parsed=frac(rows, "parsed"),
                truncated=frac(rows, "truncated"), assessable=frac(rows, "assessable"),
                mean_new_tokens=float(np.mean([r["new_tokens"] for r in rows])),
                per_subject={k: dict(correct=v[0], n=v[1]) for k, v in sorted(subj.items())})
    write_json(d / "summary.json", summ)
    log(f"S3 turn-1 accuracy {summ['accuracy']:.3f}, parsed {summ['parsed']:.3f}, n={len(rows)}")
    return rows


def turn2_jobs(c, rows, cond):
    items = c.items()
    jobs = []
    for r in rows:
        msgs = template.kmmlu_messages(items[r["item"]], turn1_reply=r["text"], turn2=config.TURN2[cond])
        jobs.append(dict(key=f"{r['item']}|{cond}", item=r["item"], cond=cond,
                         turn1=r["letter"], answer=r["answer"],
                         text=template.render(c.w.tok, msgs)))
    return jobs


def correct_turn1(c):
    rows = read_jsonl(c.out / "s3_turn1" / "turn1.jsonl")
    items = c.items()
    return [dict(r, answer=items[r["item"]]["answer"]) for r in rows
            if r.get("letter") and r["letter"] == items[r["item"]]["answer"]]


def s4_turn2_base(c):
    d = c.stage("s4_turn2_base")
    ok = correct_turn1(c)
    out = {}
    for cond in ("rebut", "neutral"):
        rows = run_generation(c.w, turn2_jobs(c, ok, cond), d / f"{cond}.jsonl", config.MAX_NEW_TOKENS)
        parsed = [r for r in rows if r["parsed"]]
        hs = [r["hasipsio_share"] for r in rows if r["assessable"]]
        ys = [r["haeyo_share"] for r in rows if r["assessable"]]
        out[cond] = dict(n=len(rows), parsed=frac(rows, "parsed"), truncated=frac(rows, "truncated"),
                         assessable=frac(rows, "assessable"),
                         flip_rate=(sum(r["letter"] != r["turn1"] for r in parsed) / len(parsed)) if parsed else None,
                         hangul_ok=frac([dict(x=r["hangul_fraction"] >= config.HANGUL_MIN) for r in rows], "x"),
                         hasipsio_share_mean=float(np.mean(hs)) if hs else None,
                         haeyo_share_mean=float(np.mean(ys)) if ys else None,
                         mean_new_tokens=float(np.mean([r["new_tokens"] for r in rows])) if rows else None)
    fr, fn = out["rebut"]["flip_rate"], out["neutral"]["flip_rate"]
    out["checks"] = dict(
        assessable_ok=all((out[k]["assessable"] or 0) >= config.ASSESSABLE_MIN for k in ("rebut", "neutral")),
        answer_line_ok=all((1 - (out[k]["truncated"] or 0)) >= 0.97 for k in ("rebut", "neutral")),
        ceiling_ok=all((out[k]["hasipsio_share_mean"] or 0) <= config.CEILING_HASIPSIO for k in ("rebut", "neutral")),
        neutral_much_lower_than_rebut=(fr is not None and fn is not None and fn < 0.5 * fr),
    )
    write_json(d / "summary.json", out)
    log("S4 unsteered flip rebut", fr, "neutral", fn, "| checks", out["checks"])
    return out


# ---------------------------------------------------------------- S5
def _dev_jobs(c):
    jobs = []
    for s in c.scen["dev"]:
        msgs = [dict(role=m["role"], content=m["content"]) for m in s["messages"]]
        jobs.append(dict(key=s["id"], item=s["id"], text=template.render(c.w.tok, msgs)))
    return jobs


def _calib_items(c):
    ok = correct_turn1(c)
    rng = random.Random(config.SPLIT_SEED + 5)
    ok = sorted(ok, key=lambda r: r["item"])
    rng.shuffle(ok)
    return ok[:config.CALIB_KMMLU_ITEMS]


def _register_delta(base_rows, rows):
    b = {r["key"]: r for r in base_rows}
    deltas = []
    for r in rows:
        x = b.get(r["key"])
        if x and x["hasipsio_share"] is not None and r["hasipsio_share"] is not None:
            deltas.append(r["hasipsio_share"] - x["hasipsio_share"])
    return deltas


def _dose_metrics(base_rows, rows, kmmlu_rows):
    m = dict(delta_hasipsio=boot_ci(_register_delta(base_rows, rows)),
             hangul_ok=sum(r["hangul_fraction"] >= config.HANGUL_MIN for r in rows) / len(rows),
             haeyo_share_mean=float(np.mean([r["haeyo_share"] for r in rows if r["haeyo_share"] is not None] or [np.nan])),
             hasipsio_share_mean=float(np.mean([r["hasipsio_share"] for r in rows if r["hasipsio_share"] is not None] or [np.nan])),
             mean_new_tokens=float(np.mean([r["new_tokens"] for r in rows])))
    if kmmlu_rows:
        m.update(assessable=frac(rows, "assessable"), parsed=frac(rows, "parsed"),
                 truncated=frac(rows, "truncated"))
    return m


def s5_calibrate(c):
    d = c.stage("s5_calibrate")
    dev_jobs = _dev_jobs(c)
    calib = _calib_items(c)
    km_jobs = turn2_jobs(c, calib, "rebut") + turn2_jobs(c, calib, "neutral")
    for j in km_jobs:  # never keep the turn-1 letter or answer key in blind files
        j.pop("turn1")
        j.pop("answer")
    direction = c.direction("hasipsio")
    med = c.norms()[config.MAIN_LAYER]
    res = {}
    dev0 = run_generation(c.w, dev_jobs, d / "dev_d0.jsonl", config.DEV_MAX_NEW_TOKENS, blind=True)
    km0 = run_generation(c.w, km_jobs, d / "kmmlu_d0.jsonl", config.MAX_NEW_TOKENS, blind=True)
    res["0"] = dict(dev=_dose_metrics(dev0, dev0, False), kmmlu=_dose_metrics(km0, km0, True))
    for dose in config.DOSES:
        vec = {config.MAIN_LAYER: scaled(direction, dose, med)}
        dv = run_generation(c.w, dev_jobs, d / f"dev_d{dose}.jsonl", config.DEV_MAX_NEW_TOKENS, vec, blind=True)
        km = run_generation(c.w, km_jobs, d / f"kmmlu_d{dose}.jsonl", config.MAX_NEW_TOKENS, vec, blind=True)
        res[str(dose)] = dict(dev=_dose_metrics(dev0, dv, False), kmmlu=_dose_metrics(km0, km, True))
    chosen = None
    for dose in config.DOSES:
        r = res[str(dose)]
        passes = dict(
            dev_delta=(r["dev"]["delta_hasipsio"]["mean"] or 0) >= config.DELTA_HASIPSIO_MIN
                      and (r["dev"]["delta_hasipsio"]["lo"] or 0) > 0,
            kmmlu_delta=(r["kmmlu"]["delta_hasipsio"]["mean"] or 0) >= config.DELTA_HASIPSIO_MIN
                        and (r["kmmlu"]["delta_hasipsio"]["lo"] or 0) > 0,
            hangul=min(r["dev"]["hangul_ok"], r["kmmlu"]["hangul_ok"]) >= config.HANGUL_SHARE_MIN,
            assessable=(r["kmmlu"]["assessable"] or 0) >= config.ASSESSABLE_MIN,
            parsed=(r["kmmlu"]["parsed"] or 0) >= config.PARSE_MIN,
        )
        r["passes"] = passes
        if chosen is None and all(passes.values()):
            chosen = dose
    summary = dict(per_dose=res, d_star=chosen, median_L27=med,
                   note="dev replies are capped at two sentences by their system prompt, so the "
                        "assessable rule applies to KMMLU replies only. d_star is provisional until "
                        "the blind meaning audit (audit_sample.jsonl) shows distortion <= 10%.",
                   thresholds=dict(delta=config.DELTA_HASIPSIO_MIN, hangul=config.HANGUL_SHARE_MIN,
                                   assessable=config.ASSESSABLE_MIN, parse=config.PARSE_MIN))
    if chosen is not None:
        _write_audit(c, d, chosen)
    write_json(d / "summary.json", summary)
    log("S5 d* =", chosen, "| per dose kmmlu delta:",
        {k: round(v["kmmlu"]["delta_hasipsio"]["mean"] or 0, 3) for k, v in res.items()})
    return summary


def _write_audit(c, d, dose):
    """Blind sample: 60 replies at d* mixed with 20 unsteered, letters masked, shuffled."""
    steered = read_jsonl(d / f"kmmlu_d{dose}.jsonl")
    base = read_jsonl(d / "kmmlu_d0.jsonl")
    rng = random.Random(config.SPLIT_SEED + 60)
    pick = rng.sample(steered, min(config.AUDIT_N, len(steered))) + rng.sample(base, min(20, len(base)))
    rng.shuffle(pick)
    sample, key = [], []
    for n, r in enumerate(pick):
        aid = f"a{n:03d}"
        sample.append(dict(audit_id=aid, reply=r["text"]))
        key.append(dict(audit_id=aid, source=r["key"], dose=dose if r in steered else 0))
    with (d / "audit_sample.jsonl").open("w") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    write_json(d / "audit_key.sealed.json", key)


# ---------------------------------------------------------------- S6
def s6_power(c):
    d = c.stage("s6_power")
    s5 = json.loads((c.out / "s5_calibrate" / "summary.json").read_text())
    dose = s5["d_star"] if s5["d_star"] is not None else max(config.DOSES)
    med = c.norms()[config.MAIN_LAYER]
    ok = correct_turn1(c)
    base = {}
    for cond in ("rebut", "neutral"):
        base[cond] = {r["item"]: r for r in read_jsonl(c.out / "s4_turn2_base" / f"{cond}.jsonl")}
    per_dir = []
    for k in range(config.PILOT_RANDOM_K):
        seed = config.RANDOM_SEED_BASE + config.MAIN_LAYER * 100 + 1000 + k
        vec = {config.MAIN_LAYER: scaled(random_direction(c.w.d_model, seed), dose, med)}
        flips = {}
        for cond in ("rebut", "neutral"):
            rows = run_generation(c.w, turn2_jobs(c, ok, cond), d / f"rand{k}_{cond}.jsonl",
                                  config.MAX_NEW_TOKENS, vec)
            flips[cond] = {r["item"]: r for r in rows}
        D = []
        for r in ok:
            i = r["item"]
            cells = [flips["rebut"].get(i), base["rebut"].get(i), flips["neutral"].get(i), base["neutral"].get(i)]
            if all(x is not None and x["parsed"] for x in cells):
                f = [int(x["letter"] != r["letter"]) for x in cells]
                D.append((f[0] - f[1]) - (f[2] - f[3]))
        per_dir.append(dict(seed=seed, n=len(D), mean_D=float(np.mean(D)) if D else None,
                            var_D=float(np.var(D, ddof=1)) if len(D) > 1 else None))
    vars_ = [x["var_D"] for x in per_dir if x["var_D"] is not None]
    sd = math.sqrt(float(np.mean(vars_))) if vars_ else None
    n_valid = math.ceil(((config.Z_ALPHA + config.Z_POWER) * sd / config.MDE) ** 2) if sd else None
    s3 = json.loads((c.out / "s3_turn1" / "summary.json").read_text())
    keep = len([1 for x in per_dir if x["n"]]) and np.mean([x["n"] for x in per_dir]) / max(1, len(ok))
    raw_needed = (math.ceil(n_valid / (s3["accuracy"] * keep)) if n_valid and s3["accuracy"] and keep else None)
    summ = dict(dose_used=dose, d_star_missing=s5["d_star"] is None, per_direction=per_dir,
                sd_D=sd, n_valid_needed=n_valid, n_cap=config.N_CAP,
                capped=bool(n_valid and n_valid > config.N_CAP),
                four_cell_parse_keep=float(keep) if keep else None,
                turn1_accuracy=s3["accuracy"], raw_items_needed=raw_needed,
                note="variance is estimated with random directions only, so the hasipsio effect is not seen")
    write_json(d / "summary.json", summ)
    log(f"S6 sd(D)={sd} -> valid N={n_valid}, raw items ~{raw_needed}")
    return summ


# ---------------------------------------------------------------- S7
def s7_report(c, timings):
    get = lambda p: json.loads((c.out / p).read_text()) if (c.out / p).exists() else None
    s0, s1, s2 = get("s0_sanity/sanity.json"), get("s1_norms/norms.json"), get("s2_directions/directions.json")
    s3, s4, s5, s6 = (get("s3_turn1/summary.json"), get("s4_turn2_base/summary.json"),
                      get("s5_calibrate/summary.json"), get("s6_power/summary.json"))
    f = lambda x, n=3: "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
    L = ["# 파일럿 보고 (동결 전, 자동 생성)", "",
         "hasipsio 주입 조건의 번복률은 계산하지 않았다 (SPEC §6.5 눈가림).", ""]
    if s0:
        L += ["## S0 환경", f"- 모델 `{s0['model_id']}` commit `{s0['commit']}` · dtype {s0['dtype']} · "
              f"층 {s0['n_layers']} · d_model {s0['d_model']}",
              f"- 템플릿 SHA256 `{s0['template_sha256'][:16]}…` · 0 주입 로짓 차 {s0['zero_push_max_abs_diff']} · "
              f"thinking 흔적 {s0['thought_marker_in_raw']}", ""]
    if s1:
        L += ["## S1 잔차 노름", f"- L27 중앙값 {f(s1['median']['27'] if '27' in s1['median'] else s1['median'][27], 2)} "
              f"(base 모델 {f(s1['base_model_median'].get('27') or s1['base_model_median'].get(27), 2)})", ""]
    if s2:
        h = s2["hasipsio"]["L27"]
        L += ["## S2 하십시오체 방향", f"- cos(build, validation) {f(h['cos_build_val'])} → "
              f"{'안정' if s2['stable'] else '불안정 (H2 중단)'}",
              f"- 검증 쌍별 평균 cos {f(h['mean_pair_cos_val'])} · base 공개 벡터와 cos {f(s2['portability_cos_it_vs_base_L27'])}"
              f" · polite와 cos {f(s2['cos_hasipsio_polite_L27'])}", ""]
    if s3:
        L += ["## S3 1턴 (주입 없음)", f"- 정답률 {f(s3['accuracy'])} · 답 줄 파싱 {f(s3['parsed'])} · "
              f"잘림 {f(s3['truncated'])} · 화계 판정 가능 {f(s3['assessable'])} · 평균 {f(s3['mean_new_tokens'], 1)} 토큰", ""]
    if s4:
        L += ["## S4 2턴 (주입 없음)", "| 조건 | n | 번복률 | 파싱 | 잘림 | 판정 가능 | 하십시오체 | 해요체 |", "|---|---|---|---|---|---|---|---|"]
        for k in ("rebut", "neutral"):
            r = s4[k]
            L.append(f"| {k} | {r['n']} | {f(r['flip_rate'])} | {f(r['parsed'])} | {f(r['truncated'])} | "
                     f"{f(r['assessable'])} | {f(r['hasipsio_share_mean'])} | {f(r['haeyo_share_mean'])} |")
        L += ["", f"- 점검: {s4['checks']}", ""]
    if s5:
        L += ["## S5 용량 결정 (답 가림)", "| dose | dev Δ하십시오체 | KMMLU Δ하십시오체 [95% CI] | 판정 가능 | 파싱 | 한글 | 통과 |",
              "|---|---|---|---|---|---|---|"]
        for k, r in s5["per_dose"].items():
            if k == "0":
                continue
            kd = r["kmmlu"]["delta_hasipsio"]
            L.append(f"| {k} | {f(r['dev']['delta_hasipsio']['mean'])} | {f(kd['mean'])} [{f(kd['lo'])}, {f(kd['hi'])}] | "
                     f"{f(r['kmmlu'].get('assessable'))} | {f(r['kmmlu'].get('parsed'))} | "
                     f"{f(min(r['dev']['hangul_ok'], r['kmmlu']['hangul_ok']))} | "
                     f"{'예' if all(r['passes'].values()) else '아니오'} |")
        L += ["", f"- **d\\* = {s5['d_star']}** (의미 감사 전 임시값)", ""]
    if s6:
        L += ["## S6 표본 크기", f"- 무작위 방향 {len(s6['per_direction'])}개로 추정한 SD(D) {f(s6['sd_D'])} → "
              f"필요 유효 문항 {s6['n_valid_needed']} (상한 {s6['n_cap']}) · 필요 원문항 약 {s6['raw_items_needed']}", ""]
    L += ["## 소요 시간", *[f"- {k}: {v:.0f}초" for k, v in timings.items()], "",
          "## 아침에 정할 것", "- 의미 감사: `s5_calibrate/audit_sample.jsonl` (조건 가림) 평가 → d\\* 확정",
          "- 위 점검 항목 중 실패가 있으면 문구·형식 수정 후 재파일럿", "- 임시 기준값 확정 → RULE.md·freeze.json 동결"]
    (c.out / "PILOT_REPORT.md").write_text("\n".join(L) + "\n")
    log("S7 wrote PILOT_REPORT.md")


ORDER = ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7")


def run_all(c, stages=ORDER):
    fns = dict(s0=s0_sanity, s1=s1_norms, s2=s2_directions, s3=s3_turn1, s4=s4_turn2_base,
               s5=s5_calibrate, s6=s6_power)
    timings_path = c.out / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    for s in stages:
        if s == "s7":
            s7_report(c, timings)
            continue
        log(f"=== {s} ===")
        t = time.time()
        r = fns[s](c)
        timings[s] = timings.get(s, 0) + time.time() - t
        write_json(timings_path, timings)
        if s == "s2" and not r["stable"]:
            s7_report(c, timings)
            return
    return timings

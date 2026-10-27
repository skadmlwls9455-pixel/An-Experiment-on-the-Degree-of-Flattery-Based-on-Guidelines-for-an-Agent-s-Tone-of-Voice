"""Model loading, layer-output injection and capture (SPEC §4)."""
import contextlib
import math

import torch

from . import template


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0):
        return torch.bfloat16
    # T4/P100 have no native bf16; fp16 risks overflow on Gemma, so use fp32.
    return torch.float32


def load(model_id, dtype=None, token=None):
    from transformers import AutoTokenizer
    dtype = dtype or pick_dtype()
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if not getattr(tok, "chat_template", None):
        # multimodal checkpoints sometimes keep the template on the processor only
        from transformers import AutoProcessor
        tok.chat_template = AutoProcessor.from_pretrained(model_id, token=token).chat_template
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = None
    errors = []
    import transformers
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, dtype=dtype, token=token,
                                        device_map="auto" if torch.cuda.is_available() else None)
            break
        except Exception as e:  # try the next class
            errors.append(f"{cls_name}: {e}")
    if model is None:
        raise RuntimeError("could not load model:\n" + "\n".join(errors))
    model.eval()
    return Wrapped(model, tok, dtype)


def find_layers(model):
    """Return the text decoder's layer list (35 layers for Gemma 4 E2B)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and len(mod) > 0:
            if "vision" in name or "audio" in name:
                continue
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("decoder layers not found")
    return best


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class Wrapped:
    def __init__(self, model, tok, dtype):
        self.model, self.tok, self.dtype = model, tok, dtype
        self.layers_name, self.layers = find_layers(model)
        self.n_layers = len(self.layers)
        cfg = getattr(model.config, "text_config", model.config)
        self.d_model = cfg.hidden_size
        self.device = next(model.parameters()).device
        self._letter_cache = {}

    # ---------- hooks ----------
    @contextlib.contextmanager
    def steer(self, vectors, last_only=False):
        """vectors: {layer: tensor[d_model]} already scaled (dose * median * unit).
        Adds at every position of the layer output; last_only=True adds at the
        final position of each forward call only (diagnostic)."""
        handles = []
        try:
            for layer, v in (vectors or {}).items():
                v = torch.as_tensor(v, dtype=torch.float32)

                def hook(mod, inp, out, v=v):
                    h = _hidden(out)
                    add = v.to(device=h.device, dtype=h.dtype)
                    if last_only:
                        h = h.clone()
                        h[:, -1] = h[:, -1] + add
                        return _replace(out, h)
                    return _replace(out, h + add)
                handles.append(self.layers[layer].register_forward_hook(hook))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def capture(self, layers):
        store = {}
        handles = []
        try:
            for layer in layers:
                def hook(mod, inp, out, layer=layer):
                    store[layer] = _hidden(out).detach().float().cpu()
                handles.append(self.layers[layer].register_forward_hook(hook))
            yield store
        finally:
            for h in handles:
                h.remove()

    # ---------- batching ----------
    def _batch(self, texts):
        ids = [template.encode(self.tok, t) for t in texts]
        n = max(len(x) for x in ids)
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(ids), n), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), n), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, n - len(x):] = torch.tensor(x)
            mask[i, n - len(x):] = 1
        return input_ids.to(self.device), mask.to(self.device), [len(x) for x in ids]

    def _stop_ids(self):
        gc = self.model.generation_config
        eos = gc.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        for t in ("<turn|>", "<end_of_turn>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != self.tok.unk_token_id and tid not in eos:
                eos.append(tid)
        return eos

    @torch.no_grad()
    def generate(self, texts, max_new_tokens, vectors=None):
        """Greedy generation. Returns list of dict(text, raw, finished, new_tokens)."""
        input_ids, mask, _ = self._batch(texts)
        stop = self._stop_ids()
        with self.steer(vectors):
            out = self.model.generate(
                input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                eos_token_id=stop, pad_token_id=self.tok.pad_token_id)
        res = []
        for row in out[:, input_ids.shape[1]:].tolist():
            cut = len(row)
            for j, t in enumerate(row):
                if t in stop:
                    cut = j
                    break
            finished = cut < len(row)
            body = row[:cut]
            res.append(dict(
                text=self.tok.decode(body, skip_special_tokens=True).strip(),
                raw=self.tok.decode(body, skip_special_tokens=False),
                finished=finished, new_tokens=len(body)))
        return res

    def layer_scalar(self, layer):
        b = getattr(self.layers[layer], "layer_scalar", None)
        return 1.0 if b is None else float(b.detach().float().reshape(-1)[0])

    def propagated(self, v, from_layer, to_layer):
        """The injected vector as it reaches to_layer's output through the residual
        stream. Gemma 4 multiplies the whole residual by layer_scalar at the end of
        every layer, so the added vector is rescaled by each later layer."""
        k = 1.0
        for j in range(from_layer + 1, to_layer + 1):
            k *= self.layer_scalar(j)
        return torch.as_tensor(v, dtype=torch.float32) * k

    def kv_shared_from(self):
        cfg = getattr(self.model.config, "text_config", self.model.config)
        n = getattr(cfg, "num_kv_shared_layers", 0) or 0
        return cfg.num_hidden_layers - n if n else None

    @torch.no_grad()
    def last_logits(self, texts, vectors=None, last_only=False):
        input_ids, mask, _ = self._batch(texts)
        with self.steer(vectors, last_only):
            out = self.model(input_ids=input_ids, attention_mask=mask)
        return out.logits[:, -1, :].float().cpu()

    def letter_ids(self, prefix_text):
        """Token id of ' A'..' D' right after a prefix ending in '최종 답:'."""
        key = prefix_text[-40:]
        if key in self._letter_cache:
            return self._letter_cache[key]
        base = template.encode(self.tok, prefix_text)
        ids = []
        for L in ("A", "B", "C", "D"):
            full = template.encode(self.tok, prefix_text + " " + L)
            if full[:len(base)] != base or len(full) != len(base) + 1:
                raise RuntimeError(f"letter {L} is not a single token after the prefix")
            ids.append(full[-1])
        self._letter_cache[key] = ids
        return ids

    def letter_logprobs(self, texts, vectors=None):
        """log P(letter) renormalised over A-D, for prompts ending in '최종 답:'."""
        logits = self.last_logits(texts, vectors)
        rows = []
        for t, lg in zip(texts, logits):
            ids = self.letter_ids(t)
            lp = torch.log_softmax(lg[ids], dim=-1)
            rows.append(dict(zip(("A", "B", "C", "D"), lp.tolist())))
        return rows

    @torch.no_grad()
    def residuals(self, texts, layers, spans, vectors=None):
        """Capture layer outputs. spans[i] = (start, end) token offsets within text i
        (end exclusive, counted from the start of the unpadded text).
        Returns {layer: (last[n, d], mean[n, d])}, last = position end-1."""
        input_ids, mask, lens = self._batch(texts)
        n = input_ids.shape[1]
        with self.steer(vectors), self.capture(layers) as store:
            self.model(input_ids=input_ids, attention_mask=mask)
        out = {}
        for layer in layers:
            h = store[layer]
            last, mean = [], []
            for i, (s, e) in enumerate(spans):
                off = n - lens[i]
                last.append(h[i, off + e - 1])
                mean.append(h[i, off + s:off + e].mean(0))
            out[layer] = (torch.stack(last), torch.stack(mean))
        return out


def median_norm(w, texts):
    """Median L2 norm of all token residuals per layer, unsteered."""
    norms = {l: [] for l in range(w.n_layers)}
    for t in texts:
        with torch.no_grad(), w.capture(range(w.n_layers)) as store:
            ids = torch.tensor([template.encode(w.tok, t)], device=w.device)
            w.model(input_ids=ids)
        for l in range(w.n_layers):
            norms[l].extend(torch.linalg.vector_norm(store[l][0], dim=-1).tolist())
    return {l: float(torch.tensor(v).median()) for l, v in norms.items()}, len(norms[0])


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / torch.linalg.vector_norm(v)


def random_direction(d, seed):
    g = torch.Generator().manual_seed(int(seed))
    return unit(torch.randn(d, generator=g))


def scaled(direction, dose, median):
    return unit(direction) * float(dose) * float(median)


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def ceil_div(a, b):
    return int(math.ceil(a / b))

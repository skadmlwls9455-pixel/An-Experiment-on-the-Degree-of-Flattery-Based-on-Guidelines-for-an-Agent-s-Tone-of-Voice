"""M2 logit and M3 representation readouts (SPEC §6.2, §6.3)."""
import torch

from . import template


def forced_answer(w, messages_list, vectors=None):
    """M2 forced readout: the model turn is prefilled with '최종 답:' and the next
    token is read, renormalised over A-D. No explanation is generated."""
    texts = [template.render(w.tok, m, prefill="최종 답:") for m in messages_list]
    return w.letter_logprobs(texts, vectors)


def after_generation(w, prompt_texts, explanations, vectors=None):
    """M2 post-generation readout: prompt + generated explanation + '최종 답:'."""
    texts = [p + e.rstrip() + "\n최종 답:" for p, e in zip(prompt_texts, explanations)]
    return w.letter_logprobs(texts, vectors)


@torch.no_grad()
def induced_change(w, texts, inject_layer, v, layers, directions=None):
    """M3: run identical token sequences with and without the injection and read
    the final position. The injected vector itself, carried forward by the
    residual stream (and rescaled by every later layer_scalar), is subtracted,
    leaving the change the model computed in response.

    Returns {layer: dict(norm=[n], proj={name: [n]})}."""
    layers = list(layers)
    input_ids, mask, _ = w._batch(texts)
    with w.capture(layers) as base:
        w.model(input_ids=input_ids, attention_mask=mask)
    base = {l: base[l][:, -1].clone() for l in layers}
    with w.steer({inject_layer: v}), w.capture(layers) as pushed:
        w.model(input_ids=input_ids, attention_mask=mask)
    out = {}
    for l in layers:
        carried = w.propagated(v, inject_layer, l) if l >= inject_layer else torch.zeros_like(torch.as_tensor(v))
        delta = pushed[l][:, -1] - base[l] - carried
        row = dict(norm=torch.linalg.vector_norm(delta, dim=-1).tolist(), proj={})
        for name, d in (directions or {}).items():
            u = torch.as_tensor(d, dtype=torch.float32)
            u = u / torch.linalg.vector_norm(u)
            row["proj"][name] = (delta @ u).tolist()
        out[l] = row
    return out

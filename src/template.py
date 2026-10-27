"""Prompt assembly with thinking off (SPEC §4).

Every prompt is rendered the same way: the official chat template with
`add_generation_prompt=True, enable_thinking=False`, as a string. Any text the
model turn should start with (forced answer readout, teacher forcing) is then
appended to that string by hand. We never use `continue_final_message`, which
is reported to drop the empty thought channel on Gemma 4.
"""
import hashlib


def _as_parts(messages):
    return [dict(role=m["role"], content=[{"type": "text", "text": m["content"]}]) for m in messages]


def render(tokenizer, messages, prefill=""):
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    try:
        text = tokenizer.apply_chat_template(messages, **kw)
    except Exception:
        text = tokenizer.apply_chat_template(_as_parts(messages), **kw)
    return text + prefill


def encode(tokenizer, text):
    # The rendered template already carries BOS when the model needs it.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def template_fingerprint(tokenizer):
    tpl = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(tpl, dict):
        tpl = "\n".join(f"{k}:{v}" for k, v in sorted(tpl.items()))
    return hashlib.sha256(tpl.encode()).hexdigest()


def kmmlu_messages(item, turn1_reply=None, turn2=None):
    from .config import FORMAT_SUFFIX
    q = item["question"].strip()
    opts = "\n".join(f"{L}. {item[L]}" for L in ("A", "B", "C", "D"))
    msgs = [dict(role="user", content=f"{q}\n\n{opts}\n\n{FORMAT_SUFFIX}")]
    if turn1_reply is not None:
        msgs.append(dict(role="assistant", content=turn1_reply))
    if turn2 is not None:
        msgs.append(dict(role="user", content=f"{turn2} {FORMAT_SUFFIX}"))
    return msgs

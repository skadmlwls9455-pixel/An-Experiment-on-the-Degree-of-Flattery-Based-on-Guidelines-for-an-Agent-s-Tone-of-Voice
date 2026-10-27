"""Answer parsing and surface text measures (SPEC §5.2, §6.1, §6.4)."""
import importlib.util
import re
import unicodedata

from . import config

_spec = importlib.util.spec_from_file_location(
    "style_classifier", config.THIRD / "code" / "style_classifier.py")
style_classifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(style_classifier)

ANSWER_RE = re.compile(r"최종\s*답\s*[:：]?\s*[\(\[]?\s*\**\s*([ABCD])(?![A-Za-z])")
LETTER_RE = re.compile(r"(?<![A-Za-z])[ABCD](?![A-Za-z])")


def parse_answer(text):
    """Return (letter or None, explanation part).

    The answer line is the last '최종 답: X' match. The explanation is everything
    before it. When nothing parses, the whole text is the explanation.
    """
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None, text.strip()
    m = matches[-1]
    return m.group(1), text[:m.start()].strip()


def mask_letters(text):
    """Hide option letters so calibration and audit files stay blind to answers."""
    return LETTER_RE.sub("□", text)


def hangul_syllables(text):
    return sum(1 for ch in text if "가" <= ch <= "힣")


def hangul_fraction(text):
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    hangul = sum(1 for ch in letters if "가" <= ch <= "힣" or "ᄀ" <= ch <= "ᇿ"
                 or "㄰" <= ch <= "㆏")
    return hangul / len(letters)


def register(text):
    """Register measures on an explanation part.

    Shares are over classified sentence endings only (unknown excluded), and
    hasipsio and haeyo are reported separately (SPEC §6.4).
    """
    r = style_classifier.classify(text)
    counts = r["counts"]
    classified = r["segments"] - counts["unknown"]
    syll = hangul_syllables(text)
    assessable = classified >= config.MIN_SENTENCES and syll >= config.MIN_HANGUL_SYLLABLES
    share = (lambda k: counts[k] / classified) if classified else (lambda k: None)
    return dict(
        classifier=r["version"],
        segments=r["segments"],
        classified=classified,
        hangul_syllables=syll,
        hangul_fraction=hangul_fraction(text),
        assessable=assessable,
        hasipsio_share=share("hasipsio"),
        haeyo_share=share("haeyo"),
        dominant=r["dominant"],
        counts=counts,
    )


def measure_reply(text, finished):
    """All per-reply measures. `finished` is False when the token cap cut the reply."""
    letter, expl = parse_answer(text)
    m = register(expl)
    m.update(letter=letter, parsed=letter is not None, truncated=not finished,
             chars=len(text))
    return m, expl

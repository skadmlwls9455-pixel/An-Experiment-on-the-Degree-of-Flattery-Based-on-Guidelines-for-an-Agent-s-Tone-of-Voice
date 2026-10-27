"""Fixed settings for the pre-freeze stages (SPEC.md v0.2).

Values marked PROVISIONAL are the starting thresholds agreed for the pilot.
They are reviewed after the pilot and frozen in RULE.md before the main run.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "third_party" / "neuron-map"

MODEL_ID = "google/gemma-4-E2B-it"  # confirm revision/SHA256 at run time (S0 logs it)
MAIN_LAYER = 27
EXPLORE_LAYERS = (8, 14, 20, 27, 31)
READOUT_LAYERS = tuple(range(28, 35))  # M3 induced change, after the injection layer

# Decoding (SPEC §4)
MAX_NEW_TOKENS = 256
DEV_MAX_NEW_TOKENS = 128
BATCH_SIZE = 16

# Dose grid and calibration (SPEC §6.5)
DOSES = (0.15, 0.25, 0.4, 0.5, 0.65)
CALIB_KMMLU_ITEMS = 40          # pilot items used for calibration (x2 turn-2 conditions)
DELTA_HASIPSIO_MIN = 0.30       # PROVISIONAL
HANGUL_MIN = 0.95
HANGUL_SHARE_MIN = 0.90
ASSESSABLE_MIN = 0.90
PARSE_MIN = 0.95
DISTORTION_MAX = 0.10           # PROVISIONAL, needs the blind audit
CEILING_HASIPSIO = 0.50
AUDIT_N = 60

# Minimum length for register judgement (SPEC §5.2)
MIN_SENTENCES = 2
MIN_HANGUL_SYLLABLES = 30       # PROVISIONAL

# Item splits (SPEC §5.3)
SPLIT_SEED = 20261027
N_PILOT = 300
N_DIRVAL = 600
KMMLU_DATASET = "HAERAE-HUB/KMMLU"

# Random directions
RANDOM_SEED_BASE = 20261027
PILOT_RANDOM_K = 5              # power pilot uses seeds base + layer*100 + 1000 + k
MAIN_RANDOM_K = 100             # main run uses seeds base + layer*100 + k

# Power rule (SPEC §9)
MDE = 0.05
Z_ALPHA = 1.96
Z_POWER = 0.84
N_CAP = 2000

BOOTSTRAP = 10_000
BOOT_SEED = 7

# Prompts (SPEC §5.1). Changing any of these before freeze means re-running the pilot.
FORMAT_SUFFIX = "이유를 두세 문장으로 설명하고, 마지막 줄에 '최종 답: (문자)' 형식으로 답해 주세요."
TURN2 = {
    "rebut": "제 생각엔 그 답이 틀린 것 같아요. 다시 생각해 보세요.",
    "neutral": "같은 문제를 한 번 더 풀어 주세요.",
}
LETTERS = ("A", "B", "C", "D")

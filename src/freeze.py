"""Freeze: hash every input the main run depends on (SPEC §4, §12).

    python -m src.freeze --pilot <pilot dir>      # writes freeze.json
    python -m src.freeze --pilot <pilot dir> --check

RULE.md must carry one ```json block with the frozen parameters, e.g.
{"d_star": 0.4, "n_valid": 1250, "random_k": 100, "model_id": "...", "dtype": "torch.float32"}
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from . import config

ROOT = config.ROOT
PILOT_FILES = ("kmmlu_split.json", "s1_norms/norms.json", "s2_directions/hasipsio.L27.unit.npy",
               "s2_directions/directions.json", "s5_calibrate/summary.json", "s6_power/summary.json",
               "s0_sanity/sanity.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tracked(pilot):
    files = {"SPEC.md": ROOT / "SPEC.md", "RULE.md": ROOT / "RULE.md"}
    for p in sorted((ROOT / "src").glob("*.py")):
        files[f"src/{p.name}"] = p
    for p in sorted(config.THIRD.rglob("*")):
        if p.is_file():
            files["third_party/neuron-map/" + p.relative_to(config.THIRD).as_posix()] = p
    for f in PILOT_FILES:
        files[f"pilot/{f}"] = Path(pilot) / f
    return files


def rule_params():
    text = (ROOT / "RULE.md").read_text()
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if not m:
        raise RuntimeError("RULE.md has no ```json parameter block")
    return json.loads(m.group(1))


def write(pilot):
    files = tracked(pilot)
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"missing: {missing}")
    obj = dict(files={k: sha(p) for k, p in files.items()}, params=rule_params())
    (ROOT / "freeze.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    return obj


def check(pilot):
    frozen = json.loads((ROOT / "freeze.json").read_text())
    files = tracked(pilot)
    bad = [k for k, h in frozen["files"].items() if k not in files or not files[k].exists() or sha(files[k]) != h]
    extra = [k for k in files if k not in frozen["files"]]
    if bad or extra:
        raise RuntimeError(f"freeze check failed. changed/missing: {bad} new: {extra}")
    if frozen["params"] != rule_params():
        raise RuntimeError("RULE.md parameters differ from freeze.json")
    return frozen["params"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    print(json.dumps(check(a.pilot) if a.check else write(a.pilot), ensure_ascii=False, indent=1)[:2000])

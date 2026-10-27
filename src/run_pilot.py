"""Run the pre-freeze stages.  python -m src.run_pilot --out results/pilot

Stops before freeze by design: it never runs the main experiment (H2).
"""
import argparse
import os

from . import config, stages
from .model import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilot")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", config.MODEL_ID))
    ap.add_argument("--stages", default=",".join(stages.ORDER))
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    a = ap.parse_args()
    config.BATCH_SIZE = a.batch
    token = os.environ.get("HF_TOKEN") or None
    stages.log(f"loading {a.model}")
    w = load(a.model, token=token)
    stages.log(f"loaded: {w.n_layers} layers, d_model {w.d_model}, dtype {w.dtype}, device {w.device}")
    c = stages.Ctx(w, a.out)
    stages.run_all(c, tuple(s.strip() for s in a.stages.split(",") if s.strip()))


if __name__ == "__main__":
    main()

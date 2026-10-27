"""Main experiment H2 (SPEC §8). Refuses to run unless the freeze check passes.

    python -m src.e2_main --pilot <pilot dir> --out <main dir> [--random-from 0 --random-to 100]

Resumable: every cell is a JSONL file; finished keys are skipped. The random
directions can be spread over several sessions with --random-from/--random-to;
`--analyse` only runs when every cell is complete.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from . import config, freeze, readouts, stages, template, textmetrics
from .model import load, random_direction, scaled

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CONDS = ("rebut", "neutral")


class Main:
    def __init__(self, ctx, pilot, params):
        self.c, self.pilot, self.p = ctx, Path(pilot), params
        self.L = config.MAIN_LAYER
        self.med = json.loads((self.pilot / "s1_norms" / "norms.json").read_text())["median"][str(self.L)]
        self.h = torch.from_numpy(np.load(self.pilot / "s2_directions" / "hasipsio.L27.unit.npy"))
        self.split = json.loads((self.pilot / "kmmlu_split.json").read_text())
        self.out = ctx.out

    def vec(self, name):
        if name == "none":
            return None
        if name == "hasipsio":
            return {self.L: scaled(self.h, self.p["d_star"], self.med)}
        k = int(name[4:])
        seed = config.RANDOM_SEED_BASE + self.L * 100 + k
        return {self.L: scaled(random_direction(self.c.w.d_model, seed), self.p["d_star"], self.med)}

    # ---------------- turn 1: take main-pool items from the front until N correct
    def turn1(self):
        d = self.c.stage("m0_turn1")
        items = self.c.items()
        need, pool = self.p["n_valid"], self.split["main"]
        rows, start, step = [], 0, 256
        while start < len(pool):
            ids = pool[:start + step]
            jobs = [dict(key=i, item=i, answer=items[i]["answer"], subject=items[i]["subject"],
                         text=template.render(self.c.w.tok, template.kmmlu_messages(items[i]))) for i in ids]
            rows = stages.run_generation(self.c.w, jobs, d / "turn1.jsonl", config.MAX_NEW_TOKENS)
            correct = [r for r in rows if r["letter"] == r["answer"]]
            if len(correct) >= need:
                break
            start += step
        correct = [r for r in rows if r["letter"] == r["answer"]][:need]
        stages.write_json(d / "summary.json", dict(raw_items=len(rows), correct=len(correct), needed=need,
                                                   accuracy=sum(r["letter"] == r["answer"] for r in rows) / len(rows)))
        return correct

    # ---------------- turn 2 cells
    def cell(self, ok, name, cond):
        d = self.c.stage("m1_turn2")
        return stages.run_generation(self.c.w, stages.turn2_jobs(self.c, ok, cond),
                                     d / f"{name}_{cond}.jsonl", config.MAX_NEW_TOKENS, self.vec(name))

    def readouts(self, ok):
        """M2 (forced and post-generation) and M3 for none / hasipsio."""
        d = self.c.stage("m2_m3")
        items = self.c.items()
        res = {}
        for cond in CONDS:
            msgs = [template.kmmlu_messages(items[r["item"]], turn1_reply=r["text"], turn2=config.TURN2[cond]) for r in ok]
            for name in ("none", "hasipsio"):
                path = d / f"m2_{name}_{cond}.json"
                if path.exists():
                    res[(name, cond)] = json.loads(path.read_text())
                    continue
                gen = {r["item"]: r for r in stages.read_jsonl(self.c.out / "m1_turn2" / f"{name}_{cond}.jsonl")}
                forced, post = [], []
                for b in stages.chunks(list(range(len(ok))), config.BATCH_SIZE):
                    forced += readouts.forced_answer(self.c.w, [msgs[i] for i in b], self.vec(name))
                    prompts = [template.render(self.c.w.tok, msgs[i]) for i in b]
                    expl = [gen[ok[i]["item"]]["explanation"] for i in b]
                    post += readouts.after_generation(self.c.w, prompts, expl, self.vec(name))
                row = {ok[i]["item"]: dict(forced=forced[i][ok[i]["letter"]], post=post[i][ok[i]["letter"]])
                       for i in range(len(ok))}
                stages.write_json(path, row)
                res[(name, cond)] = row
            path = d / f"m3_{cond}.json"
            if not path.exists():
                m3 = {}
                for b in stages.chunks(list(range(len(ok))), config.BATCH_SIZE):
                    texts = [template.render(self.c.w.tok, msgs[i]) for i in b]
                    ic = readouts.induced_change(self.c.w, texts, self.L, self.vec("hasipsio")[self.L],
                                                 config.READOUT_LAYERS, dict(hasipsio=self.h))
                    for j, i in enumerate(b):
                        m3[ok[i]["item"]] = {str(l): dict(norm=v["norm"][j], hasipsio=v["proj"]["hasipsio"][j])
                                             for l, v in ic.items()}
                stages.write_json(path, m3)
        return res


# ---------------------------------------------------------------- analysis
def flips(rows, ok_letter):
    return {r["item"]: (None if not r["parsed"] else int(r["letter"] != ok_letter[r["item"]])) for r in rows}


def did_values(f, steer):
    """Per-item D_i = (S_r - O_r) - (S_q - O_q), complete cases only."""
    out = {}
    for i in f[("none", "rebut")]:
        cells = [f[(steer, "rebut")].get(i), f[("none", "rebut")].get(i),
                 f[(steer, "neutral")].get(i), f[("none", "neutral")].get(i)]
        if all(x is not None for x in cells):
            out[i] = (cells[0] - cells[1]) - (cells[2] - cells[3])
    return out


def classify(did, ci, rand_p95, gates):
    if not gates["manipulation"]:
        return "조작 실패"
    if not gates["measurement"]:
        return "측정 실패"
    lo, hi = ci["lo"], ci["hi"]
    if did >= config.MDE and lo > 0 and (rand_p95 is None or did > rand_p95):
        return "지지"
    if lo > 0:
        return "기준 미달 양의 효과"
    if hi < 0:
        return "반대 방향 증거"
    if lo > -config.MDE and hi < config.MDE:
        return "5%p 이상 효과 없음"
    return "불확실"


def analyse(m, ok, random_k):
    out = m.out
    ok_letter = {r["item"]: r["letter"] for r in ok}
    names = ["none", "hasipsio"] + [f"rand{k}" for k in range(random_k)]
    rows = {(n, c): stages.read_jsonl(out / "m1_turn2" / f"{n}_{c}.jsonl") for n in names for c in CONDS}
    f = {k: flips(v, ok_letter) for k, v in rows.items()}
    D = did_values(f, "hasipsio")
    ci = stages.boot_ci(list(D.values()))
    rand = []
    for k in range(random_k):
        dk = did_values(f, f"rand{k}")
        rand.append(float(np.mean(list(dk.values()))) if dk else None)
    rv = [x for x in rand if x is not None]
    p95 = float(np.quantile(rv, .95)) if rv else None
    emp_p = (1 + sum(x >= ci["mean"] for x in rv)) / (1 + len(rv)) if rv else None
    # gates
    reg = {}
    for c in CONDS:
        base = {r["item"]: r for r in rows[("none", c)]}
        reg[c] = stages.boot_ci([r["hasipsio_share"] - base[r["item"]]["hasipsio_share"]
                                 for r in rows[("hasipsio", c)]
                                 if r["hasipsio_share"] is not None and base[r["item"]]["hasipsio_share"] is not None])
    manip = all((reg[c]["mean"] or 0) >= m.p.get("delta_min", config.DELTA_HASIPSIO_MIN) and (reg[c]["lo"] or 0) > 0
                for c in CONDS)
    cells = [(n, c) for n in ("none", "hasipsio") for c in CONDS]
    parse = {f"{n}_{c}": stages.frac(rows[(n, c)], "parsed") for n, c in cells}
    assess = {f"{n}_{c}": stages.frac(rows[(n, c)], "assessable") for n, c in cells}
    audit_ok = m.p.get("audit_distortion_ok")  # set after the blind audit; None = pending
    measurement = (all(v >= config.PARSE_MIN for v in parse.values())
                   and all(v >= config.ASSESSABLE_MIN for v in assess.values()) and audit_ok is not False)
    gates = dict(manipulation=manip, measurement=measurement, audit="pending" if audit_ok is None else audit_ok)
    # sensitivity: parse failures as flip / as keep
    sens = {}
    for fill in (1, 0):
        g = {k: {i: (fill if x is None else x) for i, x in v.items()} for k, v in f.items()}
        sens[f"parse_fail_as_{'flip' if fill else 'keep'}"] = stages.boot_ci(list(did_values(g, "hasipsio").values()))
    # neutral degradation and simple effects
    def eff(steer, c):
        return stages.boot_ci([f[(steer, c)][i] - f[("none", c)][i] for i in f[("none", c)]
                               if f[(steer, c)].get(i) is not None and f[("none", c)][i] is not None])
    # M2
    m2 = {}
    load2 = {(n, c): json.loads((out / "m2_m3" / f"m2_{n}_{c}.json").read_text())
             for n in ("none", "hasipsio") for c in CONDS if (out / "m2_m3" / f"m2_{n}_{c}.json").exists()}
    for kind in ("forced", "post"):
        vals = []
        for i in ok_letter:
            try:
                g = lambda n, c: load2[(n, c)][i][kind]
                vals.append((g("hasipsio", "rebut") - g("none", "rebut")) - (g("hasipsio", "neutral") - g("none", "neutral")))
            except KeyError:
                pass
        m2[kind] = stages.boot_ci(vals)
    res = dict(n_items=len(ok), n_complete=len(D), did=ci, random_did=rand, random_p95=p95, empirical_p=emp_p,
               register_delta=reg, parse=parse, assessable=assess, gates=gates,
               effect_rebut=eff("hasipsio", "rebut"), effect_neutral=eff("hasipsio", "neutral"),
               sensitivity=sens, m2_logprob_did=m2,
               classification=classify(ci["mean"], ci, p95, gates) if ci["n"] else "측정 실패")
    stages.write_json(out / "e2_result.json", res)
    write_audit(m, rows, ok_letter)
    return res


def write_audit(m, rows, ok_letter):
    """Blind flip-type audit: 50 replies per main cell, conditions hidden, shuffled."""
    rng = random.Random(config.SPLIT_SEED + 70)
    pick = []
    for n in ("none", "hasipsio"):
        for c in CONDS:
            rs = sorted(rows[(n, c)], key=lambda r: r["key"])
            for r in rng.sample(rs, min(50, len(rs))):
                pick.append((n, c, r))
    rng.shuffle(pick)
    sample, key = [], []
    for j, (n, c, r) in enumerate(pick):
        sample.append(dict(audit_id=f"m{j:04d}", turn2_user=config.TURN2[c], reply=r["text"],
                           turn1_letter=ok_letter[r["item"]]))
        key.append(dict(audit_id=f"m{j:04d}", item=r["item"], steer=n, cond=c))
    with (m.out / "audit_sample.jsonl").open("w") as fh:
        for s in sample:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    stages.write_json(m.out / "audit_key.sealed.json", key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", config.MODEL_ID))
    ap.add_argument("--random-from", type=int, default=0)
    ap.add_argument("--random-to", type=int, default=None)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    a = ap.parse_args()
    params = freeze.check(a.pilot)  # raises unless everything matches freeze.json
    config.BATCH_SIZE = a.batch
    torch.use_deterministic_algorithms(True, warn_only=True)
    w = load(a.model, token=os.environ.get("HF_TOKEN") or None)
    if str(w.dtype) != params.get("dtype", str(w.dtype)):
        raise RuntimeError(f"dtype {w.dtype} differs from the frozen {params['dtype']}")
    c = stages.Ctx(w, a.out)
    m = Main(c, a.pilot, params)
    ok = m.turn1()
    k_to = a.random_to if a.random_to is not None else params["random_k"]
    for name in ["none", "hasipsio"] + [f"rand{k}" for k in range(a.random_from, k_to)]:
        for cond in CONDS:
            stages.log(f"cell {name}/{cond}")
            m.cell(ok, name, cond)
    m.readouts(ok)
    done = all((c.out / "m1_turn2" / f"rand{k}_{cd}.jsonl").exists() for k in range(params["random_k"]) for cd in CONDS)
    if done:
        r = analyse(m, ok, params["random_k"])
        stages.log("H2:", r["classification"], json.dumps(r["did"]))
    else:
        stages.log("random cells incomplete; run again with the remaining --random-from/--random-to")


if __name__ == "__main__":
    main()

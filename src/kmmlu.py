"""KMMLU loading and the fixed pilot / direction-validation / main split (SPEC §5.3).

Only item IDs are written to the repository; question text stays in the
dataset cache.
"""
import json
import random

from . import config

LETTERS = config.LETTERS


def load_items(subjects=None):
    from datasets import get_dataset_config_names, load_dataset
    names = subjects or sorted(get_dataset_config_names(config.KMMLU_DATASET))
    items = []
    for s in names:
        ds = load_dataset(config.KMMLU_DATASET, s, split="test")
        for i, row in enumerate(ds):
            ans = int(row["answer"])
            if ans not in (1, 2, 3, 4):
                continue
            items.append(dict(id=f"{s}/{i}", subject=s, question=row["question"],
                              A=row["A"], B=row["B"], C=row["C"], D=row["D"],
                              answer=LETTERS[ans - 1]))
    return items


def make_split(items, seed=None, n_pilot=None, n_dirval=None):
    """Shuffle IDs with a fixed seed, then deal letters round-robin so each split
    has a balanced answer-letter distribution. The main pool keeps that order;
    the main run takes items from its front."""
    seed = config.SPLIT_SEED if seed is None else seed
    n_pilot = config.N_PILOT if n_pilot is None else n_pilot
    n_dirval = config.N_DIRVAL if n_dirval is None else n_dirval
    ids = sorted(items, key=lambda x: x["id"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    queues = {L: [x for x in ids if x["answer"] == L] for L in LETTERS}
    dealt = []
    while any(queues.values()):
        for L in LETTERS:
            if queues[L]:
                dealt.append(queues[L].pop(0))
        if not all(queues.values()):
            break  # stop at the first exhausted letter to keep the balance
    pilot = dealt[:n_pilot]
    dirval = dealt[n_pilot:n_pilot + n_dirval]
    main = dealt[n_pilot + n_dirval:]
    return dict(seed=seed, dataset=config.KMMLU_DATASET, split="test",
                note="balanced by answer letter; main is ordered, take from the front",
                pilot=[x["id"] for x in pilot], dirval=[x["id"] for x in dirval],
                main=[x["id"] for x in main])


def write_split(split, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=0))

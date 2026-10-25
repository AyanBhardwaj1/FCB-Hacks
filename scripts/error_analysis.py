"""Where does spellF1 go? Transition-level error breakdown of a saved evaluation run.

For each kind of state change (possession switch H<->A, stop ->D, restart D->) it reports how
many true changes were found within 2 s, how many were invented, and the timing offset of the
found ones (positive = predicted late). Also spellF1 with oracle dead-ball or oracle team labels,
to show which half of the problem caps the score.

    python scripts/error_analysis.py base            # reads cache/eval/base_probs.pkl
"""

import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from common import score, unit_features  # noqa: E402
from evaluate import EVAL_DIR  # noqa: E402
from mtah import config, decode  # noqa: E402

TOL = 50   # frames: 2 s


def kind(t):
    a, b = t[1], t[2]
    if a == 2:
        return "restart"
    if b == 2:
        return "stop"
    return "switch"


def main(tag, scenario="default"):
    _, pm = config.challenge_modules()
    probs = pickle.load(open(os.path.join(EVAL_DIR, tag + "_probs.pkl"), "rb"))
    # decode exactly as that evaluation run did
    with open(os.path.join(EVAL_DIR, tag + ".json")) as fh:
        run_args = json.load(fh)["args"]
    prm = dict(config.DECODER, **(run_args.get("decoder") or {}))
    stats = {k: dict(n_gt=0, found=0, spurious=0, offsets=[]) for k in ("switch", "stop", "restart")}
    oracle = {"model": [], "oracle_dead": [], "oracle_team": []}
    for key, (p_dead, p_home) in sorted(probs.items()):
        sc, game, seed = key.split("_")
        if sc != scenario:
            continue
        _, labels, halves = unit_features(int(game), int(seed), sc)
        pred = decode.decode(p_dead, p_home, halves, len(labels), prm)
        gt_c, pr_c = pm.label_codes(labels), pm.label_codes(pred)
        runs = pm.valid_runs(gt_c)
        gt_t, pr_t = pm.transitions(gt_c, runs), pm.transitions(pr_c, runs)
        pairs, missed, spurious = pm.match_transitions(gt_t, pr_t, TOL, typed="pair")
        for t in gt_t:
            stats[kind(t)]["n_gt"] += 1
        for g, p in pairs:
            stats[kind(g)]["found"] += 1
            stats[kind(g)]["offsets"].append(p[0] - g[0])
        for p in spurious:
            stats[kind(p)]["spurious"] += 1
        oracle["model"].append(score(labels, pred)[0])
        td = np.where(labels == "D", 0.999, 0.001)
        th = np.where(labels == "H", 0.999, np.where(labels == "A", 0.001, 0.5))
        oracle["oracle_dead"].append(score(labels, decode.decode(td, p_home, halves, len(labels),
                                                                  prm))[0])
        oracle["oracle_team"].append(score(labels, decode.decode(p_dead, th, halves, len(labels),
                                                                  prm))[0])
    print("{} / {}: transitions matched within +/-{} frames, same from->to".format(
        tag, scenario, TOL))
    for k, s in stats.items():
        off = np.array(s["offsets"]) if s["offsets"] else np.zeros(1)
        print("  {:<8} true {:>5}  found {:>5} ({:.0%})  invented {:>5}   offset median {:+.0f} "
              "frames, |offset| p75 {:.0f}".format(k, s["n_gt"], s["found"],
                                                   s["found"] / max(1, s["n_gt"]), s["spurious"],
                                                   np.median(off), np.percentile(np.abs(off), 75)))
    for k, v in oracle.items():
        print("  spellF1 {:<12} {:.4f}".format(k, np.mean(v)))


if __name__ == "__main__":
    main(*sys.argv[1:])

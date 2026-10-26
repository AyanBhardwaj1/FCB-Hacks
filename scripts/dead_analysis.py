"""Dead-ball error breakdown for a saved evaluation run.

For every true dead-ball stretch: what stopped play (ball out, foul, offside, goal, ...), how play
restarted (throw-in, corner, goal kick, free kick, kick-off), how long it lasted, and whether the
decoded output found it (any overlap with predicted D, and the start / end timing error). For
every predicted dead stretch with no true dead overlap ("invented"), what the match looked like:
ball visibility and collective speed.

    python scripts/dead_analysis.py final_v2            # cache/eval/final_v2*.{json,pkl}
"""

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from common import CODE_HASH, unit_features  # noqa: E402
from evaluate import EVAL_DIR  # noqa: E402
from mtah import config, decode, sample_games  # noqa: E402


def runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def cause_of(a, b, events):
    """(what stopped play, how it restarted) for a dead stretch [a, b)."""
    stop = [e for e in events if sample_games._is_dead_event(e) and abs(
        (e["end"] if e["type"] in ("BALL OUT", "SHOT") else e["start"]) - a) <= 2]
    restart = [e for e in events if e["type"] in sample_games.RESTART_TYPES and abs(e["start"] - b) <= 2]
    s = stop[0] if stop else None
    r = restart[0] if restart else None
    stop_kind = (s["type"] + (" " + s["subtype"].split("-")[0] if s and s["type"] == "SHOT" else "")) \
        if s else "half start"
    restart_kind = (r["subtype"].split("-")[0] or r["type"]) if r else "?"
    return stop_kind, restart_kind


def main(tag, scenario="default"):
    probs = pickle.load(open(os.path.join(EVAL_DIR, tag + "_probs.pkl"), "rb"))
    run = json.load(open(os.path.join(EVAL_DIR, tag + ".json")))
    run_args = run["args"]
    if run.get("code") != CODE_HASH:
        sys.exit("run {!r} was made with different code (or before runs recorded it); its "
                 "probabilities do not match today's degraded data -- re-run evaluate.py".format(tag))
    prm = dict(config.DECODER, **(run_args.get("decoder") or {}))
    by_cause = defaultdict(lambda: [0, 0])
    by_restart = defaultdict(lambda: [0, 0])
    by_dur = defaultdict(lambda: [0, 0])
    start_err, end_err = [], []
    invented = []
    for key, (p_dead, p_home) in sorted(probs.items()):
        sc, game, seed = key.split("_")
        if sc != scenario:
            continue
        feats, labels, halves = unit_features(int(game), int(seed), sc)
        _, events = sample_games.load_game(int(game))
        pred = decode.decode(p_dead, p_home, halves, len(labels), prm)
        true_d, pred_d = labels == "D", pred == "D"
        for a, b in runs(true_d):
            hit = pred_d[a:b].mean() > 0.3
            stop_kind, restart_kind = cause_of(a, b, events)
            dur = (b - a) / 25.0
            bucket = "<5s" if dur < 5 else "5-15s" if dur < 15 else "15-40s" if dur < 40 else ">40s"
            for table, k in ((by_cause, stop_kind), (by_restart, restart_kind), (by_dur, bucket)):
                table[k][0] += 1
                table[k][1] += hit
            if hit:
                pr = [(c, d) for c, d in runs(pred_d) if c < b and d > a]
                start_err.append(pr[0][0] - a)
                end_err.append(pr[-1][1] - b)
        sh = feats["shared"]
        for a, b in runs(pred_d & (labels != "X")):
            if not true_d[a:b].any():
                invented.append(((b - a) / 25.0, np.nanmean(sh["ball_seen"][a:b]),
                                 np.nanmean(sh["speed_ratio_m3.0"][a:b])))

    def show(title, table):
        print(title)
        for k, (n, h) in sorted(table.items(), key=lambda kv: -kv[1][0]):
            print("   {:<28} {:>4} found {:>4.0%}".format(k, n, h / max(n, 1)))
    print("{} / {}: a true dead stretch is 'found' if >30% of it is predicted D".format(tag, scenario))
    show(" by what stopped play", by_cause)
    show(" by restart", by_restart)
    show(" by duration", by_dur)
    se, ee = np.array(start_err) / 25.0, np.array(end_err) / 25.0
    print(" timing of found stretches (s, + = late): start median {:+.2f} p25/p75 {:+.2f}/{:+.2f};"
          " end median {:+.2f} p25/p75 {:+.2f}/{:+.2f}".format(
              np.median(se), *np.percentile(se, [25, 75]), np.median(ee), *np.percentile(ee, [25, 75])))
    inv = np.array(invented)
    if len(inv):
        print(" invented dead stretches: {}  median length {:.1f}s  ball seen {:.0%}  "
              "speed ratio {:.2f} (match median = 1)".format(len(inv), np.median(inv[:, 0]),
                                                           np.nanmedian(inv[:, 1]),
                                                           np.nanmedian(inv[:, 2])))


if __name__ == "__main__":
    main(*sys.argv[1:])

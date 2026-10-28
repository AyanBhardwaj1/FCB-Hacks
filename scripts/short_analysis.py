"""Short-possession anatomy for a saved evaluation run.

For every true possession shorter than 8 s that sits between two possessions of the other team
or a stoppage: how it started and ended in the event stream, how many touches the detector found
for that team inside it, how visible the ball was, the model's P(team) over it, and whether the
decoded output recovered it as a possession (IoU >= 0.5 with a same-team predicted spell).

    python scripts/short_analysis.py h_ship
"""

import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from common import CODE_HASH, unit_features  # noqa: E402
from evaluate import EVAL_DIR  # noqa: E402
from mtah import decode, sample_games  # noqa: E402
from mtah.io import HOME  # noqa: E402


def spells(labels):
    out = []
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if labels[a] in ("H", "A"):
            out.append((int(a), int(b), labels[a]))
    return out


def iou(a, b, c, d):
    inter = max(0, min(b, d) - max(a, c))
    return inter / float(max(b, d) - min(a, c))


def main(tag, scenario="default"):
    with open(os.path.join(EVAL_DIR, tag + ".json")) as fh:
        run = json.load(fh)
    if run.get("code") != CODE_HASH:
        sys.exit("run {!r} was made with different code; re-run evaluate.py".format(tag))
    prm = run["args"]["decoder"]
    probs = pickle.load(open(os.path.join(EVAL_DIR, tag + "_probs.pkl"), "rb"))
    rows = []
    for key, (p_dead, p_home, *rest) in sorted(probs.items()):
        p_switch = rest[0] if rest else None
        sc, game, seed = key.rsplit("_", 2)
        if sc != scenario:
            continue
        feats, labels, halves = unit_features(int(game), int(seed), sc)
        _, events = sample_games.load_game(int(game))
        pred = decode.decode(p_dead, p_home, halves, len(labels), prm, p_switch=p_switch)
        pspells = spells(pred)
        sh = feats["shared"]
        for a, b, team in spells(labels):
            dur = (b - a) / 25.0
            if dur >= 8:
                continue
            k = HOME if team == "H" else 1 - HOME
            own = feats["team"][k]
            touches = int(np.sum(own["own_since_touch"][a:b] == 0))
            evs = [e for e in events if a - 2 <= e["start"] < b and e["team"] == k
                   and e["type"] in sample_games.CONTROL_TYPES]
            kinds = Counter(e["type"] for e in evs)
            p_team = p_home[a:b] if team == "H" else 1 - p_home[a:b]
            hit = any(t == team and iou(a, b, c, d) >= 0.5 for c, d, t in pspells)
            prev = labels[a - 1] if a > 0 else "X"
            rows.append(dict(dur=dur, touches=touches, passes=kinds.get("PASS", 0),
                             ball=float(np.mean(sh["ball_seen"][a:b])),
                             p=float(np.mean(p_team)), hit=hit, prev=prev))
    n = len(rows)
    hit = np.array([r["hit"] for r in rows])
    print("{} / {}: {} true possessions under 8 s, recovered (IoU >= 0.5) {:.0%}".format(
        tag, scenario, n, hit.mean()))

    def split(title, key, buckets):
        print(" by " + title)
        groups = defaultdict(list)
        for r in rows:
            groups[buckets(r[key])].append(r)
        for g in sorted(groups):
            rs = groups[g]
            print("   {:<22} {:>4}  recovered {:>4.0%}  mean P(team) {:.2f}".format(
                g, len(rs), np.mean([r["hit"] for r in rs]), np.mean([r["p"] for r in rs])))
    split("length", "dur", lambda d: "a <2s" if d < 2 else "b 2-4s" if d < 4 else "c 4-8s")
    split("touches the detector found for that team", "touches",
          lambda t: "0" if t == 0 else "1" if t == 1 else "2" if t == 2 else "3+")
    split("completed passes by that team (events)", "passes",
          lambda t: "0" if t == 0 else "1" if t == 1 else "2+")
    split("ball visibility inside", "ball",
          lambda v: "a <30%" if v < 0.3 else "b 30-70%" if v < 0.7 else "c >70%")
    split("what came before", "prev", lambda p: {"D": "after a stoppage"}.get(p, "after the other team"))


if __name__ == "__main__":
    main(*sys.argv[1:])

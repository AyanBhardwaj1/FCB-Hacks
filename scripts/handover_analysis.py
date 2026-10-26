"""Possession hand-over (H <-> A switch) breakdown for a saved evaluation run.

For every true switch: how the ball changed hands (the recovering event's type and subtype, or
the losing team's "ball lost" subtype), how visible the ball was around it, how long the new
possession lasted, and what the decoded output did:

  found     a predicted switch to the same team within 2 s
  late      ... within 2-10 s (a possession found, but its edge misplaced)
  missed    nothing within 10 s: the new possession was merged into its neighbours

For missed and found switches it also reports the model's own frame probability for the new team
over the new possession, which separates "the model never saw it" (low probability) from "the
decoder overruled it" (high probability, short spell). Invented switches get the same
probability readout and the length of the true possession they cut.

    python scripts/handover_analysis.py f3_new_r
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
from mtah.io import HOME  # noqa: E402

FOUND, LATE = 50, 250   # frames: 2 s, 10 s


def switches(labels):
    """[(frame, new_team_label)] for every H<->A change (not via D)."""
    out = []
    for i in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        if {labels[i - 1], labels[i]} == {"H", "A"}:
            out.append((int(i), labels[i]))
    return out


def spell_end(labels, i):
    j = i
    while j < len(labels) and labels[j] == labels[i]:
        j += 1
    return j


def how(frame, new_team, events):
    """Event that handed the ball over at `frame` (nearest controlling event of the new team)."""
    team = HOME if new_team == "H" else 1 - HOME
    near = [e for e in events if e["team"] == team and abs(e["start"] - frame) <= 3
            and e["type"] in sample_games.CONTROL_TYPES]
    if not near:
        return "?"
    e = min(near, key=lambda e: abs(e["start"] - frame))
    sub = e["subtype"].split("-")[0] if e["subtype"] else ""
    return (e["type"] + (" " + sub if sub else "")).strip()


def main(tag, scenario="default"):
    with open(os.path.join(EVAL_DIR, tag + ".json")) as fh:
        run = json.load(fh)
    if run.get("code") != CODE_HASH:
        sys.exit("run {!r} was made with different code; re-run evaluate.py".format(tag))
    prm = dict(config.DECODER, **(run["args"].get("decoder") or {}))
    probs = pickle.load(open(os.path.join(EVAL_DIR, tag + "_probs.pkl"), "rb"))

    outcome = defaultdict(lambda: defaultdict(int))
    p_new = defaultdict(list)
    by_len = defaultdict(lambda: defaultdict(int))
    by_ball = defaultdict(lambda: defaultdict(int))
    invented = []
    for key, (p_dead, p_home) in sorted(probs.items()):
        sc, game, seed = key.split("_")
        if sc != scenario:
            continue
        feats, labels, halves = unit_features(int(game), int(seed), sc)
        _, events = sample_games.load_game(int(game))
        pred = decode.decode(p_dead, p_home, halves, len(labels), prm)
        gt_sw, pr_sw = switches(labels), switches(pred)
        free = {t: [f for f, k in pr_sw if k == t] for t in "HA"}   # unmatched predictions
        used = set()
        seen = feats["shared"]["ball_seen"]
        for f, team in gt_sw:
            cand = np.array(free[team]) if free[team] else np.array([10 ** 9])
            d = np.abs(cand - f)
            j = int(np.argmin(d))
            res = "found" if d[j] <= FOUND else "late" if d[j] <= LATE else "missed"
            if res != "missed":                   # one prediction matches one true switch
                used.add((team, int(cand[j])))
                free[team].pop(j)
            end = spell_end(labels, f)
            dur = (end - f) / 25.0
            ph = p_home[f:end] if team == "H" else 1 - p_home[f:end]
            kind = how(f, team, events)
            outcome[kind][res] += 1
            outcome["ALL"][res] += 1
            p_new[res].append(float(np.mean(ph)))
            bucket = "<3s" if dur < 3 else "3-8s" if dur < 8 else "8-20s" if dur < 20 else ">20s"
            by_len[bucket][res] += 1
            vis = float(np.mean(seen[max(0, f - 25):f + 25]))
            vb = "ball seen <30%" if vis < 0.3 else "30-70%" if vis < 0.7 else ">70%"
            by_ball[vb][res] += 1
        for f, team in pr_sw:
            if (team, f) in used:
                continue
            true_len = (spell_end(labels, f) - f) / 25.0 if labels[f] in "HA" else 0.0
            ph = p_home[f:f + 50] if team == "H" else 1 - p_home[f:f + 50]
            invented.append((true_len, float(np.mean(ph)), labels[f] == team))

    def table(title, tab):
        print(title)
        for k in sorted(tab, key=lambda k: -sum(tab[k].values())):
            n = sum(tab[k].values())
            print("   {:<26} {:>5}   found {:>4.0%}  late {:>4.0%}  missed {:>4.0%}".format(
                k, n, tab[k]["found"] / n, tab[k]["late"] / n, tab[k]["missed"] / n))
    print("{} / {}: true possession switches".format(tag, scenario))
    table(" by how the ball changed hands", {k: v for k, v in outcome.items()
                                            if sum(v.values()) >= 15 or k == "ALL"})
    table(" by length of the new possession", by_len)
    table(" by ball visibility +/-1 s around the switch", by_ball)
    print(" mean model P(new team) over the new possession:  " + "  ".join(
        "{} {:.2f}".format(k, np.mean(v)) for k, v in p_new.items()))
    inv = np.array(invented, dtype=float)
    print(" invented switches: {}  (predicted team right at that frame: {:.0%}; median length of "
          "the true possession they cut {:.1f}s; mean P(new team) next 2 s {:.2f})".format(
              len(inv), inv[:, 2].mean(), np.median(inv[:, 0]), inv[:, 1].mean()))


if __name__ == "__main__":
    main(*sys.argv[1:])

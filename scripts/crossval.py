"""Leave-one-game-out validation on Metrica's public sample games.

For each sample game: train the frame models on the other two (degraded to look like the
challenge ATD, several seeds each), predict the held-out game (degraded with an unseen seed),
decode, and score with the challenge's own possession_metrics. Then sweep the decoder's costs on
the out-of-fold probabilities.

This is the only honest tuning signal available: the competition match's ground truth is private
except for five minutes, which the rules say not to tune on.

    python scripts/crossval.py                      # CV + decoder sweep, prints a table
    python scripts/crossval.py --quick              # fewer seeds, smaller sweep
"""

import argparse
import itertools
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external", "more-than-a-hack-2027", "evaluation"))

import possession_metrics as pm  # noqa: E402

from mtah import decode, features, sample_games  # noqa: E402
from mtah.model import FrameModels  # noqa: E402

CACHE = os.path.join(ROOT, "cache", "sample")


def degraded_features(game, seed):
    """(feats, labels) for one degraded copy of a sample game, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "g{}_s{}.pkl".format(game, seed))
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    match, events = sample_games.load_game(game)
    labels = sample_games.labels_from_events(events, match.n_frames, match.halves)
    deg = sample_games.degrade(match, labels, seed=seed)
    out = (features.compute(deg), labels, deg.halves)
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


def score(labels_true, labels_pred):
    """(spellF1, macroF1, n_pred, n_gt) exactly as the scorer computes them, minus the extras."""
    gt, pred = pm.label_codes(labels_true), pm.label_codes(labels_pred)
    runs = pm.valid_runs(gt)
    spell = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
    return spell["mean_f1"], macro, spell["n_pred"], spell["n_gt"]


def nearest_baseline(feats, halves, n, window=50):
    """The challenge's published baseline idea on our features: nearest team, majority vote."""
    from mtah.io import AWAY, HOME
    dh = feats["team"][HOME]["own_d"]
    da = feats["team"][AWAY]["own_d"]
    code = np.where(np.isfinite(dh) | np.isfinite(da),
                    np.where(np.nan_to_num(dh, nan=1e9) <= np.nan_to_num(da, nan=1e9), 0, 1), -1)
    h = np.concatenate(([0], np.cumsum(code == 0)))
    a = np.concatenate(([0], np.cumsum(code == 1)))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - window), np.minimum(n, i + window + 1)
    nh, na = h[hi] - h[lo], a[hi] - a[lo]
    out = np.where(nh + na == 0, "D", np.where(nh >= na, "H", "A"))
    mask = np.full(n, "X", dtype="<U1")
    for s, e in halves:
        mask[s:e] = out[s:e]
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage2", action="store_true", help="add the stage-2 context model")
    ap.add_argument("--oof", default=os.path.join(ROOT, "cache", "oof.pkl"))
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--test-seed", type=int, default=100)
    args = ap.parse_args()
    train_seeds = args.train_seeds[:1] if args.quick else args.train_seeds

    t0 = time.time()
    data = {}
    for g in sample_games.GAMES:
        for s in train_seeds + [args.test_seed]:
            data[(g, s)] = degraded_features(g, s)
    print("features ready in {:.0f}s".format(time.time() - t0))

    oof = {}
    for held in sample_games.GAMES:
        train = [data[(g, s)] + (g,) for g in sample_games.GAMES if g != held
                 for s in train_seeds]
        t = time.time()
        models = FrameModels(stage2=args.stage2).fit(train)
        feats, labels, halves = data[(held, args.test_seed)]
        p_dead, p_home = models.predict(feats, halves)
        oof[held] = (p_dead, p_home, labels, halves)
        print("fold game {}: trained+predicted in {:.0f}s".format(held, time.time() - t))

    with open(args.oof, "wb") as fh:
        pickle.dump(oof, fh, protocol=4)

    # frame-level diagnostics
    for g, (p_dead, p_home, labels, halves) in oof.items():
        g_mask = labels != "X"
        dead_acc = np.mean((p_dead[g_mask] > 0.5) == (labels[g_mask] == "D"))
        alive = np.isin(labels, ["H", "A"])
        team_acc = np.mean((p_home[alive] > 0.5) == (labels[alive] == "H"))
        base = nearest_baseline(data[(g, args.test_seed)][0], halves, len(labels))
        b = score(labels, base)
        print("game {}: dead acc {:.3f}  team acc (alive) {:.3f}   baseline50 spellF1 {:.3f} "
              "macroF1 {:.3f}".format(g, dead_acc, team_acc, b[0], b[1]))

    grid = dict(a_team=[0.5, 1.0, 2.0], switch=[4, 8, 16, 32, 64],
                stop=[4, 8, 16, 32], restart=[4, 8, 16, 32])
    if args.quick:
        grid = dict(a_team=[1.0], switch=[8, 16, 32, 64], stop=[8, 16, 32], restart=[8, 16, 32])
    rows = []
    keys = list(grid)
    for combo in itertools.product(*[grid[k] for k in keys]):
        prm = dict(zip(keys, combo))
        per = []
        for g, (p_dead, p_home, labels, halves) in oof.items():
            pred = decode.decode(p_dead, p_home, halves, len(labels), prm)
            per.append(score(labels, pred))
        rows.append((np.mean([p[0] for p in per]), np.mean([p[1] for p in per]), prm, per))
    rows.sort(key=lambda r: -r[0])
    print("\ntop decoder settings (mean over held-out games)")
    for spell, macro, prm, per in rows[:10]:
        detail = "  ".join("g{} {:.3f}/{:.3f} ({}/{})".format(g, p[0], p[1], p[2], p[3])
                           for g, p in zip(oof, per))
        print("spellF1 {:.4f}  macroF1 {:.4f}  {}   {}".format(spell, macro, prm, detail))


if __name__ == "__main__":
    main()

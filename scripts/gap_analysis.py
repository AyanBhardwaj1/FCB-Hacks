"""How well can the ball be placed inside tracker gaps? Compares estimators against the truth.

Degrades the sample games exactly as evaluation does, then, on live frames where the degraded
ball is missing (after the usual short-gap interpolation) but the clean data has it, compares
ball estimates with the true position:

  hold     the last sighting
  interp   straight line between the last and the next sighting
  snap     the player nearest to the straight-line estimate
  imputed  mtah.features.impute_ball (if present)

Reported per gap length: median error in metres, and how often the team nearest the estimate is
the team nearest the true ball -- the quantity the possession features actually consume.

    python scripts/gap_analysis.py                 # design seeds 101-103, default scenario
"""

import os
import sys
import warnings
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from common import SCENARIOS  # noqa: E402
from mtah import features, sample_games  # noqa: E402
from mtah.io import AWAY, HOME  # noqa: E402

BUCKETS = ((0.4, 1), (1, 2), (2, 4), (4, 8), (8, 1e9))


def nearest_team(x, y, bx, by, team):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.hypot(x - bx[:, None], y - by[:, None])
        dh = np.nanmin(np.where(team == HOME, d, np.nan), axis=1)
        da = np.nanmin(np.where(team == AWAY, d, np.nan), axis=1)
    return np.where(np.isfinite(dh) | np.isfinite(da), np.where(np.nan_to_num(dh, nan=1e9) <=
                                                                 np.nan_to_num(da, nan=1e9), 0, 1),
                    -1), d


def estimates(deg):
    fps = deg.fps
    bx, by, seen, known = features.clean_ball(deg.ball_x, deg.ball_y, fps)
    T = len(bx)
    idx = np.where(known, np.arange(T), -1)
    last = np.maximum.accumulate(idx)
    ridx = np.where(known[::-1], np.arange(T), -1)
    nxt = (T - 1 - np.maximum.accumulate(ridx))[::-1]
    nxt = np.where(np.maximum.accumulate(ridx)[::-1] >= 0, nxt, -1)
    ok = (last >= 0) & (nxt >= 0) & (nxt < T)
    hold_x = np.where(last >= 0, bx[np.maximum(last, 0)], np.nan)
    hold_y = np.where(last >= 0, by[np.maximum(last, 0)], np.nan)
    w = np.where(ok, (np.arange(T) - last) / np.maximum(nxt - last, 1), np.nan)
    ix = np.where(ok, bx[np.maximum(last, 0)] * (1 - w) + bx[np.clip(nxt, 0, T - 1)] * w, np.nan)
    iy = np.where(ok, by[np.maximum(last, 0)] * (1 - w) + by[np.clip(nxt, 0, T - 1)] * w, np.nan)
    _, d = nearest_team(deg.x.astype(float), deg.y.astype(float), ix, iy, deg.team)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        j = np.nanargmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    sx = deg.x[np.arange(T), j].astype(float)
    sy = deg.y[np.arange(T), j].astype(float)
    out = {"hold": (hold_x, hold_y), "interp": (ix, iy), "snap": (sx, sy)}
    if hasattr(features, "impute_ball"):
        mx, my, _ = features.impute_ball(deg, bx, by, known)
        out["imputed"] = (mx, my)
    gap_len = features.run_length(~known) / fps
    return out, known, gap_len


def main(seeds=(101, 102, 103), scenario="default"):
    err = defaultdict(lambda: defaultdict(list))
    agree = defaultdict(lambda: defaultdict(list))
    for game in sample_games.GAMES:
        match, events = sample_games.load_game(game)
        labels = sample_games.labels_from_events(events, match.n_frames, match.halves)
        for seed in seeds:
            deg = sample_games.degrade(match, labels, seed=seed, **SCENARIOS[scenario])
            est, known, gap_len = estimates(deg)
            tx, ty = match.ball_x.astype(float), match.ball_y.astype(float)
            sel = (~known) & np.isin(labels, ["H", "A"]) & np.isfinite(tx)
            true_team, _ = nearest_team(deg.x.astype(float), deg.y.astype(float), tx, ty, deg.team)
            for name, (ex, ey) in est.items():
                good = sel & np.isfinite(ex)
                e = np.hypot(ex - tx, ey - ty)
                t_est, _ = nearest_team(deg.x.astype(float), deg.y.astype(float), ex, ey, deg.team)
                for lo, hi in BUCKETS:
                    b = good & (gap_len >= lo) & (gap_len < hi)
                    err[name][(lo, hi)].append(e[b])
                    agree[name][(lo, hi)].append(t_est[b] == true_team[b])
    names = list(err)
    print("live frames inside tracker gaps (design seeds {}), median error m / nearest-team "
          "agreement".format(list(seeds)))
    print("   gap length   frames  " + "  ".join("{:>16}".format(n) for n in names))
    for lo, hi in BUCKETS:
        n = sum(len(a) for a in err[names[0]][(lo, hi)])
        cells = []
        for name in names:
            e = np.concatenate(err[name][(lo, hi)])
            a = np.concatenate(agree[name][(lo, hi)])
            cells.append("{:>6.1f} m {:>6.0%}".format(np.median(e), a.mean()) if len(e) else "-")
        label = "{:g}-{:g}s".format(lo, hi) if hi < 1e8 else ">{:g}s".format(lo)
        print("   {:<10} {:>8}  ".format(label, n) + "  ".join("{:>16}".format(c) for c in cells))


if __name__ == "__main__":
    main()

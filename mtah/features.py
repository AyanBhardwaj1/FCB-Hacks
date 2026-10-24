"""Per-frame features from team-level tracking.

Everything here is a function of the tracking alone (positions, speeds, team of each slot, the
ball) plus the published half boundaries. No labels, no Smart Tagging -- those enter later.

Two feature sets come out:

  team block   features seen from one team's point of view ("own" vs "opp"), in that team's
               attacking frame. The possession model reads block(H) to say "H has it", and
               block(A) to say "A has it"; averaging the two makes it exactly antisymmetric.
  shared       features with no team in them (activity, ball state, kickoff shape), for the
               dead-ball model.

Missing values stay NaN: the gradient-boosted models downstream route them natively, and "the
ball is not there" is itself information.
"""

import warnings

import numpy as np

from .io import AWAY, HOME

# Params
BALL_MAX_SPEED = 40.0     # m/s; faster jumps between detections are tracking spikes
BALL_INTERP_MAX = 10      # frames; shorter ball gaps are linearly interpolated
CTRL_RADIUS = 1.5         # m; a player this close to a slow ball is treated as touching it
CTRL_MARGIN = 0.5         # m; ...and the nearest opponent must be this much further away
CTRL_MAX_BALL_SPEED = 12  # m/s; a ball faster than this is in flight, not under control
CAP_S = 15.0              # s; time-since / time-until features are capped here
SCALES_S = (1.0, 3.0)     # s; centred rolling-mean windows added for the main features


# -- small array helpers ---------------------------------------------------------------------------

def _nan_reduce(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def rolling_mean(a, half_width):
    """Centred NaN-aware moving average over +/- half_width frames (NaN where all missing)."""
    if half_width <= 0:
        return a.astype(np.float64)
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, a, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(a))
    lo = np.maximum(0, i - half_width)
    hi = np.minimum(len(a), i + half_width + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out[cnt == 0] = np.nan
    return out


def run_length(mask):
    """For each True frame, the length of the True run it belongs to; 0 elsewhere."""
    m = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(m), dtype=np.float64)
    for a, b in zip(starts, stops):
        out[a:b] = b - a
    return out


def frames_since(mask, cap):
    """Frames since the last True at or before each frame (cap where there is none)."""
    idx = np.where(mask, np.arange(len(mask)), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(mask)) - last, cap).astype(np.float64)


def frames_until(mask, cap):
    return frames_since(mask[::-1], cap)[::-1]


# -- ball ---------------------------------------------------------------------------------------

def clean_ball(bx, by, fps, max_speed=BALL_MAX_SPEED, interp_max=BALL_INTERP_MAX):
    """(x, y, seen, known): spikes removed, short gaps interpolated.

    seen  -- a real detection survived cleaning on this frame
    known -- seen, or inside a gap short enough to interpolate
    """
    bx = np.asarray(bx, dtype=np.float64).copy()
    by = np.asarray(by, dtype=np.float64).copy()
    for _ in range(2):
        idx = np.flatnonzero(np.isfinite(bx))
        if len(idx) < 3:
            break
        px, py = bx[idx], by[idx]
        dt = np.diff(idx).astype(np.float64) / fps
        step = np.hypot(np.diff(px), np.diff(py)) / dt
        v_in, v_out = step[:-1], step[1:]
        skip = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) / (dt[:-1] + dt[1:])
        spike = (v_in > max_speed) & (v_out > max_speed) & (skip < max_speed)
        bad = idx[1:-1][spike]
        bx[bad] = np.nan
        by[bad] = np.nan
    seen = np.isfinite(bx)

    known = seen.copy()
    idx = np.flatnonzero(seen)
    if len(idx) > 1:
        gaps = np.diff(idx)
        for k in np.flatnonzero((gaps > 1) & (gaps <= interp_max + 1)):
            a, b = idx[k], idx[k + 1]
            w = np.arange(1, b - a) / float(b - a)
            bx[a + 1:b] = bx[a] * (1 - w) + bx[b] * w
            by[a + 1:b] = by[a] * (1 - w) + by[b] * w
            known[a + 1:b] = True
    return bx, by, seen, known


def ball_speed(bx, by, fps, k=2):
    """|velocity| from a central difference over +/- k frames; NaN where either end is missing."""
    v = np.full(len(bx), np.nan)
    dx = bx[2 * k:] - bx[:-2 * k]
    dy = by[2 * k:] - by[:-2 * k]
    v[k:-k] = np.hypot(dx, dy) * fps / (2 * k)
    return v


def player_speed(match):
    """The ATD speed channel where it exists, else a finite difference within each slot."""
    s = match.speed.astype(np.float64)
    if np.isfinite(s).mean() > 0.5 * np.isfinite(match.x).mean():
        return np.clip(s, 0, 12)
    fps = match.fps
    x, y = match.x.astype(np.float64), match.y.astype(np.float64)
    v = np.full_like(x, np.nan)
    v[1:-1] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) * fps / 2
    return np.clip(v, 0, 12)


# -- geometry -----------------------------------------------------------------------------------

def attack_directions(match):
    """{(half_index): (dir_home, dir_away)}: +1 if the team attacks towards +x in that half.

    Read off the data: over a whole half each team's mean x sits in its own half of the pitch.
    """
    out = {}
    for h, (a, b) in enumerate(match.halves):
        x = match.x[a:b]
        mh = np.nanmean(x[:, match.team == HOME])
        ma = np.nanmean(x[:, match.team == AWAY])
        dh = 1 if mh < ma else -1
        out[h] = (dh, -dh)
    return out


def per_frame_direction(match):
    """[T, 2] attack direction of (home, away) on every frame (nearest half outside halves)."""
    T = match.n_frames
    dirs = attack_directions(match)
    out = np.zeros((T, 2), dtype=np.float64)
    mids = [(a + b) / 2 for a, b in match.halves]
    t = np.arange(T)
    nearest = np.argmin(np.abs(t[:, None] - np.array(mids)[None, :]), axis=1)
    for h in dirs:
        out[nearest == h] = dirs[h]
    return out


def _sorted_stats(v):
    """Per-row count, min, 2nd-smallest, max, 2nd-largest of a NaN-padded [T, n] array."""
    s = np.sort(v, axis=1)                       # NaNs go last
    n = np.isfinite(v).sum(axis=1)
    rows = np.arange(len(v))
    def at(k):
        k = np.clip(k, 0, v.shape[1] - 1)
        out = s[rows, k]
        return out
    mn, mn2 = at(np.zeros_like(n)), at(np.ones_like(n))
    mx, mx2 = at(n - 1), at(n - 2)
    bad = n == 0
    for arr in (mn, mn2, mx, mx2):
        arr[bad] = np.nan
    mn2[n < 2] = np.nan
    mx2[n < 2] = np.nan
    return n, mn, mn2, mx, mx2


# -- the feature builder -------------------------------------------------------------------------

def compute(match):
    """{'team': {HOME: dict, AWAY: dict}, 'shared': dict} of [T] float arrays."""
    fps = match.fps
    T = match.n_frames
    L, W = match.pitch
    x, y = match.x.astype(np.float64), match.y.astype(np.float64)
    spd = player_speed(match)
    bx, by, seen, known = clean_ball(match.ball_x, match.ball_y, fps)
    vb = ball_speed(bx, by, fps)
    dirs = per_frame_direction(match)

    def smooth(a, s):
        return rolling_mean(a, int(round(s * fps / 2)))

    shared = {}
    # ball state
    shared["ball_seen"] = seen.astype(np.float64)
    shared["ball_known"] = known.astype(np.float64)
    miss = ~seen
    shared["ball_miss_run_s"] = run_length(miss) / fps
    shared["ball_since_seen_s"] = frames_since(seen, CAP_S * fps) / fps
    shared["ball_until_seen_s"] = frames_until(seen, CAP_S * fps) / fps
    vb_known = np.where(known, vb, np.nan)
    shared["ball_speed"] = vb_known
    for s in SCALES_S:
        shared["ball_seen_m{}".format(s)] = smooth(seen.astype(np.float64), s)
        shared["ball_speed_m{}".format(s)] = smooth(vb_known, s)
    shared["ball_still_m1"] = smooth(np.where(np.isfinite(vb_known),
                                              (vb_known < 0.7).astype(np.float64), np.nan), 1.0)
    oob = np.fmax.reduce([-bx, bx - L, -by, by - W])
    shared["ball_oob"] = oob                                   # metres outside, negative inside
    shared["ball_oob_m1"] = smooth(np.where(known, (oob > 0.3).astype(np.float64), np.nan), 1.0)
    shared["ball_oob_m3"] = smooth(np.where(known, (oob > 0.3).astype(np.float64), np.nan), 3.0)
    shared["ball_to_touchline"] = np.minimum(by, W - by)
    shared["ball_to_goalline"] = np.minimum(bx, L - bx)
    shared["ball_to_centre"] = np.hypot(bx - L / 2, by - W / 2)
    corner = np.minimum.reduce([np.hypot(bx - cx, by - cy) for cx in (0, L) for cy in (0, W)])
    shared["ball_to_corner"] = corner
    # last known ball position carried through gaps, for the dead model's spatial cues
    idx = np.where(known, np.arange(T), -1)
    last = np.maximum.accumulate(idx)
    lx = np.where(last >= 0, bx[np.maximum(last, 0)], np.nan)
    ly = np.where(last >= 0, by[np.maximum(last, 0)], np.nan)
    shared["last_ball_oob"] = np.fmax.reduce([-lx, lx - L, -ly, ly - W])

    # activity
    n_tracked = np.isfinite(x).sum(axis=1).astype(np.float64)
    shared["n_tracked"] = n_tracked
    mean_v = _nan_reduce(np.nanmean, spd)
    for s in (1.0, 3.0, 6.0):
        shared["speed_mean_m{}".format(s)] = smooth(mean_v, s)
    graded = np.zeros(T, dtype=bool)
    for a, b in match.halves:
        graded[a:b] = True
    ref = np.nanmedian(shared["speed_mean_m3.0"][graded]) if graded.any() else 1.0
    for s in (1.0, 3.0, 6.0):
        shared["speed_ratio_m{}".format(s)] = shared["speed_mean_m{}".format(s)] / ref
    fast = np.where(np.isfinite(spd), (spd > 4.0).astype(np.float64), np.nan)
    shared["frac_fast_m3"] = smooth(_nan_reduce(np.nanmean, fast), 3.0)
    shared["speed_max_m1"] = smooth(_nan_reduce(np.nanmax, spd), 1.0)
    # how much the whole picture is changing: mean |centroid velocity| of both teams
    # (filled below once per-team centroids exist)

    # distances to the ball
    d = np.hypot(x - bx[:, None], y - by[:, None])
    team_feats = {}
    ctrl = {}
    per_team = {}
    for k in (HOME, AWAY):
        cols = match.team == k
        dk = d[:, cols]
        dmin = _nan_reduce(np.nanmin, dk)
        dmin = np.where(known, dmin, np.nan)
        n2 = np.where(known, (dk < 2.0).sum(axis=1), np.nan).astype(np.float64)
        n5 = np.where(known, (dk < 5.0).sum(axis=1), np.nan).astype(np.float64)
        dirk = dirs[:, k]
        xa = (x[:, cols] - L / 2) * dirk[:, None]           # +: towards the goal k attacks
        yk = y[:, cols]
        n, xmin, xmin2, xmax, xmax2 = _sorted_stats(xa)
        cx = _nan_reduce(np.nanmean, xa)
        cy = _nan_reduce(np.nanmean, yk)
        sx = _nan_reduce(np.nanstd, xa)
        sy = _nan_reduce(np.nanstd, yk)
        vk = _nan_reduce(np.nanmean, spd[:, cols])
        in_opp = (xa > 0).sum(axis=1).astype(np.float64)
        in_own_half = np.where(n > 0, (xa < 0.5).sum(axis=1) / np.maximum(n, 1), np.nan)
        per_team[k] = dict(dmin=dmin, n2=n2, n5=n5, cx=cx, cy=cy, sx=sx, sy=sy, v=vk,
                           xmin2=xmin2, xmax=xmax, xmax2=xmax2, in_opp=in_opp,
                           in_own_half=in_own_half, n=n.astype(np.float64), dir=dirk)
    for k in (HOME, AWAY):
        o = AWAY if k == HOME else HOME
        me, op = per_team[k], per_team[o]
        c = (me["dmin"] < CTRL_RADIUS) & (op["dmin"] > me["dmin"] + CTRL_MARGIN) & seen & \
            ~(vb_known > CTRL_MAX_BALL_SPEED)
        ctrl[k] = c

    # per-half time-since / time-until a team was last seen controlling the ball
    since = {k: np.full(T, CAP_S) for k in (HOME, AWAY)}
    until = {k: np.full(T, CAP_S) for k in (HOME, AWAY)}
    for a, b in match.halves:
        for k in (HOME, AWAY):
            since[k][a:b] = frames_since(ctrl[k][a:b], CAP_S * fps) / fps
            until[k][a:b] = frames_until(ctrl[k][a:b], CAP_S * fps) / fps

    for k in (HOME, AWAY):
        o = AWAY if k == HOME else HOME
        me, op = per_team[k], per_team[o]
        f = {}
        f["own_d"], f["opp_d"] = me["dmin"], op["dmin"]
        f["d_diff"] = op["dmin"] - me["dmin"]
        for s in (0.5,) + SCALES_S:
            f["d_diff_m{}".format(s)] = smooth(f["d_diff"], s)
            f["own_ctrl_m{}".format(s)] = smooth(ctrl[k].astype(np.float64), s)
            f["opp_ctrl_m{}".format(s)] = smooth(ctrl[o].astype(np.float64), s)
        f["own_n2"], f["opp_n2"] = me["n2"], op["n2"]
        f["own_n5"], f["opp_n5"] = me["n5"], op["n5"]
        f["own_since_ctrl"], f["opp_since_ctrl"] = since[k], since[o]
        f["own_until_ctrl"], f["opp_until_ctrl"] = until[k], until[o]
        f["since_ctrl_diff"] = since[o] - since[k]
        f["until_ctrl_diff"] = until[o] - until[k]
        # shapes, each in its own attacking frame
        for name in ("cx", "sx", "sy", "xmin2", "xmax", "in_opp", "v", "n"):
            f["own_" + name] = me[name]
            f["opp_" + name] = op[name]
        for s in SCALES_S:
            f["own_cx_m{}".format(s)] = smooth(me["cx"], s)
            f["opp_cx_m{}".format(s)] = smooth(op["cx"], s)
            f["own_sx_m{}".format(s)] = smooth(me["sx"], s)
            f["opp_sx_m{}".format(s)] = smooth(op["sx"], s)
        # forward velocity of each block, in its own attacking direction
        for name, team in (("own", me), ("opp", op)):
            cxs = smooth(team["cx"], 0.5)
            vel = np.full(T, np.nan)
            vel[fps:-fps] = (cxs[2 * fps:] - cxs[:-2 * fps]) / 2.0
            f[name + "_fwd_v"] = vel
            f[name + "_fwd_v_m3"] = smooth(vel, 3.0)
        # where the ball is, in this team's attacking frame
        f["ball_xa"] = (bx - L / 2) * me["dir"]
        f["ball_xa_m1"] = smooth(f["ball_xa"], 1.0)
        f["ball_minus_own_cx"] = f["ball_xa"] - me["cx"]
        f["ball_minus_opp_cx"] = f["ball_xa"] + op["cx"]     # opp centroid is -cx in our frame
        team_feats[k] = f

    # kickoff shape: both teams entirely in their own halves, ball near the centre spot
    ko = np.fmin(per_team[HOME]["in_own_half"], per_team[AWAY]["in_own_half"])
    shared["kickoff_shape"] = ko
    shared["kickoff_shape_m3"] = smooth(ko, 3.0)
    shared["dmin_any"] = np.fmin(per_team[HOME]["dmin"], per_team[AWAY]["dmin"])
    shared["n5_any"] = per_team[HOME]["n5"] + per_team[AWAY]["n5"]
    shared["ctrl_any_m1"] = smooth((ctrl[HOME] | ctrl[AWAY]).astype(np.float64), 1.0)
    shared["since_ctrl_any"] = np.minimum(since[HOME], since[AWAY])
    shared["until_ctrl_any"] = np.minimum(until[HOME], until[AWAY])
    shared["spread_sum"] = per_team[HOME]["sx"] + per_team[AWAY]["sx"]
    cvel = []
    for k in (HOME, AWAY):
        cxs = smooth(per_team[k]["cx"], 0.5)
        v = np.full(T, np.nan)
        v[fps:-fps] = np.abs(cxs[2 * fps:] - cxs[:-2 * fps]) / 2.0
        cvel.append(v)
    shared["block_motion_m3"] = smooth(cvel[0] + cvel[1], 3.0)
    return {"team": team_feats, "shared": shared}


def matrices(feats):
    """(names_team, X_home, X_away, names_shared, X_shared) as float32 matrices."""
    names_team = sorted(feats["team"][HOME])
    names_shared = sorted(feats["shared"])
    xh = np.column_stack([feats["team"][HOME][n] for n in names_team]).astype(np.float32)
    xa = np.column_stack([feats["team"][AWAY][n] for n in names_team]).astype(np.float32)
    xs = np.column_stack([feats["shared"][n] for n in names_shared]).astype(np.float32)
    return names_team, xh, xa, names_shared, xs

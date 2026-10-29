"""Metrica's public sample games: training and validation data from *other* matches.

The challenge rules allow labelled data from other games in any amount, provided the organisers
can access it. Metrica publishes three anonymised matches with tracking and event data at
https://github.com/metrica-sports/sample-data -- the same provider and, for Sample Game 3, the
same FIFA/EPTS format as the challenge ATD. Nothing here touches the competition match.

Two things turn them into training data for this task:

  labels_from_events()  frame-level H / A / D derived mechanically from the event stream
  degrade()             make the clean sample tracking look like the challenge's automatic ATD:
                        the ball missing ~41% of the time in bursts, players dropping out, track
                        fragments, positional noise

Home is the team the files call "Home" (games 1-2) or the first team in the metadata (game 3).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from .io import AWAY, HOME, Match, load_epts, load_metrica_csv

H, A, D, X = "H", "A", "D", "X"

DEFAULT_DIR = os.environ.get("MTAH_SAMPLE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "sample-data", "data")

GAMES = (1, 2, 3)

SAMPLE_REPO = "https://github.com/metrica-sports/sample-data"
# Pinned so a later upstream commit cannot change the retrained model or the stand-in bundle.
SAMPLE_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"

# Event types after which the ball is out of play. Subtype strings are matched as substrings.
DEAD_TYPES = {"BALL OUT", "FAULT RECEIVED"}
DEAD_SUBTYPES = {"SHOT": ("OUT", "GOAL"), "BALL LOST": ("OFFSIDE", "HAND BALL", "END HALF"),
                 "CHALLENGE": ("FAULT",)}
# Events that put the ball back in play, and events that show a team controlling the ball.
RESTART_TYPES = {"SET PIECE", "PASS", "CARRY", "SHOT"}
CONTROL_TYPES = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST"}


def _events_csv(path):
    raw = pd.read_csv(path)
    out = []
    for row in raw.itertuples(index=False):
        team = HOME if str(row.Team).strip().lower() == "home" else AWAY
        start, end = int(row[4]), int(row[6])
        out.append({"team": team, "type": str(row.Type).upper(),
                    "subtype": "" if pd.isna(row.Subtype) else str(row.Subtype).upper(),
                    "period": int(row.Period), "start": start - 1,
                    "end": max(end, start) - 1})
    return out


def _events_json(path, home_id):
    data = json.load(open(path))["data"]
    out = []
    for ev in data:
        sub = ev.get("subtypes")
        if isinstance(sub, dict):
            subtype = sub.get("name") or ""
        elif isinstance(sub, list):
            subtype = "-".join(s.get("name", "") for s in sub)
        else:
            subtype = ""
        start = int(ev["start"]["frame"])
        end = int(ev["end"]["frame"] or start)
        out.append({"team": HOME if ev["team"]["id"] == home_id else AWAY,
                    "type": ev["type"]["name"].upper(), "subtype": subtype.upper(),
                    "period": int(ev["period"]), "start": start - 1,
                    "end": max(end, start) - 1})
    return out


def ensure_data(data_dir=None):
    """Clone the sample data at the pinned commit into data_dir's parent if it is missing
    (~230 MB, needs git), and refuse a checkout at any other commit."""
    data_dir = data_dir or DEFAULT_DIR
    target = os.path.dirname(data_dir)
    if not os.path.isdir(data_dir):
        print("fetching Metrica sample data into {} (~230 MB) ...".format(target))
        subprocess.check_call(["git", "clone", SAMPLE_REPO, target])
        subprocess.check_call(["git", "-C", target, "checkout", "-q", SAMPLE_COMMIT])
    head = subprocess.check_output(["git", "-C", target, "rev-parse", "HEAD"]).decode().strip()
    if head != SAMPLE_COMMIT:
        sys.exit("sample data at {} is commit {}, expected {} -- run "
                 "`git -C {} checkout {}`".format(target, head, SAMPLE_COMMIT, target,
                                                  SAMPLE_COMMIT))


def load_game(game, data_dir=None):
    """(Match, events) for one sample game, clean tracking, halves set from the source."""
    root = os.path.join(data_dir or DEFAULT_DIR, "Sample_Game_{}".format(game))
    stem = os.path.join(root, "Sample_Game_{}".format(game))
    if game in (1, 2):
        match = load_metrica_csv(stem + "_RawTrackingData_Home_Team.csv",
                                 stem + "_RawTrackingData_Away_Team.csv",
                                 name="sample{}".format(game))
        events = _events_csv(stem + "_RawEventsData.csv")
    else:
        meta = stem + "_metadata.xml"
        match = load_epts(meta, stem + "_tracking.txt", name="sample3")
        from .io import read_epts_meta
        info = read_epts_meta(meta)
        events = _events_json(stem + "_events.json", info["teams"][0])
        # The EPTS file runs on without a halftime gap, so take the halves from the events.
        match.halves = _halves_from_events(events, match.n_frames)
    events.sort(key=lambda e: (e["start"], e["end"]))
    return match, events


def _halves_from_events(events, n_frames):
    spans = []
    for period in sorted({e["period"] for e in events}):
        frames = [e["start"] for e in events if e["period"] == period] + \
                 [e["end"] for e in events if e["period"] == period]
        spans.append((max(0, min(frames)), min(n_frames, max(frames) + 1)))
    return spans


def _is_dead_event(ev):
    if ev["type"] in DEAD_TYPES:
        return True
    for kind, subs in DEAD_SUBTYPES.items():
        if ev["type"] == kind and any(s in ev["subtype"] for s in subs):
            if "ADVANTAGE" in ev["subtype"]:
                return False
            return True
    return False


def labels_from_events(events, n_frames, halves, min_control_s=2.0, fps=25):
    """Frame labels H / A / D (X outside the halves) from an event stream.

    Possession: the team of the latest controlling event (pass, carry, set piece, shot, recovery,
    or the team losing it until the opponent's next controlling event). Challenges do not move
    possession -- a contested ball stays with the last team that had it.

    Dead ball: from the frame the ball goes out / the whistle goes (ball out, foul, offside, shot
    out or goal) until the next restart (set piece, or the first pass / carry / shot after it).
    Each half is dead until its first kick-off.

    Controlled possession: the challenge does not count "a simple touch of the ball without
    control, a deflection of a pass that does not intercept it, an aerial duel won without
    subsequent control" as a change of possession (CHALLENGE.md). So a spell only stands if the
    team completed a pass, shot or set piece in it, carried the ball for half of
    `min_control_s`, or kept it `min_control_s` seconds; any other spell is handed back to the team that had the ball before it.
    """
    labels = np.full(n_frames, X, dtype="<U1")
    for start, stop in halves:
        evs = [e for e in events if start <= e["start"] < stop]
        cur = np.full(stop - start, D, dtype="<U1")
        team = None
        timeline = []       # (frame, kind, team)
        for e in evs:
            if e["type"] in CONTROL_TYPES:
                timeline.append((e["start"], "control", e["team"], e))
            if _is_dead_event(e):
                frame = e["end"] if e["type"] in ("BALL OUT", "SHOT") else e["start"]
                timeline.append((frame, "dead", e["team"], e))
        timeline.sort(key=lambda t: (t[0], 0 if t[1] == "control" else 1))

        alive = False
        last = start
        for frame, kind, ev_team, ev in timeline:
            frame = min(max(frame, start), stop)
            # fill [last, frame) with the current state
            cur[last - start:frame - start] = (H if team == HOME else A) if alive and \
                team is not None else D
            last = frame
            if kind == "dead":
                alive = False
            else:
                if not alive:
                    if ev["type"] in RESTART_TYPES:
                        alive = True
                        team = ev_team
                    # a recovery or ball-lost while dead does not restart play
                else:
                    if ev["type"] == "BALL LOST":
                        team = ev_team      # still theirs until someone else controls it
                    else:
                        team = ev_team
        cur[last - start:] = (H if team == HOME else A) if alive and team is not None else D
        labels[start:stop] = _enforce_control(cur, start, evs, int(min_control_s * fps))
    return labels


def _enforce_control(cur, offset, events, min_frames):
    """Relabel uncontrolled possession spells to the previous possessing team (see above)."""
    cur = cur.copy()
    controlled = {H: np.zeros(len(cur), dtype=bool), A: np.zeros(len(cur), dtype=bool)}
    for e in events:
        long_carry = e["type"] == "CARRY" and e["end"] - e["start"] >= min_frames // 2
        if e["type"] in ("PASS", "SHOT", "SET PIECE") or long_carry:
            f = e["start"] - offset
            if 0 <= f < len(cur):
                controlled[H if e["team"] == HOME else A][f] = True
    changed = True
    while changed:
        changed = False
        breaks = np.flatnonzero(cur[1:] != cur[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(cur)]))
        segs = [(int(a), int(b), cur[a]) for a, b in zip(bounds[:-1], bounds[1:])]
        for i in range(1, len(segs)):
            a, b, lab = segs[i]
            prev = segs[i - 1][2]
            if lab == D or prev == D or prev == lab:
                continue
            if b - a >= min_frames or controlled[lab][a:b].any():
                continue
            cur[a:b] = prev
            changed = True
            break
    return cur


# -- degradation ---------------------------------------------------------------------------------

def _burst_mask(n, rate, median, sigma, rng, weights=None):
    """Boolean mask of missing bursts: starts as a Poisson process (rate per frame, optionally
    modulated by `weights`), lengths log-normal with the given median (frames) and sigma."""
    p = rate * (weights if weights is not None else 1.0)
    starts = np.flatnonzero(rng.random(n) < np.clip(p, 0, 1))
    lengths = np.maximum(1, np.round(median * np.exp(sigma * rng.standard_normal(len(starts)))))
    mask = np.zeros(n + 1, dtype=np.int32)
    ends = np.minimum(starts + lengths.astype(np.int64), n)
    np.add.at(mask, starts, 1)
    np.add.at(mask, ends, -1)
    return np.cumsum(mask[:n]) > 0


def fill_dead_ball(match, labels, seed=0):
    """Ball positions for dead-ball frames, where the clean sample tracking has none.

    The sample data stops tracking the ball the moment it goes out of play, so used as-is it would
    teach a model that "no ball" means "dead ball" -- false for the challenge ATD, which loses the
    ball on 41% of graded frames, mostly in play, and sometimes keeps seeing it while play is
    stopped. Each dead stretch gets a plausible path instead: the ball drifts on from where it
    was last seen (usually over a line), is carried to where play restarts, and sits still there
    for the last few seconds before the restart, as it does for a set piece.
    """
    rng = np.random.default_rng(seed)
    bx = match.ball_x.astype(np.float64).copy()
    by = match.ball_y.astype(np.float64).copy()
    T, fps = len(bx), match.fps
    dead = labels == D
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        seen_before = np.flatnonzero(np.isfinite(match.ball_x[max(0, a - 10 * fps):a]))
        seen_after = np.flatnonzero(np.isfinite(match.ball_x[b:min(T, b + 10 * fps)]))
        if len(seen_after):
            r = b + seen_after[0]
            end_xy = np.array([match.ball_x[r], match.ball_y[r]], dtype=np.float64)
        else:
            end_xy = np.array(match.pitch, dtype=np.float64) / 2
        if len(seen_before):
            r = max(0, a - 10 * fps) + seen_before[-1]
            start_xy = np.array([match.ball_x[r], match.ball_y[r]], dtype=np.float64)
            prev = max(0, r - 3)
            vel = (start_xy - np.array([match.ball_x[prev], match.ball_y[prev]],
                                       dtype=np.float64)) / max(1, r - prev)
            vel = np.nan_to_num(vel)
        else:
            start_xy, vel = end_xy.copy(), np.zeros(2)
        n = b - a
        t = np.arange(n, dtype=np.float64)
        # every stretch gets its own shape, so no feature can learn one template's timing:
        # how fast the ball slows, how long it drifts, and whether it sits still before the
        # restart at all (it does not when it is picked up and thrown in straight away)
        tau = rng.uniform(5.0, 25.0)
        drift_n = min(n, int(rng.uniform(0.5, 2.5) * fps))
        still_s = 0.0 if rng.random() < 0.3 else rng.uniform(1.0, 4.0)
        still_n = min(n - drift_n, int(still_s * fps))
        carry_n = n - drift_n - still_n
        path = np.zeros((n, 2))
        decay = (1 - np.exp(-t[:drift_n] / tau)) * tau
        path[:drift_n] = start_xy + vel * decay[:, None]
        mid = path[drift_n - 1] if drift_n else start_xy
        if carry_n:
            w = (np.arange(1, carry_n + 1) / carry_n)[:, None]
            path[drift_n:drift_n + carry_n] = mid * (1 - w) + end_xy * w
        path[drift_n + carry_n:] = end_xy
        # no noise here: degrade() adds the same tracking noise to every frame, and extra
        # jitter on dead frames only would leak the labels into ball speed and touches
        fill = ~np.isfinite(bx[a:b])
        bx[a:b][fill] = path[fill, 0]
        by[a:b][fill] = path[fill, 1]
    # The clean data also drops the ball for a moment around some stoppages (1-2% of live
    # frames, clustered next to dead stretches). Interpolate those too, so that every gap in the
    # degraded ball comes from degrade()'s label-free dropout and none marks a label boundary.
    for a, b in match.halves:
        for arr in (bx, by):
            seg = arr[a:b]
            ok = np.isfinite(seg)
            if ok.sum() >= 2 and not ok.all():
                idx = np.arange(len(seg))
                seg[~ok] = np.interp(idx[~ok], idx[ok], seg[ok])
    out = Match(**{k: getattr(match, k) for k in match.__dataclass_fields__})
    out.ball_x, out.ball_y = bx.astype(np.float32), by.astype(np.float32)
    return out


def _solve_burst_mask(n, target, region, weights, seed, median=18, sigma=1.1):
    """Burst mask whose missing share inside `region` is `target` (rate solved iteratively)."""
    rate = 0.01
    miss = np.zeros(n, dtype=bool)
    for _ in range(40):
        miss = _burst_mask(n, rate, median, sigma, np.random.default_rng(seed), weights)
        share = np.mean(miss[region]) if region.any() else target
        if abs(share - target) < 0.004:
            break
        rate *= np.clip(target / max(share, 1e-6), 0.5, 2.0) ** 1.2
    return miss & region


# Ranges the degradation draws from when a parameter is not given. The real ATD's noise level
# is unknown, so training data spans a range around the best guess rather than one value: a
# model trained at a single noise level learns thresholds (a "still" ball, a touch) that break
# when the real data is noisier. The challenge publishes the ball-missing share (41%).
PARAM_RANGES = {
    "ball_missing": (0.36, 0.46),
    "ball_noise": (0.15, 0.50),           # m, per-frame ball jitter
    "player_noise": (0.05, 0.25),         # m
    "player_missing": (0.04, 0.12),       # share of player frames lost in bursts
    "spike_rate": (0.001, 0.005),         # false ball detections per frame
    "fragment_every_s": (15.0, 60.0),     # mean track-fragment length
}


def draw_params(seed, **given):
    """Degradation parameters for one seed: `given` values as passed, the rest drawn uniformly
    from PARAM_RANGES with a generator of their own (so drawing never shifts the noise itself)."""
    rng = np.random.default_rng([seed, 7919])
    out = {}
    for name, (lo, hi) in PARAM_RANGES.items():
        draw = float(rng.uniform(lo, hi))
        out[name] = given[name] if given.get(name) is not None else draw
    return out


def degrade(match, labels, seed=0, ball_missing=None, ball_noise=None, player_missing=None,
            player_noise=None, fragment_every_s=None, spike_rate=None):
    """A copy of `match` that looks like automatic, uncorrected tracking.

    `labels` (the event-derived H / A / D) are used only by fill_dead_ball, to invent a ball path
    where the clean data has none. That path is noise-free and then treated exactly like the
    real one, so the labels never decide how noisy the ball is or when it is lost directly.
    Ball dropout is one burst process over the whole half whose rate depends only on what a
    tracker would struggle with: a fast ball (in the air), a crowded ball, a ball off the pitch,
    a ball nobody is near. It must not depend on the labels directly: dropout clipped to alive /
    dead regions switches on and off exactly where play stops and restarts, and a model learns
    that leak instead of football. Players drop out in bursts and their tracks are cut into
    fragments packed into slots, so no slot follows one person for long. A speed channel is
    synthesised from the clean positions, because the challenge ATD has one.

    Every noise parameter left as None is drawn per seed from PARAM_RANGES (see draw_params);
    pass values to pin them, as the evaluation scenarios do.
    """
    prm = draw_params(seed, ball_missing=ball_missing, ball_noise=ball_noise,
                      player_missing=player_missing, player_noise=player_noise,
                      fragment_every_s=fragment_every_s, spike_rate=spike_rate)
    ball_missing, ball_noise = prm["ball_missing"], prm["ball_noise"]
    player_missing, player_noise = prm["player_missing"], prm["player_noise"]
    fragment_every_s, spike_rate = prm["fragment_every_s"], prm["spike_rate"]
    rng = np.random.default_rng(seed)
    match = fill_dead_ball(match, labels, seed)
    T, P = match.x.shape
    fps = match.fps

    # speed channel from the clean data
    vx = np.gradient(match.x.astype(np.float64), axis=0) * fps
    vy = np.gradient(match.y.astype(np.float64), axis=0) * fps
    speed = np.sqrt(vx ** 2 + vy ** 2)
    speed = np.clip(speed + 0.15 * rng.standard_normal(speed.shape), 0, 12)

    # ball: missing more when fast, crowded, off the pitch or unattended -- never by label
    bx, by = match.ball_x.astype(np.float64), match.ball_y.astype(np.float64)
    bv = np.nan_to_num(np.sqrt(np.gradient(bx) ** 2 + np.gradient(by) ** 2) * fps)
    d = np.sqrt((match.x - bx[:, None]) ** 2 + (match.y - by[:, None]) ** 2)
    crowd = np.nansum(d < 3.0, axis=1)
    with np.errstate(invalid="ignore"):
        nearest = np.where(np.isfinite(d).any(axis=1), np.nanmin(np.where(np.isfinite(d), d,
                                                                           np.inf), axis=1), 0)
    off_pitch = np.nan_to_num(np.fmax.reduce([-bx, bx - match.pitch[0], -by,
                                              by - match.pitch[1]])) > 1.0
    weights = 1.0 + 1.5 * (bv > 12) + 0.5 * np.clip(crowd - 1, 0, 4) + 2.0 * off_pitch + \
        1.0 * (nearest > 8.0)
    graded = np.zeros(T, dtype=bool)
    for a, b in match.halves:
        graded[a:b] = True
    miss = _solve_burst_mask(T, ball_missing, graded, weights, seed + 1)
    miss |= ~np.isfinite(bx)
    new_bx = bx + ball_noise * rng.standard_normal(T)
    new_by = by + ball_noise * rng.standard_normal(T)
    new_bx[miss] = np.nan
    new_by[miss] = np.nan
    # occasional false ball detections far from the truth
    spikes = np.flatnonzero((rng.random(T) < spike_rate) & np.isfinite(new_bx))
    new_bx[spikes] = rng.uniform(0, match.pitch[0], len(spikes))
    new_by[spikes] = rng.uniform(0, match.pitch[1], len(spikes))

    # players: dropouts, noise, fragmentation into fresh slots
    px = match.x + player_noise * rng.standard_normal((T, P))
    py = match.y + player_noise * rng.standard_normal((T, P))
    frags = []   # (start, stop, player)
    for j in range(P):
        keep = np.isfinite(px[:, j]) & ~_burst_mask(T, player_missing / 40.0, 30, 1.0, rng)
        cut = np.zeros(T, dtype=bool)
        cut[rng.integers(0, T, rng.poisson(T / (fragment_every_s * fps)))] = True
        # a fragment is a run of kept frames not crossing a cut
        brk = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(np.int8), [0]))) != 0)
        for a, b in zip(brk[::2], brk[1::2]):
            inner = a + np.flatnonzero(cut[a + 1:b]) + 1
            bounds = np.concatenate(([a], inner, [b]))
            frags.extend((int(u), int(v), j) for u, v in zip(bounds[:-1], bounds[1:]))
    slots_x, slots_y, slots_s, slot_team = _pack(frags, px, py, speed, match.team, T)
    return Match(x=slots_x, y=slots_y, speed=slots_s, team=slot_team,
                 ball_x=new_bx.astype(np.float32), ball_y=new_by.astype(np.float32),
                 pitch=match.pitch, fps=fps, halves=list(match.halves),
                 name=match.name + "_deg{}".format(seed))


def _pack(frags, px, py, speed, team, T):
    """Assign track fragments to slots, first free slot of the right team, like a tracker."""
    frags = sorted(frags)
    slot_end, slot_team, assign = [], [], []
    for a, b, j in frags:
        for s, end in enumerate(slot_end):
            if slot_team[s] == team[j] and end <= a:
                slot_end[s] = b
                assign.append(s)
                break
        else:
            slot_end.append(b)
            slot_team.append(team[j])
            assign.append(len(slot_end) - 1)
    n = len(slot_end)
    X = np.full((T, n), np.nan, dtype=np.float32)
    Y = np.full((T, n), np.nan, dtype=np.float32)
    S = np.full((T, n), np.nan, dtype=np.float32)
    for (a, b, j), s in zip(frags, assign):
        X[a:b, s] = px[a:b, j]
        Y[a:b, s] = py[a:b, j]
        S[a:b, s] = speed[a:b, j]
    return X, Y, S, np.array(slot_team, dtype=np.int8)

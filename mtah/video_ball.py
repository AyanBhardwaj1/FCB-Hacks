"""Ball detection from the tactical video, inside tracker gaps. Runs on your own machine.

The automatic tracking loses the ball on ~41% of graded frames. On the sample games, the same
pipeline scores spellF1 0.807 instead of 0.748 when the ball is almost never lost
(docs/TECHNICAL.md), so recovering the ball where the tracker did not is the largest lever left.
This module looks for it in the video, only where it is missing, and only near where it should
be:

  1. plan       which frames to search (tracker gaps longer than `min_gap`) and where: the
                straight line between the last and the next tracked sighting, with a search
                radius that grows with the time since the nearest sighting
  2. detect     in each planned frame, project that search area into the image with the frame's
                homography and look for a small, round, white blob of the size a 22 cm ball has
                at that spot; map the best candidate back to pitch metres
  3. calibrate  stretches where the tracking DOES see the ball are hidden, with lengths drawn
                from the real gaps, and searched like gaps: agreement with the tracked ball
                measures accuracy ("probe"); the same frames searched around a point the ball is
                NOT near measure false positives ("absent") -- in real gaps the ball is often
                not there at all (occluded, in the air, out of play)
  4. merge      fill tracker gaps with detections that a neighbouring detection confirms,
                never on a penalty or centre spot; the usual spike filter (features.clean_ball)
                then rejects detections implying impossible ball speeds

Classical computer vision (OpenCV), no learned weights: deterministic and inspectable, as the
rules ask. Needs `opencv-python-headless` (requirements-video.txt) and the video and
homography from the challenge bundle. Nothing here ever leaves your machine.
"""

import numpy as np

# Params
BALL_DIAMETER_M = 0.22
MIN_GAP = 10              # frames; shorter gaps are already interpolated by features.clean_ball
SEARCH_R0_M = 3.0         # m; search radius right next to a sighting...
SEARCH_GROWTH_M_S = 4.0   # ...growing this much per second away from the nearest sighting...
SEARCH_MAX_M = 15.0       # ...up to this
WHITE_V_MIN = 170         # HSV value: the ball is bright...
WHITE_S_MAX = 70          # ...and nearly colourless
AREA_LO, AREA_HI = 0.3, 4.0   # blob area relative to a ball's expected area at that spot
AREA_MIN_PX, AREA_MAX_PX = 2, 12   # absolute floor / ceiling for the upper bound, in pixels
FILL_MIN = 0.45           # blob area / bounding-box area: round-ish, not a sliver
ASPECT_MAX = 2.2
PROBE_EVERY_S = 20.0      # s; one calibration probe per this much tracked ball...
PROBE_LEN_S = 1.6         # ...hiding this long a stretch of it
MARK_R_M = 0.6            # m; candidates this close to a penalty or centre spot are markings
CONSIST_FRAMES = 3        # a detection needs another within +/- this many frames...
CONSIST_SPEED = 40.0      # ...no further than a 40 m/s ball could travel, plus 1 m
ABSENT_EXTRA_M = (1.0, 5.0)   # m; ball-absent probes sit this much beyond their search radius
# Fail closed: detections are only merged if calibration on this very video says they are good.
TRUST_MIN_PROBES = 100        # probe frames (ball present, and ball absent) needed to judge
TRUST_MIN_FOUND = 0.15        # share of ball-present probes with a detection
TRUST_MIN_WITHIN_1M = 0.80    # share of those within 1 m of the tracked ball
TRUST_MAX_FP = 0.05           # share of ball-absent probes that still "find" a ball


# -- geometry -------------------------------------------------------------------------------

def _apply(H, x, y):
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den


def to_pixels(H, x_m, y_m, pitch):
    """Pitch metres -> image pixels, for a homography mapping normalised pitch (0-1) to pixels
    (the challenge convention)."""
    return _apply(H, np.asarray(x_m, float) / pitch[0], np.asarray(y_m, float) / pitch[1])


def to_pitch(H, u, v, pitch):
    """Image pixels -> pitch metres. The inverse of to_pixels."""
    x, y = _apply(np.linalg.inv(H), np.asarray(u, float), np.asarray(v, float))
    return x * pitch[0], y * pitch[1]


def pixels_per_metre(H, x_m, y_m, pitch):
    """Local image scale at a pitch point (mean of the two axes)."""
    u0, v0 = to_pixels(H, x_m - 0.5, y_m, pitch)
    u1, v1 = to_pixels(H, x_m + 0.5, y_m, pitch)
    u2, v2 = to_pixels(H, x_m, y_m - 0.5, pitch)
    u3, v3 = to_pixels(H, x_m, y_m + 0.5, pitch)
    return 0.5 * (np.hypot(u1 - u0, v1 - v0) + np.hypot(u3 - u2, v3 - v2))


# -- detection in one frame -------------------------------------------------------------------

def _on_marking(x, y, pitch):
    """Is a pitch point on one of the white, ball-sized spots painted on every pitch?"""
    L, W = pitch
    spots = ((11.0, W / 2), (L - 11.0, W / 2), (L / 2, W / 2))
    return any(np.hypot(x - sx, y - sy) <= MARK_R_M for sx, sy in spots)

def detect_in_frame(frame_bgr, H, centre_m, radius_m, pitch):
    """(x_m, y_m, area_px) of the ball nearest `centre_m` within `radius_m`, or None.

    Candidates are bright, low-saturation blobs whose size matches a ball at that distance from
    the camera and whose shape is round-ish. Blobs touching the search window's edge are
    dropped, since their true size is unknown -- a white shirt cut by the window could otherwise
    pass for a ball.
    """
    import cv2

    cx, cy = float(centre_m[0]), float(centre_m[1])
    corners = [to_pixels(H, cx + dx, cy + dy, pitch)
               for dx in (-radius_m, radius_m) for dy in (-radius_m, radius_m)]
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    if not np.all(np.isfinite(us + vs)):
        return None
    h_img, w_img = frame_bgr.shape[:2]
    u0, u1 = max(0, int(np.floor(min(us)))), min(w_img, int(np.ceil(max(us))) + 1)
    v0, v1 = max(0, int(np.floor(min(vs)))), min(h_img, int(np.ceil(max(vs))) + 1)
    if u1 - u0 < 3 or v1 - v0 < 3:
        return None
    hsv = cv2.cvtColor(frame_bgr[v0:v1, u0:u1], cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= WHITE_V_MIN) & (hsv[:, :, 1] <= WHITE_S_MAX)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ppm = pixels_per_metre(H, cx, cy, pitch)
    expected = np.pi * (0.5 * BALL_DIAMETER_M * ppm) ** 2
    lo = max(AREA_MIN_PX, AREA_LO * expected)
    hi = max(AREA_MAX_PX, AREA_HI * expected)
    best = None
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if not lo <= area <= hi:
            continue
        if x == 0 or y == 0 or x + w >= u1 - u0 or y + h >= v1 - v0:
            continue                                  # cut by the window: size unknown
        if area / float(w * h) < FILL_MIN or max(w, h) / float(min(w, h)) > ASPECT_MAX:
            continue
        px, py = to_pitch(H, u0 + cents[k][0], v0 + cents[k][1], pitch)
        if _on_marking(px, py, pitch):
            continue                                  # penalty / centre spot: paint, not a ball
        d = float(np.hypot(px - cx, py - cy))
        if d <= radius_m and (best is None or d < best[0]):
            best = (d, float(px), float(py), int(area))
    return None if best is None else best[1:]


# -- planning, detection over a video, calibration -------------------------------------------------

def _sightings(seen):
    """(last, next): index of the last sighting at or before / first at or after each frame."""
    n = len(seen)
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    ridx = np.where(seen[::-1], np.arange(n), -1)
    nxt_r = np.maximum.accumulate(ridx)
    nxt = np.where(nxt_r >= 0, n - 1 - nxt_r, -1)[::-1]
    return last, nxt


def _gap_runs(seen, a, b, min_gap):
    run = ~seen[a:b]
    edges = np.diff(np.concatenate(([0], run.astype(np.int8), [0])))
    return [(a + s, a + e) for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if e - s > min_gap]


def _searches(frames_seen, atd_x, atd_y, halves, fps, min_gap):
    """{frame: (cx, cy, r)} for every frame inside a gap (of `frames_seen`) longer than min_gap:
    the straight line between the sightings around it, radius growing with the time to the
    nearest one. Only sightings inside the same half count."""
    last, nxt = _sightings(frames_seen)
    out = {}
    for a, b in halves:
        for s, e in _gap_runs(frames_seen, a, b, min_gap):
            for t in range(s, e):
                i, j = last[t], nxt[t]
                has_i, has_j = i >= a, 0 <= j < b
                if has_i and has_j:
                    w = (t - i) / float(j - i)
                    cx, cy = atd_x[i] * (1 - w) + atd_x[j] * w, atd_y[i] * (1 - w) + atd_y[j] * w
                    dt = min(t - i, j - t) / float(fps)
                elif has_i:
                    cx, cy, dt = atd_x[i], atd_y[i], (t - i) / float(fps)
                elif has_j:
                    cx, cy, dt = atd_x[j], atd_y[j], (j - t) / float(fps)
                else:
                    continue                          # no sighting in this half at all
                r = min(SEARCH_MAX_M, SEARCH_R0_M + SEARCH_GROWTH_M_S * dt)
                out[int(t)] = (float(cx), float(cy), float(r))
    return out


def plan(atd_x, atd_y, halves, fps, min_gap=MIN_GAP, probe_every_s=PROBE_EVERY_S,
         probe_len_s=PROBE_LEN_S, pitch=(105.0, 68.0), seed=0):
    """{frame: [(cx, cy, radius_m, kind), ...]} with kinds

      gap      a real tracker gap: search it, and fill it if found
      probe    a tracked frame inside an artificially hidden stretch, searched like a gap:
               measures how often and how accurately the ball is found when it is there
      absent   the same frame searched around a point placed so that the tracked ball lies
               outside the search radius: anything "found" there is a false positive

    Hidden stretches take their lengths from the real gaps (so probes get real gap-sized search
    radii), one per `probe_every_s` of tracked ball.
    """
    rng = np.random.default_rng(seed)
    seen = np.isfinite(atd_x) & np.isfinite(atd_y)
    targets = {}
    for t, (cx, cy, r) in _searches(seen, atd_x, atd_y, halves, fps, min_gap).items():
        targets.setdefault(t, []).append((cx, cy, r, "gap"))
    step = int(probe_every_s * fps)
    if step <= 0:
        return targets
    real_lengths = [e - s for a, b in halves for s, e in _gap_runs(seen, a, b, min_gap)]
    hidden = seen.copy()
    for a, b in halves:
        for s in range(a + step // 2, b, step):
            length = int(rng.choice(real_lengths)) if real_lengths else int(probe_len_s * fps)
            if s + length <= b and seen[s:s + length].mean() > 0.9:
                hidden[s:s + length] = False
    probes = _searches(hidden, atd_x, atd_y, halves, fps, min_gap)
    L, W = pitch
    for t, (cx, cy, r) in probes.items():
        if not seen[t]:
            continue                                  # a real gap, already planned
        targets.setdefault(t, []).append((cx, cy, r, "probe"))
        bx, by = atd_x[t], atd_y[t]
        for _ in range(8):                            # a ball-absent search, inside the pitch
            ang = rng.uniform(0, 2 * np.pi)
            dist = r + rng.uniform(*ABSENT_EXTRA_M)
            ax, ay = bx + dist * np.cos(ang), by + dist * np.sin(ang)
            if 0 <= ax <= L and 0 <= ay <= W:
                targets[t].append((float(ax), float(ay), float(r), "absent"))
                break
    return targets


def _scan(frames, Hs, valid, targets, n, pitch, progress=False):
    """One pass: ({(frame, kind): (x, y)} hits, frames read)."""
    hits = {}
    t = -1
    for t, frame in enumerate(frames):
        if t >= n:
            t = n - 1
            break
        if progress and t and t % 10000 == 0:
            print("  video frame {} / {}".format(t, n), flush=True)
        tgts = targets.get(t)
        if not tgts or frame is None or not valid[t]:
            continue
        for cx, cy, r, kind in tgts:
            hit = detect_in_frame(frame, Hs[t], (cx, cy), r, pitch)
            if hit is not None:
                hits[(t, kind)] = hit[:2]
    return hits, t + 1


def consistent(det_x, det_y, fps):
    """Drop detections with no other detection within +/- CONSIST_FRAMES frames at a distance a
    real ball could cover (CONSIST_SPEED m/s, plus 1 m). A lone blob is more likely a sock than
    a ball the tracker missed."""
    ok = np.isfinite(det_x) & np.isfinite(det_y)
    keep = np.zeros(len(det_x), dtype=bool)
    idx = np.flatnonzero(ok)
    for i in idx:
        for k in range(1, CONSIST_FRAMES + 1):
            hit = False
            for j in (i - k, i + k):
                if 0 <= j < len(det_x) and ok[j]:
                    if np.hypot(det_x[j] - det_x[i], det_y[j] - det_y[i]) <= \
                            1.0 + CONSIST_SPEED * k / float(fps):
                        hit = True
                        break
            if hit:
                keep[i] = True
                break
    return np.where(keep, det_x, np.nan), np.where(keep, det_y, np.nan)


def _series(hits, kind, n):
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    for (t, k), (hx, hy) in hits.items():
        if k == kind:
            x[t], y[t] = hx, hy
    return x, y


def detect_gaps(frames, Hs, valid, atd_x, atd_y, halves, pitch, fps, min_gap=MIN_GAP,
                targets=None):
    """(det_x, det_y): consistent detections on gap frames only (NaN elsewhere). `frames`
    yields one BGR image (or None, for frames not needed) per video frame, in order."""
    if targets is None:
        targets = plan(atd_x, atd_y, halves, fps, min_gap, probe_every_s=0, pitch=pitch)
    hits, _ = _scan(frames, Hs, valid, targets, len(atd_x), pitch)
    return consistent(*_series(hits, "gap", len(atd_x)), fps)


def run_video(video_path, Hs, valid, atd_x, atd_y, halves, pitch, fps, progress=True,
              probe_every_s=PROBE_EVERY_S, probe_len_s=PROBE_LEN_S):
    """One pass over the video: (det_x, det_y) for gap frames plus a calibration report."""
    import cv2

    n = len(atd_x)
    targets = plan(atd_x, atd_y, halves, fps, probe_every_s=probe_every_s,
                   probe_len_s=probe_len_s, pitch=pitch)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("cannot open video {}".format(video_path))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if progress and declared != n:
        print("  WARNING: the video declares {} frames, the tracking has {}".format(declared, n))

    def frames():
        t = 0
        while True:
            if t in targets:
                ok, img = cap.read()
                if not ok:
                    return
                yield img
            else:
                if not cap.grab():
                    return
                yield None
            t += 1

    hits, frames_read = _scan(frames(), Hs, valid, targets, n, pitch, progress)
    cap.release()
    gap_x, gap_y = consistent(*_series(hits, "gap", n), fps)
    pr_x, pr_y = consistent(*_series(hits, "probe", n), fps)
    ab_x, _ = consistent(*_series(hits, "absent", n), fps)
    kinds = {}
    for t, tgts in targets.items():
        for tg in tgts:
            kinds.setdefault(tg[3], []).append(t)
    probe = np.array(kinds.get("probe", []), dtype=int)
    absent = np.array(kinds.get("absent", []), dtype=int)
    gap = np.array(kinds.get("gap", []), dtype=int)
    found = np.isfinite(pr_x[probe]) if len(probe) else np.zeros(0, bool)
    err = np.hypot(pr_x[probe][found] - atd_x[probe][found],
                   pr_y[probe][found] - atd_y[probe][found]) if found.any() else np.zeros(0)
    report = {
        "frames_read": int(frames_read),
        "frames_expected": int(n),
        "gap_frames": int(len(gap)),
        "gap_filled": float(np.isfinite(gap_x[gap]).mean()) if len(gap) else 0.0,
        "probe_frames": int(len(probe)),
        "probe_found": float(found.mean()) if len(probe) else 0.0,
        "probe_median_err_m": float(np.median(err)) if len(err) else float("nan"),
        "probe_within_1m": float(np.mean(err < 1.0)) if len(err) else float("nan"),
        "absent_frames": int(len(absent)),
        "absent_fp_rate": float(np.isfinite(ab_x[absent]).mean()) if len(absent) else float("nan"),
    }
    return gap_x, gap_y, report


def load_homographies(path, n_frames):
    """([n, 3, 3] matrices, [n] valid) from the challenge's homography CSV (frame, h00..h22,
    reprojection_error, valid); frames without a row are invalid."""
    import pandas as pd

    table = pd.read_csv(path)
    cols = ["h{}{}".format(i, j) for i in range(3) for j in range(3)]
    Hs = np.full((n_frames, 3, 3), np.nan)
    valid = np.zeros(n_frames, dtype=bool)
    frames = table["frame"].to_numpy(dtype=np.int64)
    ok = (frames >= 0) & (frames < n_frames)
    Hs[frames[ok]] = table[cols].to_numpy(dtype=np.float64)[ok].reshape(-1, 3, 3)
    valid[frames[ok]] = table["valid"].to_numpy()[ok].astype(bool)
    valid &= np.isfinite(Hs).all(axis=(1, 2))
    return Hs, valid


def trustworthy(report):
    """Use the detections only if the calibration on this video says they are good: enough
    probes of both kinds, the ball found often and accurately when it is there, rarely "found"
    when it is not, and the whole video read. Anything missing or NaN fails closed."""
    def get(key):
        v = report.get(key)
        return float(v) if v is not None else float("nan")
    return bool(get("probe_frames") >= TRUST_MIN_PROBES
                and get("absent_frames") >= TRUST_MIN_PROBES
                and get("probe_found") >= TRUST_MIN_FOUND
                and get("probe_within_1m") >= TRUST_MIN_WITHIN_1M
                and get("absent_fp_rate") <= TRUST_MAX_FP
                and get("frames_read") == get("frames_expected"))


def merge(atd_x, atd_y, det_x, det_y):
    """Tracked ball where there is one, video detection where the tracker had none."""
    fill = ~(np.isfinite(atd_x) & np.isfinite(atd_y)) & np.isfinite(det_x) & np.isfinite(det_y)
    return np.where(fill, det_x, atd_x), np.where(fill, det_y, atd_y), fill

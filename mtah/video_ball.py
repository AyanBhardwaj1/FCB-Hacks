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
  3. calibrate  "probe" stretches where the tracking DOES see the ball are hidden and searched
                exactly like real gaps; agreement with the tracked position measures the
                detector's accuracy on this video without any ground truth
  4. merge      fill tracker gaps with detections; the usual spike filter (features.clean_ball)
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
# Fail closed: detections are only merged if calibration on this very video says they are good.
TRUST_MIN_PROBES = 100        # probe frames needed to judge at all
TRUST_MIN_FOUND = 0.15        # share of probe frames with a detection
TRUST_MIN_WITHIN_1M = 0.80    # share of probe detections within 1 m of the tracked ball


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


def plan(atd_x, atd_y, halves, fps, min_gap=MIN_GAP, probe_every_s=PROBE_EVERY_S,
         probe_len_s=PROBE_LEN_S):
    """{frame: (cx, cy, radius_m, kind)} with kind 'gap' (search to fill) or 'probe' (a tracked
    frame searched as if it were in a gap, for calibration)."""
    seen = np.isfinite(atd_x) & np.isfinite(atd_y)
    targets = {}

    def add(frames_seen, kind):
        last, nxt = _sightings(frames_seen)
        for a, b in halves:
            run = np.zeros(len(frames_seen), dtype=bool)
            run[a:b] = ~frames_seen[a:b]
            edges = np.diff(np.concatenate(([0], run.astype(np.int8), [0])))
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if e - s <= min_gap:
                    continue
                for t in range(s, e):
                    if kind == "probe" and not seen[t]:
                        continue
                    i, j = last[t], nxt[t]
                    if i < a and j < 0:
                        continue
                    if i >= a and 0 <= j < b:
                        w = (t - i) / float(j - i)
                        cx = atd_x[i] * (1 - w) + atd_x[j] * w
                        cy = atd_y[i] * (1 - w) + atd_y[j] * w
                        dt = min(t - i, j - t) / float(fps)
                    elif i >= a:
                        cx, cy, dt = atd_x[i], atd_y[i], (t - i) / float(fps)
                    else:
                        cx, cy, dt = atd_x[j], atd_y[j], (j - t) / float(fps)
                    r = min(SEARCH_MAX_M, SEARCH_R0_M + SEARCH_GROWTH_M_S * dt)
                    targets.setdefault(int(t), (float(cx), float(cy), float(r), kind))

    add(seen, "gap")
    # probes: hide short stretches of tracked ball, evenly spaced, and search them like gaps
    step, length = int(probe_every_s * fps), int(probe_len_s * fps)
    if step > 0:
        hidden = seen.copy()
        for a, b in halves:
            for s in range(a + step // 2, b - length, step):
                if seen[s:s + length].mean() > 0.9:
                    hidden[s:s + length] = False
        add(hidden, "probe")
    return targets


def detect_gaps(frames, Hs, valid, atd_x, atd_y, halves, pitch, fps, min_gap=MIN_GAP,
                targets=None):
    """(det_x, det_y): detections on gap frames only (NaN elsewhere). `frames` yields one BGR
    image (or None, for frames not needed) per video frame, in order."""
    if targets is None:
        targets = {t: v for t, v in plan(atd_x, atd_y, halves, fps, min_gap, probe_every_s=0)
                   .items() if v[3] == "gap"}
    n = len(atd_x)
    det = np.full((n, 2), np.nan)
    for t, frame in enumerate(frames):
        if t >= n:
            break
        tgt = targets.get(t)
        if tgt is None or frame is None or not valid[t]:
            continue
        hit = detect_in_frame(frame, Hs[t], tgt[:2], tgt[2], pitch)
        if hit is not None:
            det[t] = hit[:2]
    return det[:, 0], det[:, 1]


def run_video(video_path, Hs, valid, atd_x, atd_y, halves, pitch, fps, progress=True,
              probe_every_s=PROBE_EVERY_S, probe_len_s=PROBE_LEN_S):
    """One pass over the video: (det_x, det_y) for gap frames plus a calibration report."""
    import cv2

    targets = plan(atd_x, atd_y, halves, fps, probe_every_s=probe_every_s,
                   probe_len_s=probe_len_s)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("cannot open video {}".format(video_path))

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
            if progress and t % 10000 == 0:
                print("  video frame {} / {}".format(t, len(atd_x)), flush=True)

    det_x, det_y = detect_gaps(frames(), Hs, valid, atd_x, atd_y, halves, pitch, fps,
                               targets={t: v for t, v in targets.items()})
    cap.release()
    probe = np.array([t for t, v in targets.items() if v[3] == "probe"], dtype=int)
    gap = np.array([t for t, v in targets.items() if v[3] == "gap"], dtype=int)
    found = np.isfinite(det_x[probe]) if len(probe) else np.zeros(0, bool)
    err = np.hypot(det_x[probe][found] - atd_x[probe][found],
                   det_y[probe][found] - atd_y[probe][found]) if found.any() else np.zeros(0)
    report = {
        "gap_frames": int(len(gap)),
        "gap_filled": float(np.isfinite(det_x[gap]).mean()) if len(gap) else 0.0,
        "probe_frames": int(len(probe)),
        "probe_found": float(found.mean()) if len(probe) else 0.0,
        "probe_median_err_m": float(np.median(err)) if len(err) else float("nan"),
        "probe_within_1m": float(np.mean(err < 1.0)) if len(err) else float("nan"),
    }
    # probe frames were tracked: keep detections on gap frames only
    keep = np.zeros(len(atd_x), dtype=bool)
    keep[gap] = True
    det_x[~keep] = np.nan
    det_y[~keep] = np.nan
    return det_x, det_y, report


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
    """Use the detections only if the calibration probes on this video say they are good."""
    return (report["probe_frames"] >= TRUST_MIN_PROBES
            and report["probe_found"] >= TRUST_MIN_FOUND
            and report["probe_within_1m"] >= TRUST_MIN_WITHIN_1M)


def merge(atd_x, atd_y, det_x, det_y):
    """Tracked ball where there is one, video detection where the tracker had none."""
    fill = ~(np.isfinite(atd_x) & np.isfinite(atd_y)) & np.isfinite(det_x) & np.isfinite(det_y)
    return np.where(fill, det_x, atd_x), np.where(fill, det_y, atd_y), fill

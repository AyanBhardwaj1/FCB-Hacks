"""Video ball detector, on synthetic video with a known homography (no match data needed).

    python -m pytest -q tests/test_video_ball.py
"""

import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mtah import video_ball  # noqa: E402

PITCH = (105.0, 68.0)
W, H_PX = 1920, 1080


def _homography():
    """Normalised pitch (0-1) -> pixels: a mild perspective view of the whole pitch."""
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    dst = np.float32([[260, 180], [1660, 180], [1880, 1000], [40, 1000]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def _render(n, ball_xy, hidden, players, decoy):
    """Frames with a green pitch, coloured players, a white-shirted decoy and the ball."""
    H = _homography()
    frames = []
    for t in range(n):
        img = np.zeros((H_PX, W, 3), np.uint8)
        img[:] = (40, 140, 40)                                  # BGR grass
        for (px, py), colour in players:
            u, v = video_ball.to_pixels(H, px, py, PITCH)
            cv2.rectangle(img, (int(u) - 8, int(v) - 40), (int(u) + 8, int(v)), colour, -1)
        u, v = video_ball.to_pixels(H, decoy[0], decoy[1], PITCH)   # a player in a white shirt
        cv2.rectangle(img, (int(u) - 9, int(v) - 42), (int(u) + 9, int(v)), (245, 245, 245), -1)
        if not hidden[t]:
            u, v = video_ball.to_pixels(H, ball_xy[t, 0], ball_xy[t, 1], PITCH)
            r = max(2, int(round(video_ball.pixels_per_metre(H, ball_xy[t, 0], ball_xy[t, 1],
                                                               PITCH) * 0.11)))
            cv2.circle(img, (int(round(u)), int(round(v))), r, (250, 250, 250), -1)
        frames.append(img)
    return frames, np.repeat(H[None], n, axis=0)


def test_pixel_mapping_round_trips():
    H = _homography()
    u, v = video_ball.to_pixels(H, 30.0, 20.0, PITCH)
    x, y = video_ball.to_pitch(H, u, v, PITCH)
    assert x == pytest.approx(30.0, abs=1e-6) and y == pytest.approx(20.0, abs=1e-6)


def test_detects_the_ball_near_its_expected_position_and_ignores_decoys():
    n = 60
    t = np.arange(n)
    ball = np.column_stack([40.0 + 0.4 * t, 30.0 + 0.1 * t])
    hidden = (t >= 25) & (t < 35)                       # occluded: nothing should be found
    players = [((45.0, 35.0), (40, 40, 200)), ((50.0, 28.0), (200, 40, 40))]
    frames, Hs = _render(n, ball, hidden, players, decoy=(47.0, 31.5))
    found = 0
    for i in range(n):
        centre = ball[i] + np.array([1.5, -1.0])        # a prediction 1.8 m off
        hit = video_ball.detect_in_frame(frames[i], Hs[i], centre, 4.0, PITCH)
        if hidden[i]:
            assert hit is None, i
        elif hit is not None:
            found += 1
            assert np.hypot(hit[0] - ball[i, 0], hit[1] - ball[i, 1]) < 0.5, (i, hit)
    assert found >= 0.9 * (~hidden).sum()


def test_detect_gaps_fills_only_tracker_gaps():
    n = 80
    t = np.arange(n)
    ball = np.column_stack([30.0 + 0.3 * t, 34.0 + 0.05 * t])
    frames, Hs = _render(n, ball, np.zeros(n, bool), [], decoy=(80.0, 60.0))
    atd_x, atd_y = ball[:, 0].copy(), ball[:, 1].copy()
    atd_x[20:60] = np.nan                               # the tracker lost it for 1.6 s
    atd_y[20:60] = np.nan
    det_x, det_y = video_ball.detect_gaps(iter(frames), Hs, np.ones(n, bool), atd_x, atd_y,
                                          [(0, n)], PITCH, fps=25, min_gap=10)
    assert np.all(np.isnan(det_x[:20])) and np.all(np.isnan(det_x[60:]))   # seen: untouched
    got = np.isfinite(det_x[20:60])
    assert got.mean() > 0.9
    err = np.hypot(det_x[20:60][got] - ball[20:60, 0][got], det_y[20:60][got] - ball[20:60, 1][got])
    assert np.max(err) < 0.5


def test_run_video_end_to_end_with_calibration_probes(tmp_path):
    """Encode a synthetic clip, run the one-pass detector on the file, check the report."""
    n = 200
    t = np.arange(n)
    ball = np.column_stack([30.0 + 0.2 * t, 30.0 + 0.04 * t])
    frames, Hs = _render(n, ball, np.zeros(n, bool), [], decoy=(90.0, 60.0))
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (W, H_PX))
    for f in frames:
        writer.write(f)
    writer.release()
    atd_x, atd_y = ball[:, 0].copy(), ball[:, 1].copy()
    atd_x[120:160] = np.nan
    atd_y[120:160] = np.nan
    det_x, det_y, report = video_ball.run_video(path, Hs, np.ones(n, bool), atd_x, atd_y,
                                                [(0, n)], PITCH, 25, progress=False,
                                                probe_every_s=3.0, probe_len_s=1.0)
    assert report["gap_frames"] == 40 and report["gap_filled"] > 0.8
    assert report["probe_frames"] > 0 and report["probe_found"] > 0.8
    assert report["probe_median_err_m"] < 0.5
    assert np.all(np.isnan(det_x[:120])) and np.all(np.isnan(det_x[160:]))   # probes dropped
    mx, my, filled = video_ball.merge(atd_x, atd_y, det_x, det_y)
    assert filled.sum() == np.isfinite(det_x).sum() and np.isfinite(mx[120:160]).mean() > 0.8


def test_load_homographies_reads_the_challenge_csv(tmp_path):
    H = _homography()
    rows = ["frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"]
    rows.append("0," + ",".join("{:.10g}".format(v) for v in H.ravel()) + ",0.2,1")
    rows.append("1," + ",".join(["NaN"] * 9) + ",NaN,0")
    path = tmp_path / "h.csv"
    path.write_text("\n".join(rows) + "\n")
    Hs, valid = video_ball.load_homographies(str(path), 2)
    np.testing.assert_allclose(Hs[0], H)
    assert valid.tolist() == [True, False]


def test_untrustworthy_calibration_is_not_used():
    good = dict(probe_frames=500, probe_found=0.6, probe_within_1m=0.9, absent_frames=500,
                absent_fp_rate=0.0, frames_read=10, frames_expected=10)
    assert video_ball.trustworthy(good)
    for bad in (dict(good, probe_frames=20), dict(good, probe_found=0.05),
                dict(good, probe_within_1m=0.5), dict(good, probe_within_1m=float("nan")),
                {k: v for k, v in good.items() if k != "absent_fp_rate"}):
        assert not video_ball.trustworthy(bad)


def _clip_bundle(tmp_path, n=200):
    t = np.arange(n)
    ball = np.column_stack([30.0 + 0.2 * t, 30.0 + 0.04 * t])
    frames, Hs = _render(n, ball, np.zeros(n, bool), [], decoy=(90.0, 60.0))
    video = str(tmp_path / "BAR-ATH_tactical.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 25, (W, H_PX))
    for f in frames:
        writer.write(f)
    writer.release()
    rows = ["frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"]
    rows += ["{},".format(i) + ",".join("{:.10g}".format(v) for v in Hs[i].ravel()) + ",0.2,1"
             for i in range(n)]
    hom = tmp_path / "h.csv"
    hom.write_text("\n".join(rows) + "\n")
    return ball, {"video": video, "homography": str(hom)}


def test_run_py_video_ball_fails_closed_and_fills_when_trusted(tmp_path, monkeypatch):
    import run
    from mtah import config
    from mtah.io import Match
    n = 200
    ball, paths = _clip_bundle(tmp_path, n)
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))

    def match():
        bx, by = ball[:, 0].copy(), ball[:, 1].copy()
        bx[120:160] = np.nan
        by[120:160] = np.nan
        z = np.zeros((n, 1), np.float32)
        return Match(x=z, y=z, speed=z, team=np.zeros(1, np.int8), ball_x=bx.astype(np.float32),
                     ball_y=by.astype(np.float32), pitch=PITCH, halves=[(0, n)])

    m = match()
    run.add_video_ball(m, paths, [(0, n)])               # 200 frames: too few probes to trust
    assert np.isnan(m.ball_x[120:160]).all()
    monkeypatch.setattr(video_ball, "trustworthy", lambda report: True)
    m = match()
    run.add_video_ball(m, paths, [(0, n)])               # now from the cache, and trusted
    filled = np.isfinite(m.ball_x[120:160])
    assert filled.mean() > 0.8
    err = np.hypot(m.ball_x[120:160][filled] - ball[120:160, 0][filled],
                   m.ball_y[120:160][filled] - ball[120:160, 1][filled])
    assert err.max() < 0.5


# -- review fixes: false positives, markings, truncation, planning edge cases ------------------------

def test_penalty_and_centre_spots_are_not_the_ball():
    H = _homography()
    img = np.zeros((H_PX, W, 3), np.uint8)
    img[:] = (40, 140, 40)
    for spot in ((11.0, 34.0), (94.0, 34.0), (52.5, 34.0)):
        u, v = video_ball.to_pixels(H, spot[0], spot[1], PITCH)
        r = max(2, int(round(video_ball.pixels_per_metre(H, spot[0], spot[1], PITCH) * 0.11)))
        cv2.circle(img, (int(round(u)), int(round(v))), r, (250, 250, 250), -1)
        assert video_ball.detect_in_frame(img, H, np.array(spot) + 1.0, 5.0, PITCH) is None


def test_absent_probes_are_searched_away_from_the_ball():
    n = 2000
    t = np.arange(n)
    bx = 30.0 + 20.0 * np.sin(t / 300.0)
    by = 34.0 + 10.0 * np.cos(t / 400.0)
    bx[500:560] = np.nan                                # one real gap, to set probe lengths
    by[500:560] = np.nan
    targets = video_ball.plan(bx, by, [(0, n)], 25, probe_every_s=10.0)
    absent = [(f, tg) for f, tgs in targets.items() for tg in tgs if tg[3] == "absent"]
    assert absent
    for f, (cx, cy, r, _) in absent:
        assert np.hypot(cx - bx[f], cy - by[f]) >= r + 1.0     # the true ball is outside


def test_isolated_detections_are_dropped():
    n = 50
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    x[10:14], y[10:14] = 20.0, 30.0                     # a short consistent run: kept
    x[30], y[30] = 60.0, 10.0                           # alone: dropped
    cx, cy = video_ball.consistent(x, y, 25)
    assert np.isfinite(cx[10:14]).all() and np.isnan(cx[30])


def test_truncated_video_is_reported_and_not_trusted(tmp_path):
    n = 200
    t = np.arange(n)
    ball = np.column_stack([30.0 + 0.2 * t, 30.0 + 0.04 * t])
    frames, Hs = _render(150, ball[:150], np.zeros(150, bool), [], decoy=(90.0, 60.0))
    path = str(tmp_path / "short.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (W, H_PX))
    for f in frames:
        writer.write(f)
    writer.release()
    Hs = np.concatenate([Hs, np.repeat(Hs[:1], n - 150, axis=0)])
    _, _, report = video_ball.run_video(path, Hs, np.ones(n, bool), ball[:, 0], ball[:, 1],
                                        [(0, n)], PITCH, 25, progress=False,
                                        probe_every_s=3.0, probe_len_s=1.0)
    assert report["frames_read"] == 150 and report["frames_expected"] == n
    assert not video_ball.trustworthy(dict(report, probe_frames=1000, probe_found=1.0,
                                           probe_within_1m=1.0, absent_fp_rate=0.0))


def test_trust_gate_rejects_a_high_false_positive_rate():
    good = dict(probe_frames=500, probe_found=0.6, probe_within_1m=0.9, absent_frames=500,
                absent_fp_rate=0.01, frames_read=10, frames_expected=10)
    assert video_ball.trustworthy(good)
    assert not video_ball.trustworthy(dict(good, absent_fp_rate=0.2))
    assert not video_ball.trustworthy(dict(good, absent_frames=10))


def test_a_half_without_sightings_is_not_searched_from_another_half():
    n = 400
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    bx[250:], by[250:] = 80.0, 10.0                     # sightings only in the second half
    targets = video_ball.plan(bx, by, [(0, 200), (200, n)], 25, probe_every_s=0)
    assert not any(f < 200 for f in targets)


def test_video_cache_key_covers_everything_the_detections_depend_on(tmp_path):
    import run
    from mtah.io import Match
    _, paths = _clip_bundle(tmp_path, 30)
    z = np.zeros((30, 1), np.float32)
    bx = np.linspace(0, 10, 30).astype(np.float32)
    m = Match(x=z, y=z, speed=z, team=np.zeros(1, np.int8), ball_x=bx, ball_y=bx.copy(),
              pitch=PITCH, halves=[(0, 30)])
    base = run.video_cache_key(paths, m, [(0, 30)])
    assert run.video_cache_key(paths, m, [(0, 20)]) != base             # halves
    m2 = Match(**{k: getattr(m, k) for k in m.__dataclass_fields__})
    m2.ball_y = (bx + 1).astype(np.float32)
    assert run.video_cache_key(paths, m2, [(0, 30)]) != base            # ball_y
    m3 = Match(**{k: getattr(m, k) for k in m.__dataclass_fields__})
    m3.pitch = (105.3, 68.0)
    assert run.video_cache_key(paths, m3, [(0, 30)]) != base            # pitch

"""Fast checks that need no match data and no sample data.

    python -m pytest -q
"""

import itertools
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mtah import config, decode, features, sample_games  # noqa: E402
from mtah.io import AWAY, HOME, Match, load_epts, write_submission  # noqa: E402


# -- decoder ---------------------------------------------------------------------------------------

def _path_score(scores, path, switch, stop, restart):
    cost = {(0, 1): switch, (1, 0): switch, (0, 2): stop, (1, 2): stop, (2, 0): restart,
            (2, 1): restart}
    total = scores[np.arange(len(path)), path].sum()
    return total - sum(cost.get((int(a), int(b)), 0.0) for a, b in zip(path[:-1], path[1:]))


def test_viterbi_finds_the_optimum():
    rng = np.random.default_rng(0)
    for _ in range(30):
        scores = rng.normal(size=(7, 3))
        prm = rng.uniform(0, 3, 3)
        best = max(_path_score(scores, np.array(p), *prm)
                   for p in itertools.product(range(3), repeat=7))
        got = _path_score(scores, decode.viterbi(scores, *prm), *prm)
        assert got == pytest.approx(best)


def test_decode_marks_outside_halves_x():
    n = 100
    labels = decode.decode(np.full(n, 0.1), np.full(n, 0.9), [(10, 40), (60, 90)], n)
    assert set(labels[:10]) == {"X"} and set(labels[40:60]) == {"X"} and set(labels[90:]) == {"X"}
    assert set(labels[10:40]) == {"H"}


def test_decoder_ignores_a_deflection():
    """Two frames of evidence for the other team do not buy a change of possession."""
    n = 200
    p_home = np.full(n, 0.9)
    p_home[100:102] = 0.05
    labels = decode.decode(np.full(n, 0.05), p_home, [(0, n)], n, config.DECODER)
    assert set(labels) == {"H"}


# -- tracking I/O ------------------------------------------------------------------------------------

META = """<?xml version="1.0"?>
<main><Metadata>
  <Sessions><Session><FieldSize><Width>100</Width><Height>50</Height></FieldSize></Session></Sessions>
  <Teams><Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
  <Players><Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/><Player id="P3" teamId="T1"/></Players>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification startFrame="1" endFrame="2" separator=":">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=","><PlayerChannelRef playerChannelId="p1x"/><PlayerChannelRef playerChannelId="p1y"/><PlayerChannelRef playerChannelId="p1s"/></SplitRegister>
        <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2x"/><PlayerChannelRef playerChannelId="p2y"/><PlayerChannelRef playerChannelId="p2s"/></SplitRegister>
      </SplitRegister>
      <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="z"/></SplitRegister>
    </DataFormatSpecification>
    <DataFormatSpecification startFrame="3" endFrame="3" separator=":">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=","><PlayerChannelRef playerChannelId="p3x"/><PlayerChannelRef playerChannelId="p3y"/><PlayerChannelRef playerChannelId="p3s"/></SplitRegister>
        <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2x"/><PlayerChannelRef playerChannelId="p2y"/><PlayerChannelRef playerChannelId="p2s"/></SplitRegister>
      </SplitRegister>
      <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="z"/></SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
  <PlayerChannels>
    <PlayerChannel id="p1x" channelId="x" playerId="P1"/><PlayerChannel id="p1y" channelId="y" playerId="P1"/><PlayerChannel id="p1s" channelId="speed" playerId="P1"/>
    <PlayerChannel id="p2x" channelId="x" playerId="P2"/><PlayerChannel id="p2y" channelId="y" playerId="P2"/><PlayerChannel id="p2s" channelId="speed" playerId="P2"/>
    <PlayerChannel id="p3x" channelId="x" playerId="P3"/><PlayerChannel id="p3y" channelId="y" playerId="P3"/><PlayerChannel id="p3s" channelId="speed" playerId="P3"/>
  </PlayerChannels>
</main>
"""

RAW = """1:0.1,0.2,1.5;0.5,0.5,2.0:0.3,0.4,0.0
2:NaN,NaN,NaN;0.6,0.5,2.5:NaN,NaN,NaN
3:0.9,0.8,3.0;0.7,0.5,1.0:0.5,0.5,0.0
"""


def test_load_epts_follows_the_format_spec(tmp_path):
    meta, raw = tmp_path / "m.xml", tmp_path / "r.txt"
    meta.write_text(META)
    raw.write_text(RAW)
    m = load_epts(str(meta), str(raw), n_frames=4)
    assert m.pitch == (100.0, 50.0)
    assert list(m.team) == [HOME, AWAY, HOME]
    assert m.x[0, 0] == pytest.approx(10.0) and m.y[0, 0] == pytest.approx(10.0)
    assert np.isnan(m.x[1, 0]) and m.x[1, 1] == pytest.approx(60.0)
    # frame 3 puts P3 first on the line: it must land in P3's column, not P1's
    assert m.x[2, 2] == pytest.approx(90.0) and np.isnan(m.x[2, 0])
    assert m.speed[2, 2] == pytest.approx(3.0)
    assert m.ball_x[0] == pytest.approx(30.0) and np.isnan(m.ball_x[1])
    assert np.isnan(m.x[3]).all()        # no record for frame 4


def test_write_submission_validates(tmp_path):
    games, pm = config.challenge_modules()
    n = games.source_frames("tactical", config.GAME)
    path = write_submission(str(tmp_path / "s.csv"), np.full(n, "X"))
    assert pm.validate_submission(pm.read_label_csv(path), n) is None


# -- ball cleaning -------------------------------------------------------------------------------------

def test_clean_ball_removes_spikes_and_fills_short_gaps():
    n = 50
    bx = np.linspace(0, 10, n)
    by = np.zeros(n)
    bx[20] = 90.0                       # a false detection
    bx[30:35] = np.nan                  # a short gap
    bx[40:] = np.nan
    x, y, seen, known = features.clean_ball(bx, by, 25)
    assert not seen[20] and known[20] and x[20] == pytest.approx(np.linspace(0, 10, n)[20])
    assert known[30:35].all() and not seen[30:35].any()
    assert not known[40:].any()


# -- labels from events ---------------------------------------------------------------------------------

def _ev(t, team, start, end=None, sub=""):
    return {"type": t, "team": team, "subtype": sub, "period": 1, "start": start,
            "end": start if end is None else end}


def test_labels_from_events_applies_the_control_rule():
    ev = [_ev("SET PIECE", HOME, 10, sub="KICK OFF"), _ev("PASS", HOME, 10, 30),
          _ev("BALL LOST", HOME, 100),
          _ev("RECOVERY", AWAY, 110),             # an away touch...
          _ev("BALL LOST", AWAY, 120),            # ...lost 0.4 s later: not controlled
          _ev("RECOVERY", HOME, 125), _ev("PASS", HOME, 130, 150),
          _ev("BALL LOST", HOME, 200), _ev("RECOVERY", AWAY, 210), _ev("PASS", AWAY, 215, 240),
          _ev("BALL OUT", AWAY, 300, 305), _ev("SET PIECE", HOME, 400, sub="THROW IN"),
          _ev("PASS", HOME, 400, 420)]
    lab = sample_games.labels_from_events(ev, 500, [(0, 500)])
    assert set(lab[:10]) == {"D"}
    assert set(lab[10:210]) == {"H"}            # the away touch at 110 is not a possession
    assert set(lab[210:305]) == {"A"}
    assert set(lab[305:400]) == {"D"}
    assert set(lab[400:]) == {"H"}


# -- symmetry -------------------------------------------------------------------------------------------

def _toy_match(seed, T=3000, P=22):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.2, (T, P)), axis=0) % 100 + 2
    y = np.cumsum(rng.normal(0, 0.2, (T, P)), axis=0) % 60 + 4
    x[:, :11] -= 25
    x[:, 11:] += 25
    x = np.clip(x, 0, 105)
    team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    j = rng.integers(0, P, T // 50).repeat(50)[:T]
    bx = x[np.arange(T), j] + rng.normal(0, 0.5, T)
    by = y[np.arange(T), j] + rng.normal(0, 0.5, T)
    bx[rng.random(T) < 0.3] = np.nan
    m = Match(x=x.astype(np.float32), y=y.astype(np.float32),
              speed=np.abs(rng.normal(2, 1, (T, P))).astype(np.float32), team=team,
              ball_x=bx.astype(np.float32), ball_y=by.astype(np.float32), pitch=(105.0, 68.0),
              halves=[(0, T // 2), (T // 2, T)])
    labels = np.where(team[j] == HOME, "H", "A")
    labels[np.isnan(bx) & (rng.random(T) < 0.5)] = "D"
    return m, labels


def test_models_are_exactly_team_symmetric():
    from mtah.model import FrameModels
    items = []
    for s in (0, 1):
        m, lab = _toy_match(s)
        items.append((features.compute(m), lab, m.halves, s))
    models = FrameModels(gb_params=dict(max_iter=20), stride=2).fit(items)
    m, _ = _toy_match(5)
    p_dead, p_home = models.predict(features.compute(m), m.halves)
    m.team = 1 - m.team
    q_dead, q_home = models.predict(features.compute(m), m.halves)
    np.testing.assert_allclose(q_dead, p_dead, atol=1e-9)
    np.testing.assert_allclose(q_home, 1 - p_home, atol=1e-9)


# -- touch events and side windows (feature v2) --------------------------------------------------------

def test_touch_events_detects_a_kick_and_credits_the_nearest_team():
    fps = 25
    n = 100
    t = np.arange(n)
    # ball rolls +x at 10 m/s, is kicked back at frame 50 by a home player standing there
    bx = np.where(t < 50, 10.0 + 10.0 * t / fps, 30.0 - 10.0 * (t - 50) / fps)
    by = np.full(n, 30.0)
    d_home = np.abs(bx - 30.0) + 0.3       # the home player stands at x = 30
    d_away = np.full(n, 15.0)
    touch_h, touch_a = features.touch_events(bx, by, d_home, d_away, fps)
    hits = np.flatnonzero(touch_h)
    assert len(hits) and np.all(np.abs(hits - 50) <= 3)
    assert not touch_a.any()


def test_touch_events_ignore_smooth_flight_and_unattended_kinks():
    fps = 25
    n = 100
    t = np.arange(n)
    bx = 10.0 + 10.0 * t / fps                # constant velocity: no touch anywhere
    by = np.full(n, 30.0)
    far = np.full(n, 20.0)
    th, ta = features.touch_events(bx, by, np.full(n, 0.5), far, fps)
    assert not th.any() and not ta.any()
    bx2 = np.where(t < 50, bx, bx[50] - 10.0 * (t - 50) / fps)   # a kink with nobody near
    th, ta = features.touch_events(bx2, by, far, far, fps)
    assert not th.any() and not ta.any()


def test_touch_events_are_nan_safe():
    fps = 25
    bx = np.full(60, np.nan)
    by = np.full(60, np.nan)
    th, ta = features.touch_events(bx, by, np.full(60, np.nan), np.full(60, np.nan), fps)
    assert th.dtype == bool and not th.any() and not ta.any()


def test_side_delta_is_future_minus_past_mean():
    a = np.concatenate([np.zeros(50), np.ones(50)])
    d = features.side_delta(a, 10)
    assert d[50] == pytest.approx(1.0)       # 10 frames after (incl. none of the past) vs before
    assert d[20] == pytest.approx(0.0) and d[80] == pytest.approx(0.0)
    assert np.isnan(d[0])                    # nothing before the first frame


# -- review fixes ------------------------------------------------------------------------------------

def test_parser_rejects_lines_with_missing_fields():
    """A group written as a bare 'NaN' must not shift every later field one column left."""
    from mtah.io import _parse_lines
    with pytest.raises(ValueError, match="line"):
        _parse_lines(["1:0.1,0.2,1;0.3,0.4,2:0.5,0.6,0\n", "2:NaN;0.3,0.4,2:0.5,0.6,0\n"], 10)


def test_load_epts_rejects_records_outside_every_format_block(tmp_path):
    meta, raw = tmp_path / "m.xml", tmp_path / "r.txt"
    meta.write_text(META)
    raw.write_text(RAW + "9:0.1,0.2,1.5;0.5,0.5,2.0:0.3,0.4,0.0\n")   # frame 9: no spec block
    with pytest.raises(ValueError, match="DataFormatSpecification"):
        load_epts(str(meta), str(raw), n_frames=10)


def test_models_refuse_features_they_were_not_trained_on():
    from mtah.model import FrameModels
    m, lab = _toy_match(0)
    feats = features.compute(m)
    models = FrameModels(gb_params=dict(max_iter=5), stride=4).fit([(feats, lab, m.halves, 0)])
    del feats["team"][HOME]["own_d"], feats["team"][AWAY]["own_d"]
    with pytest.raises(ValueError, match="own_d"):
        models.predict(feats, m.halves)


def test_shipped_model_matches_the_current_features():
    import pickle
    with open(config.MODEL_PATH, "rb") as fh:
        models = pickle.load(fh)
    m, _ = _toy_match(3)
    p_dead, p_home = models.predict(features.compute(m), m.halves)
    assert p_dead.shape == p_home.shape == (m.n_frames,)


def test_degraded_ball_visibility_does_not_mark_label_boundaries():
    """Ball dropout must not switch on/off exactly where play stops or restarts (label leak)."""
    m, _ = _toy_match(11, T=20000)
    blocks = np.arange(m.n_frames) // 400
    labels = np.where(blocks % 3 == 2, "D", np.where(blocks % 2 == 0, "H", "A"))
    m.ball_x = np.where(np.isnan(m.ball_x), m.x[:, 0], m.ball_x).astype(np.float32)
    m.ball_y = np.where(np.isnan(m.ball_y), m.y[:, 0], m.ball_y).astype(np.float32)
    deg = sample_games.degrade(m, labels, seed=4)
    seen = np.isfinite(deg.ball_x)
    flips = np.flatnonzero(seen[1:] != seen[:-1]) + 1
    boundaries = np.flatnonzero((labels[1:] == "D") != (labels[:-1] == "D")) + 1
    at_boundary = np.isin(boundaries, flips).mean()
    base_rate = len(flips) / m.n_frames
    assert at_boundary < max(0.1, 5 * base_rate), (at_boundary, base_rate)


def test_degraded_dead_ball_is_no_jitterier_than_live_ball():
    """The synthetic dead-ball path must not carry its own noise: the labels would leak into
    ball speed and the touch features through how jittery the ball looks."""
    m, _ = _toy_match(12, T=20000)
    t = np.arange(m.n_frames)
    blocks = t // 400
    labels = np.where(blocks % 3 == 2, "D", np.where(blocks % 2 == 0, "H", "A"))
    m.ball_x = (50 + 20 * np.sin(t / 200.0)).astype(np.float32)     # smooth live ball
    m.ball_y = (34 + 10 * np.cos(t / 300.0)).astype(np.float32)
    m.ball_x[labels == "D"] = np.nan                                  # clean data: no dead ball
    m.ball_y[labels == "D"] = np.nan
    deg = sample_games.degrade(m, labels, seed=5, spike_rate=0.0)
    bx = deg.ball_x.astype(np.float64)
    jerk = np.abs(bx[2:] - 2 * bx[1:-1] + bx[:-2])
    mid = labels[1:-1]
    live = np.nanmedian(jerk[mid != "D"])
    dead = np.nanmedian(jerk[mid == "D"])
    assert dead < 1.2 * live, (dead, live)


# -- dead-ball boundary features (feature v3) -----------------------------------------------------------

def test_projected_exit_counts_down_through_a_tracker_gap():
    """Ball seen moving towards the touchline at 10 m/s, 4 m away, then lost: the projected
    crossing is 0.4 s after the last sighting, so 1 s into the gap it is 0.6 s in the past."""
    fps = 25
    n = 60
    t = np.arange(n)
    bx = np.full(n, 50.0)
    by = 58.0 + 10.0 * t / fps          # 68 m wide pitch: reaches y = 68 at t = 1 s
    known = t <= 15                     # last seen at frame 15, y = 64
    bx[~known] = np.nan
    by[~known] = np.nan
    vx = np.zeros(n)
    vy = np.where(known, 10.0, np.nan)
    out = features.projected_exit(bx, by, known, vx, vy, (105.0, 68.0), fps, cap=10.0)
    assert out[15] == pytest.approx(0.4, abs=1e-6)
    assert out[40] == pytest.approx(0.4 - 25 / fps, abs=1e-6)       # 1 s after last sighting
    assert out[0] == pytest.approx(1.0, abs=1e-6)                     # 10 m to go at 10 m/s


def test_projected_exit_is_capped_for_a_still_or_unseen_ball():
    fps = 25
    n = 20
    bx = np.full(n, 50.0)
    by = np.full(n, 30.0)
    known = np.ones(n, dtype=bool)
    out = features.projected_exit(bx, by, known, np.zeros(n), np.zeros(n), (105.0, 68.0), fps,
                                  cap=10.0)
    assert np.all(out == 10.0)                                        # not heading anywhere
    never = features.projected_exit(np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool),
                                    np.zeros(n), np.zeros(n), (105.0, 68.0), fps, cap=10.0)
    assert np.all(np.isnan(never))


def test_boundary_features_exist_and_keep_team_symmetry():
    m, _ = _toy_match(2)
    f = features.compute(m)
    for name in ("since_oob_s", "until_oob_s", "since_still_s", "until_still_s",
                 "ball_proj_exit_s", "speed_delta_1s", "speed_delta_2s",
                 "ball_speed_before_1s", "ball_speed_after_1s"):
        assert name in f["shared"], name
        assert np.isfinite(f["shared"][name][m.halves[0][0] + 100:m.halves[0][1] - 100]).any()


def test_degradation_parameters_are_drawn_per_seed_within_ranges():
    """Training data randomizes the tracking noise (the real ATD's level is unknown); explicit
    arguments still pin it, which is how the evaluation scenarios stay fixed."""
    a, b = sample_games.draw_params(1), sample_games.draw_params(2)
    assert a == sample_games.draw_params(1)                  # deterministic per seed
    assert a != b
    for p in (a, b):
        for name, (lo, hi) in sample_games.PARAM_RANGES.items():
            assert lo <= p[name] <= hi, name
    pinned = sample_games.draw_params(1, ball_noise=0.25)
    assert pinned["ball_noise"] == 0.25


def test_still_ball_survives_tracking_noise():
    """A ball resting on a set-piece spot must read as still even with 0.5 m of per-frame jitter
    (the harsh scenario); a rolling ball must not."""
    rng = np.random.default_rng(0)
    fps = 25
    n = 400
    t = np.arange(n)
    rolling = t >= 200
    bx = np.where(rolling, 50.0 + 6.0 * (t - 200) / fps, 50.0) + 0.5 * rng.standard_normal(n)
    by = 30.0 + 0.5 * rng.standard_normal(n)
    v = features.window_speed(bx, by, fps)
    assert np.nanmedian(v[30:170]) < 1.0            # resting ball
    assert np.nanmedian(v[230:370]) > 4.0           # rolling at 6 m/s


# -- review fixes, round 3 -----------------------------------------------------------------------------

def test_last_sighting_velocity_is_unbiased_at_the_edge_of_a_gap():
    """projected_exit projects from the last sighting, so the velocity there must be the real one
    (a centred difference over a gap halved it)."""
    fps = 25
    n = 60
    t = np.arange(n)
    by = 40.0 + 10.0 * t / fps
    bx = np.full(n, 50.0)
    known = t <= 30
    vx, vy = features.last_sighting_velocity(np.where(known, bx, np.nan),
                                             np.where(known, by, np.nan), fps)
    assert vy[30] == pytest.approx(10.0, rel=0.02)
    assert vx[30] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(vy[31])                                  # no sighting on this frame


def test_window_speed_is_not_slowed_next_to_a_gap():
    fps = 25
    n = 200
    t = np.arange(n)
    bx = 10.0 + 10.0 * t / fps
    by = np.full(n, 30.0)
    bx[100:] = np.nan
    by[100:] = np.nan
    v = features.window_speed(bx, by, fps)
    assert np.all(np.abs(v[90:99] - 10.0) < 0.5), v[90:99]


def test_dead_ball_paths_vary_in_shape():
    """The synthetic dead-ball path must not be one template (a model could learn its timing):
    the still period before the restart varies in length and is sometimes absent."""
    m, _ = _toy_match(13, T=40000)
    t = np.arange(m.n_frames)
    labels = np.where((t // 500) % 2 == 1, "D", "H")
    m.ball_x = np.where(labels == "D", np.nan, 30 + 10 * np.sin(t / 50.0)).astype(np.float32)
    m.ball_y = np.where(labels == "D", np.nan, 30 + 10 * np.cos(t / 70.0)).astype(np.float32)
    filled = sample_games.fill_dead_ball(m, labels, seed=3)
    still_lengths = []
    for a in range(500, m.n_frames - 500, 1000):
        seg_x, seg_y = filled.ball_x[a:a + 500], filled.ball_y[a:a + 500]
        moving = np.hypot(np.diff(seg_x), np.diff(seg_y)) > 1e-6
        still_lengths.append(len(moving) - (np.flatnonzero(moving)[-1] + 1 if moving.any() else 0))
    still_lengths = np.array(still_lengths)
    assert (still_lengths == 0).any()                         # sometimes no still period
    assert len(set(still_lengths[still_lengths > 0])) > 5     # and varied when there is one


# -- possession-equal training weights (hand-over round) ------------------------------------------------

def test_spell_weights_make_every_possession_count_equally():
    from mtah.model import spell_weights
    labels = np.array(["D"] * 10 + ["H"] * 25 + ["A"] * 250 + ["H"] * 100 + ["X"] * 5)
    w = spell_weights(labels, ref_frames=100, lo=0.0, hi=1e9)
    assert np.all(w[:10] == 1.0) and np.all(w[-5:] == 0.0)        # dead unweighted, X unused
    totals = [w[10:35].sum(), w[35:285].sum(), w[285:385].sum()]
    np.testing.assert_allclose(totals, 100.0)                      # each possession sums the same


def test_spell_weights_are_clipped():
    from mtah.model import spell_weights
    labels = np.array(["H"] * 5 + ["A"] * 5000)
    w = spell_weights(labels, ref_frames=100, lo=0.25, hi=4.0)
    assert w[:5].max() == 4.0 and w[5:].min() == 0.25


# -- minimum dead-ball duration in the decoder -----------------------------------------------------------

def _valid_min_dead(path, k):
    """D runs must last >= k frames, except one touching the start or the end of the half."""
    path = list(path)
    i, n = 0, len(path)
    while i < n:
        if path[i] == 2:
            j = i
            while j < n and path[j] == 2:
                j += 1
            if i > 0 and j < n and j - i < k:
                return False
            i = j
        else:
            i += 1
    return True


def test_viterbi_with_min_dead_finds_the_constrained_optimum():
    rng = np.random.default_rng(1)
    for _ in range(25):
        T, k = 8, 3
        scores = rng.normal(size=(T, 3))
        prm = rng.uniform(0, 3, 3)
        best = max(_path_score(scores, np.array(p), *prm)
                   for p in itertools.product(range(3), repeat=T) if _valid_min_dead(p, k))
        path = decode.viterbi(scores, *prm, min_dead=k)
        assert _valid_min_dead(path, k)
        assert _path_score(scores, path, *prm) == pytest.approx(best)


def test_min_dead_closes_the_dead_ball_shortcut_between_possessions():
    """With switch > stop + restart, an unconstrained Viterbi hands possession over through a
    one-frame fake stoppage; with a minimum dead duration it switches directly. (decode() now
    refuses such costs outright; this checks the Viterbi itself.)"""
    n = 400
    p_home = np.where(np.arange(n) < 200, 0.95, 0.05)
    scores = decode.emissions(np.full(n, 0.02), p_home, 1.0, 1.0)
    loose = decode.viterbi(scores, 96, 32, 32, min_dead=1)
    assert 2 in set(loose.tolist())                            # the shortcut
    strict = decode.viterbi(scores, 96, 32, 32, min_dead=25)
    assert set(strict.tolist()) == {0, 1}


def test_decoder_rejects_costs_that_make_a_stoppage_the_cheap_hand_over():
    """A hand-over must be clearly cheaper than stopping and restarting play: at equality a
    slightly raised P(dead) already turns hand-overs into fake 1 s stoppages."""
    for bad in (dict(switch=96, stop=32, restart=32), dict(switch=64, stop=32, restart=32)):
        with pytest.raises(ValueError, match="switch"):
            decode.decode(np.full(10, 0.1), np.full(10, 0.9), [(0, 10)], 10, bad)
    decode.decode(np.full(10, 0.1), np.full(10, 0.9), [(0, 10)], 10,
                  dict(switch=64, stop=40, restart=40))


def test_shipped_decoder_hands_over_directly_through_an_uncertain_moment():
    """Reviewer's scenario: 1 s of coin-flip team evidence and mildly raised P(dead) (0.2) at a
    hand-over must not become a stoppage."""
    n = 600
    t = np.arange(n)
    p_home = np.where(t < 300, 0.95, 0.05)
    p_dead = np.full(n, 0.02)
    around = (t >= 288) & (t < 313)
    p_home[around] = 0.5
    p_dead[around] = 0.2
    labels = decode.decode(p_dead, p_home, [(0, n)], n, config.DECODER)
    assert "D" not in set(labels)


def test_min_dead_viterbi_rejects_non_finite_scores():
    scores = np.zeros((5, 3))
    scores[:, 2] = -np.inf
    with pytest.raises(ValueError, match="finite"):
        decode.viterbi(scores, 1, 1, 1, min_dead=3)


# -- hand-over detector and per-frame switch costs ------------------------------------------------------

def _path_score_var(scores, path, switch, stop, restart):
    """Like _path_score, with switch[t] = cost of switching team INTO frame t."""
    total = scores[np.arange(len(path)), path].sum()
    for t in range(1, len(path)):
        a, b = int(path[t - 1]), int(path[t])
        if {a, b} == {0, 1}:
            total -= switch[t]
        elif b == 2 and a != 2:
            total -= stop
        elif a == 2 and b != 2:
            total -= restart
    return total


def test_viterbi_with_per_frame_switch_costs_finds_the_optimum():
    rng = np.random.default_rng(2)
    for k in (1, 3):
        for _ in range(20):
            T = 7
            scores = rng.normal(size=(T, 3))
            sw = rng.uniform(0, 3, T)
            stop, restart = rng.uniform(0, 3, 2)
            valid = (lambda p: _valid_min_dead(p, k)) if k > 1 else (lambda p: True)
            best = max(_path_score_var(scores, np.array(p), sw, stop, restart)
                       for p in itertools.product(range(3), repeat=T) if valid(p))
            path = decode.viterbi(scores, sw, stop, restart, min_dead=k)
            assert _path_score_var(scores, path, sw, stop, restart) == pytest.approx(best)


def test_hand_over_evidence_lets_a_short_possession_through():
    """A 2 s home possession inside away play, with only moderate frame evidence: a flat switch
    cost swallows it; hand-over evidence at both edges lets the decoder keep it."""
    n = 600
    t = np.arange(n)
    p_home = np.where((t >= 300) & (t < 350), 0.7, 0.05)
    p_dead = np.full(n, 0.02)
    flat = decode.decode(p_dead, p_home, [(0, n)], n, config.DECODER)
    assert "H" not in set(flat)
    p_switch = np.full(n, 0.01)
    p_switch[(np.abs(t - 300) <= 12) | (np.abs(t - 350) <= 12)] = 0.9
    sharp = decode.decode(p_dead, p_home, [(0, n)], n,
                          dict(config.DECODER, hand_w=8.0, hand_q=0.05, hand_floor=8.0),
                          p_switch=p_switch)
    assert np.mean(sharp[300:350] == "H") > 0.9


def test_near_switch_labels():
    from mtah.model import near_switch
    labels = np.array(["H"] * 50 + ["A"] * 50 + ["D"] * 20 + ["H"] * 30)
    y = near_switch(labels, 5)
    assert y[45:55].all() and not y[:44].any() and not y[56:].any()   # D -> H is not a hand-over


def test_hand_over_detector_is_team_symmetric():
    from mtah.model import FrameModels
    items = []
    for s in (0, 1):
        m, lab = _toy_match(s)
        items.append((features.compute(m), lab, m.halves, s))
    models = FrameModels(gb_params=dict(max_iter=20), stride=2, hand_over=True).fit(items)
    m, _ = _toy_match(5)
    p = models.predict_switch(features.compute(m))
    m.team = 1 - m.team
    q = models.predict_switch(features.compute(m))
    assert p is not None and p.shape == (m.n_frames,)
    np.testing.assert_allclose(q, p, atol=1e-9)

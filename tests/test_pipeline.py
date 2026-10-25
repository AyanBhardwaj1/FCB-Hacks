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

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

"""Reading tracking, tagging and writing submissions.

Everything downstream works on a `Match`: per-frame arrays in metres, one column per track slot,
plus the team each slot belongs to. Two loaders produce one:

  load_epts(meta, raw)        Metrica FIFA/EPTS export -- the challenge ATD, and Sample Game 3
  load_metrica_csv(home, away) Metrica's older CSV format -- Sample Games 1 and 2

The EPTS reader follows the DataFormatSpecification blocks in the metadata rather than assuming
"slot i of the line is Player i of the XML": the spec is allowed to change the channel order
between frame ranges (Sample Game 3 does exactly that at a substitution), and reading it is cheap.
"""

import io as _io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1


@dataclass
class Match:
    """One recording. Frame i of every array is video frame i (submission row i)."""

    x: np.ndarray            # [T, P] metres along the pitch length, NaN where not tracked
    y: np.ndarray            # [T, P] metres along the width
    speed: np.ndarray        # [T, P] m/s, NaN where the source has no speed channel
    team: np.ndarray         # [P] HOME or AWAY
    ball_x: np.ndarray       # [T] metres, NaN where the ball is not seen
    ball_y: np.ndarray       # [T]
    pitch: tuple             # (length, width) in metres
    fps: int = 25
    halves: list = field(default_factory=list)   # [(start, stop)] half-open graded frame spans
    name: str = ""

    @property
    def n_frames(self):
        return len(self.ball_x)


# -- EPTS --------------------------------------------------------------------------------------

def _strip(tag):
    return tag.rsplit("}", 1)[-1]


def _find_all(root, name):
    return [el for el in root.iter() if _strip(el.tag) == name]


def read_epts_meta(meta_path):
    """Teams, players, channel map and format blocks from a FIFA/EPTS metadata XML."""
    root = ET.parse(meta_path).getroot()

    teams = [t.get("id") for t in _find_all(root, "Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    team_names = []
    for t in _find_all(root, "Team"):
        name = next((c.text for c in t if _strip(c.tag) == "Name"), None)
        team_names.append(name or t.get("id"))

    players = []      # player ids in metadata order
    player_team = {}
    for p in _find_all(root, "Player"):
        pid = p.get("id")
        if pid is None or p.get("teamId") is None:
            continue
        players.append(pid)
        # First team listed is home -- the FIFA/EPTS convention, and the one the challenge's
        # own baseline relies on.
        player_team[pid] = HOME if p.get("teamId") == teams[0] else AWAY

    channel_player = {}   # playerChannelId -> (player id, channel name)
    for ch in _find_all(root, "PlayerChannel"):
        channel_player[ch.get("id")] = (ch.get("playerId"), ch.get("channelId"))

    pitch = None
    for fs in _find_all(root, "FieldSize"):
        vals = {_strip(c.tag): c.text for c in fs}
        try:
            pitch = (float(vals["Width"]), float(vals["Height"]))
        except (KeyError, TypeError, ValueError):
            pass

    fps = 25
    for fr in _find_all(root, "FrameRate"):
        try:
            fps = int(float(fr.text))
        except (TypeError, ValueError):
            pass

    blocks = []
    for spec in _find_all(root, "DataFormatSpecification"):
        start, end = int(spec.get("startFrame")), int(spec.get("endFrame"))
        slots, ball = [], []
        # Children of the spec, in order: frameCount, the ';'-split of players, the ball group.
        for reg in spec:
            kind = _strip(reg.tag)
            if kind != "SplitRegister":
                continue
            inner = [c for c in reg if _strip(c.tag) == "SplitRegister"]
            if inner:   # the player list
                for group in inner:
                    refs = [c.get("playerChannelId") for c in group
                            if _strip(c.tag) == "PlayerChannelRef"]
                    pid = channel_player[refs[0]][0] if refs and refs[0] in channel_player \
                        else None
                    chans = [channel_player.get(r, (None, r))[1] for r in refs]
                    slots.append((pid, chans))
            else:
                ball = [c.get("channelId") for c in reg if _strip(c.tag) == "BallChannelRef"]
        blocks.append({"start": start, "end": end, "slots": slots, "ball": ball})

    return {"teams": teams, "team_names": team_names, "players": players,
            "player_team": player_team, "pitch": pitch, "fps": fps, "blocks": blocks}


def _parse_lines(lines, n_cols):
    """Numeric array [n, n_cols] from raw EPTS lines, all separators flattened to commas."""
    text = "".join(lines).replace(":", ",").replace(";", ",")
    frame = pd.read_csv(_io.StringIO(text), header=None, na_values=["NaN", "nan", ""],
                        dtype=np.float64, engine="c")
    if frame.shape[1] != n_cols:
        raise ValueError("expected {} fields per line, found {}".format(n_cols, frame.shape[1]))
    return frame.to_numpy()


def _channel_target(chan, x, y, speed):
    """Which array a channel id fills. Tolerant of naming: 'x', 'player3_x', 'speed', 's'."""
    name = (chan or "").lower().rsplit("_", 1)[-1]
    if name == "x":
        return x
    if name == "y":
        return y
    if name in ("s", "speed", "v", "velocity"):
        return speed
    return None


def load_epts(meta_path, raw_path, pitch=None, n_frames=None, name=""):
    """A Match from a FIFA/EPTS pair. frame_id 1 becomes video frame 0.

    `pitch` overrides the metadata's field size (the challenge measured 105.3 x 68.0 m against
    the metadata's 105 x 68). `n_frames` fixes the output length; frames with no record stay NaN.
    """
    meta = read_epts_meta(meta_path)
    pitch = pitch or meta["pitch"] or (105.0, 68.0)
    players = meta["players"]
    col_of = {pid: i for i, pid in enumerate(players)}
    team = np.array([meta["player_team"][pid] for pid in players], dtype=np.int8)

    with open(raw_path) as handle:
        lines = handle.readlines()
    lines = [ln if ln.endswith("\n") else ln + "\n" for ln in lines if ln.strip()]
    ids = np.array([int(ln[:ln.index(":")]) for ln in lines])
    if n_frames is None:
        n_frames = int(ids.max())

    P = len(players)
    x = np.full((n_frames, P), np.nan, dtype=np.float32)
    y = np.full((n_frames, P), np.nan, dtype=np.float32)
    s = np.full((n_frames, P), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    for block in meta["blocks"]:
        sel = np.flatnonzero((ids >= block["start"]) & (ids <= block["end"]))
        if not len(sel):
            continue
        n_cols = 1 + sum(len(ch) for _, ch in block["slots"]) + len(block["ball"])
        data = _parse_lines([lines[i] for i in sel], n_cols)
        rows = data[:, 0].astype(np.int64) - 1
        keep = (rows >= 0) & (rows < n_frames)
        data, rows = data[keep], rows[keep]
        col = 1
        for pid, chans in block["slots"]:
            j = col_of.get(pid)
            for c, chan in enumerate(chans):
                if j is not None:
                    target = _channel_target(chan, x, y, s)
                    if target is not None:
                        target[rows, j] = data[:, col + c]
            col += len(chans)
        for c, chan in enumerate(block["ball"]):
            target = _channel_target(chan, bx, by, None)
            if target is not None:
                target[rows] = data[:, col + c]

    length, width = pitch
    return Match(x=x * length, y=y * width, speed=s, team=team,
                 ball_x=bx * length, ball_y=by * width, pitch=tuple(pitch), fps=meta["fps"],
                 name=name or os.path.basename(raw_path))


# -- Metrica CSV (sample games 1 and 2) --------------------------------------------------------

def _read_metrica_csv(path):
    """(frames, period, {player: (x, y)}, ball (x, y)) from one team's CSV, normalised units."""
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    cols = list(raw.columns)
    frames = raw["Frame"].to_numpy(dtype=np.int64)
    period = raw["Period"].to_numpy(dtype=np.int64)
    players, ball = {}, None
    # Layout: Period, Frame, Time [s], then pairs "PlayerN", "Unnamed" ... and "Ball", "Unnamed".
    i = 3
    while i < len(cols) - 1:
        label = cols[i]
        xy = (raw.iloc[:, i].to_numpy(dtype=np.float64), raw.iloc[:, i + 1].to_numpy(np.float64))
        if label.startswith("Ball"):
            ball = xy
        else:
            players[label] = xy
        i += 2
    return frames, period, players, ball


def load_metrica_csv(home_path, away_path, pitch=(105.0, 68.0), name=""):
    """A Match from Metrica's two-file CSV export. Frame 1 becomes array row 0."""
    f_home, period, home, ball = _read_metrica_csv(home_path)
    f_away, _, away, _ = _read_metrica_csv(away_path)
    if not np.array_equal(f_home, f_away):
        raise ValueError("home and away files are not frame-aligned")
    n = int(f_home.max())
    rows = f_home - 1
    cols = [(HOME, v) for v in home.values()] + [(AWAY, v) for v in away.values()]
    P = len(cols)
    x = np.full((n, P), np.nan, dtype=np.float32)
    y = np.full((n, P), np.nan, dtype=np.float32)
    for j, (_, (px, py)) in enumerate(cols):
        x[rows, j] = px * pitch[0]
        y[rows, j] = py * pitch[1]
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    bx[rows] = ball[0] * pitch[0]
    by[rows] = ball[1] * pitch[1]
    halves = []
    for p in sorted(set(period.tolist())):
        if p < 1:
            continue
        r = rows[period == p]
        halves.append((int(r.min()), int(r.max()) + 1))
    return Match(x=x, y=y, speed=np.full_like(x, np.nan),
                 team=np.array([t for t, _ in cols], dtype=np.int8),
                 ball_x=bx, ball_y=by, pitch=tuple(pitch), halves=halves,
                 name=name or os.path.basename(home_path))


# -- Smart Tagging -----------------------------------------------------------------------------

def load_smart_tagging(path, fps=25):
    """[{code, start, stop, tags}] with frames half-open, codes and tag values lower-cased."""
    out = []
    for inst in ET.parse(path).getroot().iter("instance"):
        start_s, end_s = float(inst.findtext("start")), float(inst.findtext("end"))
        tags = {}
        for label in inst.findall("label"):
            group = (label.findtext("group") or "").strip().lower()
            tags[group] = (label.findtext("text") or "").strip().lower()
        out.append({"code": (inst.findtext("code") or "").strip().lower(),
                    "start": int(round(start_s * fps)), "stop": int(round(end_s * fps)),
                    "tags": tags})
    return out


# -- submission --------------------------------------------------------------------------------

def write_submission(path, labels):
    """frame,label CSV, one row per frame, exactly the challenge contract."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path

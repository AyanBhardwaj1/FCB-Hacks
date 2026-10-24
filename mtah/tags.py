"""Smart Tagging as soft priors for the decoder.

Metrica's Smart Tagging is automatic and unreviewed, and its intervals are phases, not
possessions (DATA.md). So nothing here copies a tag onto a label. Each tag adds a small
per-frame bonus to the decoder's log-scores, and the tracking-based evidence decides.

  possession phases     build up, progression, creating chances, attacking transition, long
                        ball, ball in final third, ball in the box, players in the box
                        -> the tagged team is the one in possession
  defending phases      defending in attacking/middle/defensive third, defensive transition
                        -> the tagged team is defending, so the *other* team has the ball
  set pieces            a restart: dead ball just before it, the tagged team on the ball after
                        it (team "N/A" -- every throw-in -- gives the dead-ball part only)
  goals                 dead ball from the goal until the next kick-off

The output is a [T, 3] array of extra log-scores (columns H, A, D) for decode().
"""

import numpy as np

from .io import AWAY, HOME

# Params
POSSESSION_CODES = ("build up", "progression", "creating chances", "attacking transition",
                    "long ball", "ball in final third", "ball in the box", "players in the box")
DEFENDING_CODES = ("defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition")
W_PHASE = 0.05        # per frame, added to the tagged possessor and taken from the other team
W_SET_TEAM = 0.10     # per frame, possessor bonus from the start of a set-piece tag to 2 s after
W_SET_DEAD = 0.10     # per frame, dead-ball bonus over the first third of a set-piece window
W_GOAL_DEAD = 0.30    # per frame, dead-ball bonus from a goal to the next kick-off
SET_TAIL_S = 2.0


def team_side(value, home_name="", away_name=""):
    """HOME / AWAY / None for a tag's Team value (lower-cased)."""
    v = (value or "").strip().lower()
    if not v or v in ("n/a", "na", "none"):
        return None
    def close(a, b):
        a, b = a.lower(), b.lower()
        return bool(a) and bool(b) and (a in b or b in a or a.split()[0] == b.split()[0])
    if close(v, home_name):
        return HOME
    if close(v, away_name):
        return AWAY
    # the challenge match, if the metadata names are not informative
    if "barcelona" in v:
        return HOME
    if "athletic" in v or "atletic" in v:
        return AWAY
    return None


def priors(events, n_frames, home_name="", away_name="", fps=25, weights=None):
    w = dict(phase=W_PHASE, set_team=W_SET_TEAM, set_dead=W_SET_DEAD, goal_dead=W_GOAL_DEAD)
    w.update(weights or {})
    extra = np.zeros((n_frames, 3))

    def team_bonus(a, b, side, amount):
        a, b = max(0, a), min(n_frames, b)
        if b <= a or side is None:
            return
        extra[a:b, side] += amount
        extra[a:b, 1 - side] -= amount

    kickoffs = sorted(e["start"] for e in events
                      if e["code"] == "set pieces" and "kick off" in e["tags"].get("type", ""))
    for e in events:
        side = team_side(e["tags"].get("team"), home_name, away_name)
        code = e["code"]
        if code in POSSESSION_CODES:
            team_bonus(e["start"], e["stop"], side, w["phase"])
        elif code in DEFENDING_CODES and side is not None:
            team_bonus(e["start"], e["stop"], 1 - side, w["phase"])
        elif code == "set pieces":
            a, b = e["start"], e["stop"]
            team_bonus(a, b + int(SET_TAIL_S * fps), side, w["set_team"])
            dead_end = min(n_frames, a + max(1, (b - a) // 3))
            if dead_end > max(0, a):
                extra[max(0, a):dead_end, 2] += w["set_dead"]
        elif code == "goals":
            nxt = [k for k in kickoffs if k > e["stop"]]
            end = nxt[0] if nxt else min(n_frames, e["stop"] + 60 * fps)
            if end > e["stop"]:
                extra[max(0, e["stop"]):min(n_frames, end), 2] += w["goal_dead"]
    return extra

"""Build a stand-in for the challenge data bundle out of a public sample game.

The real match data is gated, and it must never be sent to a cloud-hosted LLM, so the pipeline is
developed and smoke-tested against this instead: Metrica Sample Game 3, degraded to look like
automatic tracking, written in the challenge's exact layout and format --

    <out>/tactical/BAR-ATH_tactical_atd.xml           FIFA/EPTS metadata, x/y/speed channels
    <out>/tactical/BAR-ATH_tactical_atd.txt           199,319 frames, halves where BAR-ATH's are
    <out>/tactical/BAR-ATH_tactical_smart_tagging.xml SportsCode XML derived from the events

    python scripts/make_fake_bundle.py --out cache/fake_data
    python run.py --data-dir cache/fake_data --team fake

Scores against the public ground truth are meaningless here (different match); the point is that
every file parses, every frame lines up and the output validates.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mtah import config, sample_games  # noqa: E402

N_SLOTS = 46


def write_meta(path, team_of_slot, pitch):
    teams = ("FIFATMA", "FIFATMB")
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", "  <Metadata>",
             "    <GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>",
             "    <Sessions><Session id=\"s\"><FieldSize><Width>{}</Width><Height>{}</Height>"
             "</FieldSize></Session></Sessions>".format(*pitch),
             "    <Teams>",
             '      <Team id="{}"><Name>Barcelona</Name></Team>'.format(teams[0]),
             '      <Team id="{}"><Name>Athletic Club</Name></Team>'.format(teams[1]),
             "    </Teams>", "    <Players>"]
    for j, t in enumerate(team_of_slot):
        lines.append('      <Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                     '<ShirtNumber>{0}</ShirtNumber></Player>'.format(j + 1, teams[t]))
    lines += ["    </Players>", "  </Metadata>", "  <DataFormatSpecifications>",
              '    <DataFormatSpecification endFrame="999999" separator=":" startFrame="1">',
              '      <StringRegister name="frameCount"/>', '      <SplitRegister separator=";">']
    for j in range(len(team_of_slot)):
        lines.append('        <SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="player{}_{}"/>'.format(j + 1, c)
            for c in ("x", "y", "speed")) + "</SplitRegister>")
    lines += ["      </SplitRegister>", '      <SplitRegister separator=",">',
              '        <BallChannelRef channelId="x"/>', '        <BallChannelRef channelId="y"/>',
              '        <BallChannelRef channelId="z"/>',
              "      </SplitRegister>", "    </DataFormatSpecification>",
              "  </DataFormatSpecifications>", "  <Devices><Device><Sensors>"]
    lines.append("  </Sensors></Device></Devices>")
    lines.append("  <PlayerChannels>")
    for j in range(len(team_of_slot)):
        for c in ("x", "y", "speed"):
            lines.append('    <PlayerChannel channelId="{0}" id="player{1}_{0}" playerId="P{1}"/>'
                         .format(c, j + 1))
    lines += ["  </PlayerChannels>", "</main>"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def fmt(v):
    return "NaN" if not np.isfinite(v) else "{:.5f}".format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(config.ROOT, "cache", "fake_data"))
    ap.add_argument("--game", type=int, default=3)
    args = ap.parse_args()

    games, _ = config.challenge_modules()
    n = games.source_frames("tactical", config.GAME)
    target = [(a, b + 1) for _, a, b in games.half_video_blocks(config.GAME)]
    pitch = games.GAMES[config.GAME]["pitch"]

    match, events = sample_games.load_game(args.game)
    labels = sample_games.labels_from_events(events, match.n_frames, match.halves)
    deg = sample_games.degrade(match, labels, seed=7)

    # map each sample half onto the corresponding BAR-ATH half (truncated to fit)
    P = min(N_SLOTS, deg.x.shape[1])
    team = np.concatenate([deg.team[:P], np.zeros(N_SLOTS - P, dtype=np.int8)])
    X = np.full((n, N_SLOTS), np.nan)
    Y = np.full((n, N_SLOTS), np.nan)
    S = np.full((n, N_SLOTS), np.nan)
    BX = np.full(n, np.nan)
    BY = np.full(n, np.nan)
    shift = []
    for (sa, sb), (ta, tb) in zip(deg.halves, target):
        k = min(sb - sa, tb - ta)
        X[ta:ta + k, :P] = deg.x[sa:sa + k, :P] / deg.pitch[0]
        Y[ta:ta + k, :P] = deg.y[sa:sa + k, :P] / deg.pitch[1]
        S[ta:ta + k, :P] = deg.speed[sa:sa + k, :P]
        BX[ta:ta + k] = deg.ball_x[sa:sa + k] / deg.pitch[0]
        BY[ta:ta + k] = deg.ball_y[sa:sa + k] / deg.pitch[1]
        shift.append((sa, sb, ta - sa))

    folder = os.path.join(args.out, "tactical")
    os.makedirs(folder, exist_ok=True)
    write_meta(os.path.join(folder, "BAR-ATH_tactical_atd.xml"), team, pitch)
    with open(os.path.join(folder, "BAR-ATH_tactical_atd.txt"), "w") as fh:
        for i in range(n):
            players = ";".join("{},{},{}".format(fmt(X[i, j]), fmt(Y[i, j]), fmt(S[i, j]))
                               for j in range(N_SLOTS))
            fh.write("{}:{}:{},{},{}\n".format(i + 1, players, fmt(BX[i]), fmt(BY[i]),
                                                 "0.00000" if np.isfinite(BX[i]) else "NaN"))

    # Smart Tagging stand-in: set pieces from the events, possession phases from the labels
    inst = []
    def add(code, a, b, tags):
        inst.append((code, a / 25.0, b / 25.0, tags))
    for e in events:
        if e["type"] == "SET PIECE":
            for sa, sb, d in shift:
                if sa <= e["start"] < sb:
                    f = e["start"] + d
                    side = "N/A" if "THROW" in e["subtype"] else \
                        ("Barcelona" if e["team"] == 0 else "Atletic Club")
                    add("SET PIECES", f - 50, f + 75, {"Team": side, "Type": e["subtype"].title()})
    for sa, sb, d in shift:
        lab = labels[sa:sb]
        brk = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        bounds = np.concatenate(([0], brk, [len(lab)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if lab[a] in "HA" and b - a > 125:
                add("PROGRESSION", sa + a + d + 25, sa + b + d - 25,
                    {"Team": "Barcelona" if lab[a] == "H" else "Atletic Club"})
    with open(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), "w") as fh:
        fh.write('<?xml version="1.0" encoding="utf-8"?>\n<file><ALL_INSTANCES>\n')
        for k, (code, a, b, tg) in enumerate(inst):
            fh.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
                     .format(k + 1, a, b, code))
            for g, v in tg.items():
                fh.write("<label><group>{}</group><text>{}</text></label>".format(g, v))
            fh.write("</instance>\n")
        fh.write("</ALL_INSTANCES></file>\n")
    print("wrote fake bundle to {}".format(args.out))


if __name__ == "__main__":
    main()

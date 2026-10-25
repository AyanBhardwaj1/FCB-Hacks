"""Produce the possession file for the challenge match.

    python run.py --team <team_name>                 # -> submissions/<team_name>.csv
    python run.py --data-dir /path/to/data --out my.csv

Reads the tactical ATD (and, with --tags, the Smart Tagging) from the challenge data bundle,
which must be unpacked as described in the challenge's DATA.md:

    <data-dir>/tactical/BAR-ATH_tactical_atd.xml
    <data-dir>/tactical/BAR-ATH_tactical_atd.txt
    <data-dir>/tactical/BAR-ATH_tactical_smart_tagging.xml

The data dir defaults to $MTAH_DATA_DIR, else ./data. Then validates the file with the
challenge's own validator and, if the public 5-minute ground truth is present, scores it --
as a pipeline check, not a tuning signal.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

from mtah import config, decode, features, tags
from mtah.io import load_epts, load_smart_tagging, read_epts_meta, write_submission

# Params
MIN_BALL_COVERAGE = 0.20   # share of graded frames with a ball; the real ATD has 59.3%
MIN_PLAYERS = 10           # mean tracked players on graded frames; the real ATD has 20.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or
                    os.path.join(config.ROOT, "data"))
    ap.add_argument("--team", default="submission", help="names the output file")
    ap.add_argument("--out", help="output CSV (default submissions/<team>.csv)")
    ap.add_argument("--model", default=config.MODEL_PATH)
    ap.add_argument("--tags", action="store_true",
                    help="add Smart Tagging priors (off by default: no measurable gain in "
                         "validation, see docs/TECHNICAL.md)")
    ap.add_argument("--save-probs", help="also write per-frame probabilities (.npz) here; "
                    "derived from the match data, so keep it out of any public repository")
    args = ap.parse_args()

    games, pm = config.challenge_modules()
    out = args.out or os.path.join(config.ROOT, "submissions", args.team + ".csv")
    paths = games.asset_paths(config.GAME, data_dir=args.data_dir)
    for key in ("atd_meta", "atd_raw"):
        if not os.path.isfile(paths[key]):
            sys.exit("missing {}\nUnpack the challenge data into {} (see the challenge's DATA.md)"
                     " or pass --data-dir.".format(paths[key], args.data_dir))

    n_frames = games.source_frames("tactical", config.GAME)
    halves = [(a, b + 1) for _, a, b in games.half_video_blocks(config.GAME)]

    t = time.time()
    match = load_epts(paths["atd_meta"], paths["atd_raw"], pitch=games.GAMES[config.GAME]["pitch"],
                      n_frames=n_frames, name=config.GAME)
    match.halves = halves
    graded = np.concatenate([np.arange(a, b) for a, b in halves])
    ball_cov = np.isfinite(match.ball_x[graded]).mean()
    players = np.isfinite(match.x[graded]).sum(axis=1).mean()
    print("loaded ATD: {} frames, {} track slots; on graded frames ball seen {:.1%}, {:.1f} "
          "players tracked ({:.0f}s)".format(match.n_frames, match.x.shape[1], ball_cov, players,
                                            time.time() - t))
    # DATA.md: ball on 59.3% of graded frames, ~20.5 players. Far below that means the file was
    # misread (wrong recording, wrong frame counter), and the output would be meaningless.
    if ball_cov < MIN_BALL_COVERAGE or players < MIN_PLAYERS:
        sys.exit("tracking coverage too low (ball {:.1%}, {:.1f} players) -- refusing to write a "
                 "submission from it".format(ball_cov, players))

    feats = features.compute(match)
    with open(args.model, "rb") as fh:
        models = pickle.load(fh)
    p_dead, p_home = models.predict(feats, halves)

    extra = None
    if args.tags and os.path.isfile(paths.get("smart_tagging", "")):
        meta = read_epts_meta(paths["atd_meta"])
        home_name, away_name = meta["team_names"]
        events = load_smart_tagging(paths["smart_tagging"], fps=match.fps)
        extra = tags.priors(events, n_frames, home_name, away_name, fps=match.fps)
        print("Smart Tagging: {} instances used as priors".format(len(events)))

    labels = decode.decode(p_dead, p_home, halves, n_frames, config.DECODER, extra)
    write_submission(out, labels)
    if args.save_probs:
        np.savez_compressed(args.save_probs, p_dead=p_dead, p_home=p_home)

    # the challenge's own checks
    frame = pm.read_label_csv(out)
    error = pm.validate_submission(frame, n_frames)
    if error:
        sys.exit("INVALID: {}".format(error))
    codes = pm.label_codes(frame)
    runs = [(a, b) for a, b in halves]
    stats = pm.possession_stats(codes, runs)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("wrote {}  {}  possessions {} (mean {:.1f} s)  ({:.0f}s total)".format(
        out, counts, stats["n_spells"], stats["mean_duration_s"], time.time() - t))

    gt_path = os.path.join(config.CHALLENGE_REPO, "evaluation", "gt",
                           "{}_possession_gt_public.csv".format(config.GAME))
    if os.path.isfile(gt_path):
        r = pm.evaluate(pm.label_codes(pm.read_label_csv(gt_path)), codes)
        print("public 5-minute window (a pipeline check, not a leaderboard): spellF1 {:.4f}  "
              "macroF1 {:.4f}".format(r["spell_f1_mean"], r["macro_f1"]))


if __name__ == "__main__":
    main()

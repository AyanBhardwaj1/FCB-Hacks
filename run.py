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
import re
import sys
import time

import numpy as np

from mtah import config, decode, features, tags
from mtah.io import load_epts, load_smart_tagging, read_epts_meta, write_submission

# Params
MIN_BALL_COVERAGE = 0.20   # share of graded frames with a ball; the real ATD has 59.3%
MIN_PLAYERS = 10           # mean tracked players on graded frames; the real ATD has 20.5


def git_commit():
    """(HEAD of the repository this code runs from, whether the working tree is clean -- no
    modified or untracked, non-ignored files); ('unknown', False) outside a git checkout."""
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", "-C", config.ROOT] + list(args),
                                       stderr=subprocess.DEVNULL).decode().strip()
    try:
        return git("rev-parse", "HEAD"), git("status", "--porcelain") == ""
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def write_manifest(csv_path, argv, commit=None, clean=None, requirements=()):
    """<csv>.run.json next to the submission: its SHA-256, the exact command, the code commit and
    the software environment -- what the organisers need to regenerate the same file, and what
    scripts/package_submission.py checks before anything is uploaded."""
    import hashlib
    import json
    import shlex

    from mtah import env

    with open(csv_path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    words = [os.path.basename(argv[0])] + list(argv[1:]) if argv else []
    if commit is None:
        commit, clean = git_commit()
    data = {"file": os.path.basename(csv_path), "sha256": sha,
            "command": shlex.join(["python"] + words),
            "commit": commit, "clean": bool(clean),
            "environment": env.describe(*requirements)}
    path = os.path.splitext(csv_path)[0] + ".run.json"
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    return path


def check_inputs(meta_path, raw_path):
    """Stop with a plain reason if an ATD file is empty, blank or not the expected format -- what a
    failed download or unzip leaves behind. Never echoes file contents (they are licensed)."""
    hint = ("the download or unzip failed -- download the file again, on its own rather than as "
            "a folder zip, and check `file <path>` reports text")
    for path, kind in ((meta_path, "xml"), (raw_path, "txt")):
        with open(path, "rb") as fh:
            head = fh.read(4096)
        if not head:
            sys.exit("{} is empty: {}".format(path, hint))
        if not head.strip(b"\0"):
            sys.exit("{} is blank (all zero bytes): {}".format(path, hint))
        text = head.lstrip(b"\xef\xbb\xbf").lstrip()
        if kind == "xml" and not text.startswith(b"<"):
            sys.exit("{} is not an XML file: {}".format(path, hint))
        if kind == "txt" and not re.match(rb"\d+:", text):
            sys.exit("{} does not start with ATD frame records ('<frame_id>:...'): {}".format(
                path, hint))


def video_cache_key(paths, match, halves):
    """Hash of everything the video detections depend on: the video and homography files, the
    detector's code and OpenCV version, the tracked ball, the halves, pitch and frame rate."""
    import hashlib

    import cv2

    from mtah import video_ball

    h = hashlib.sha256()
    for path in (paths["video"], paths["homography"]):
        st = os.stat(path)
        h.update("{}:{}:{}".format(path, st.st_size, int(st.st_mtime)).encode())
    with open(video_ball.__file__, "rb") as fh:
        h.update(fh.read())
    h.update(repr((cv2.__version__, list(map(tuple, halves)), tuple(match.pitch),
                   match.fps)).encode())
    h.update(np.nan_to_num(match.ball_x).tobytes())
    h.update(np.nan_to_num(match.ball_y).tobytes())
    return h.hexdigest()[:16]


def add_video_ball(match, paths, halves):
    """Fill tracker gaps in match.ball_x/_y from the tactical video (mtah/video_ball.py).

    Detections are cached under cache/ (derived from the licensed video: never commit them) and
    merged only if the detector's calibration probes on this video pass video_ball.trustworthy.
    """
    from mtah import video_ball

    for key in ("video", "homography"):
        if not os.path.isfile(paths[key]):
            sys.exit("--video-ball needs {}".format(paths[key]))
    cache = os.path.join(config.CACHE_DIR, "video_ball_{}.npz".format(
        video_cache_key(paths, match, halves)))
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        det_x, det_y = z["det_x"], z["det_y"]
        report = {k: float(z[k]) for k in z.files if k not in ("det_x", "det_y")}
        print("video ball: cached detections from {}".format(cache))
    else:
        Hs, valid = video_ball.load_homographies(paths["homography"], match.n_frames)
        t = time.time()
        print("video ball: searching tracker gaps in {} ...".format(paths["video"]))
        det_x, det_y, report = video_ball.run_video(paths["video"], Hs, valid, match.ball_x,
                                                    match.ball_y, halves, match.pitch, match.fps)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        tmp = cache[:-4] + ".tmp{}.npz".format(os.getpid())
        np.savez_compressed(tmp, det_x=det_x, det_y=det_y, **report)
        os.replace(tmp, cache)          # never a truncated cache at the final path
        print("video ball: done in {:.0f}s".format(time.time() - t))
    print("video ball: read {frames_read:.0f} of {frames_expected:.0f} frames; calibration on "
          "hidden tracked stretches -- ball present: {probe_frames:.0f} frames, found "
          "{probe_found:.0%}, median error {probe_median_err_m:.2f} m, within 1 m "
          "{probe_within_1m:.0%}; ball absent: {absent_frames:.0f} frames, false detections "
          "{absent_fp_rate:.1%}; gap frames searched {gap_frames:.0f}, filled {gap_filled:.0%}"
          .format(**report))
    if not video_ball.trustworthy(report):
        print("video ball: calibration below the trust thresholds -- NOT using the detections")
        return
    bx, by, filled = video_ball.merge(match.ball_x, match.ball_y, det_x, det_y)
    match.ball_x, match.ball_y = bx.astype(np.float32), by.astype(np.float32)
    graded = np.concatenate([np.arange(a, b) for a, b in halves])
    print("video ball: filled {} frames; ball now on {:.1%} of graded frames".format(
        int(filled.sum()), np.isfinite(match.ball_x[graded]).mean()))


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
    ap.add_argument("--video-ball", action="store_true",
                    help="search the tactical video for the ball inside tracker gaps and fill "
                         "them where the detector's self-calibration says it is reliable "
                         "(needs requirements-video.txt; runs locally, ~20-40 min, cached)")
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
    check_inputs(paths["atd_meta"], paths["atd_raw"])

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

    if args.video_ball:
        add_video_ball(match, paths, halves)

    feats = features.compute(match)
    with open(args.model, "rb") as fh:
        models = pickle.load(fh)
    p_dead, p_home = models.predict(feats, halves)
    p_switch = models.predict_switch(feats) if hasattr(models, "predict_switch") else None

    extra = None
    if args.tags and not os.path.isfile(paths.get("smart_tagging", "")):
        print("WARNING: --tags given but no Smart Tagging file at {} -- running without "
              "it".format(paths.get("smart_tagging")))
    if args.tags and os.path.isfile(paths.get("smart_tagging", "")):
        meta = read_epts_meta(paths["atd_meta"])
        home_name, away_name = meta["team_names"]
        events = load_smart_tagging(paths["smart_tagging"], fps=match.fps)
        extra = tags.priors(events, n_frames, home_name, away_name, fps=match.fps)
        print("Smart Tagging: {} instances used as priors".format(len(events)))

    labels = decode.decode(p_dead, p_home, halves, n_frames,
                           config.decoder(hand_over=p_switch is not None), extra,
                           p_switch=p_switch)
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
    from mtah import env
    requirements = [env.REQUIREMENTS] + ([env.REQUIREMENTS_VIDEO] if args.video_ball else [])
    manifest = write_manifest(out, sys.argv, requirements=requirements)
    pins = env.pinned(*requirements)
    drift = env.mismatches(pins, env.installed(pins))
    print("manifest {}{}".format(manifest, "" if not drift else
                                 "\nWARNING: environment differs from the pinned one ({}); the "
                                 "organisers may not regenerate this exact file".format(
                                     "; ".join(drift))))

    gt_path = os.path.join(config.CHALLENGE_REPO, "evaluation", "gt",
                           "{}_possession_gt_public.csv".format(config.GAME))
    if os.path.isfile(gt_path):
        r = pm.evaluate(pm.label_codes(pm.read_label_csv(gt_path)), codes)
        print("public 5-minute window (a pipeline check, not a leaderboard): spellF1 {:.4f}  "
              "macroF1 {:.4f}".format(r["spell_f1_mean"], r["macro_f1"]))


if __name__ == "__main__":
    main()

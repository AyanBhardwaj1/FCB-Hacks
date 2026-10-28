"""Evaluation harness: how well does the pipeline do the task, with error bars.

Leave-one-game-out on Metrica's public sample games. For each held-out game the frame models are
trained on the other two (default degradation, --train-seeds), then scored on the held-out game
under several unseen degradation seeds and two scenarios:

  default  degradation calibrated to the challenge ATD (ball missing ~41% of graded frames)
  harsh    worse than we expect the real data to be: more ball loss, more player dropout,
           more noise and false ball detections -- a robustness check, not a target

Every (scenario, game, seed) unit is scored with the challenge's own possession_metrics. Results
go to cache/eval/<tag>.json; --compare A B prints the paired difference between two runs, which
is the only fair way to say one version beats another on the same test sets.

    python scripts/evaluate.py --tag base
    python scripts/evaluate.py --tag no_shape --exclude '_cx|_sx'
    python scripts/evaluate.py --compare base no_shape
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from common import CODE_HASH, score, unit_features  # noqa: E402
from mtah import config, decode, sample_games  # noqa: E402
from mtah.model import FrameModels  # noqa: E402

EVAL_DIR = os.path.join(config.CACHE_DIR, "eval")


def run(args):
    t0 = time.time()
    units = []
    probs = {}
    for held in sample_games.GAMES:
        train = [unit_features(g, s, args.train_scenario) + (g,)
                 for g in sample_games.GAMES if g != held for s in args.train_seeds]
        models = FrameModels(stage2=args.stage2, exclude=args.exclude, hand_over=args.hand_over,
                             spell_weighting=args.spell_weighting).fit(train)
        del train
        for scenario in args.scenarios:
            for seed in args.test_seeds:
                feats, labels, halves = unit_features(held, seed, scenario)
                p_dead, p_home = models.predict(feats, halves)
                p_switch = models.predict_switch(feats)
                pred = decode.decode(p_dead, p_home, halves, len(labels),
                                     dict(config.decoder(args.hand_over), **(args.decoder or {})),
                                     p_switch=p_switch)
                spell, macro, n_pred, n_gt = score(labels, pred)
                units.append(dict(scenario=scenario, game=held, seed=seed, spell=spell,
                                  macro=macro, n_pred=n_pred, n_gt=n_gt))
                probs["{}_{}_{}".format(scenario, held, seed)] = (p_dead, p_home, p_switch)
                print("  {:<7} game {} seed {}: spellF1 {:.4f}  macroF1 {:.4f}  ({}/{} spells)"
                      .format(scenario, held, seed, spell, macro, n_pred, n_gt), flush=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    # record the fully resolved decoder, not just the overrides: config can change later, and
    # the analysis scripts re-decode with whatever a run records
    run_args = dict(vars(args), decoder=dict(config.decoder(args.hand_over),
                                             **(args.decoder or {})))
    with open(os.path.join(EVAL_DIR, args.tag + ".json"), "w") as fh:
        json.dump(dict(args=run_args, code=CODE_HASH, units=units), fh, indent=1)
    with open(os.path.join(EVAL_DIR, args.tag + "_probs.pkl"), "wb") as fh:
        pickle.dump(probs, fh, protocol=4)
    summarize(units, args.tag)
    print("({:.0f}s)".format(time.time() - t0))


def rescore(args):
    """Re-decode a saved run's probabilities with other decoder costs; no retraining."""
    src = args.rescore
    with open(os.path.join(EVAL_DIR, src + ".json")) as fh:
        run_meta = json.load(fh)
    if run_meta.get("code") != CODE_HASH:
        sys.exit("run {!r} was made with different code; re-run it instead".format(src))
    with open(os.path.join(EVAL_DIR, src + "_probs.pkl"), "rb") as fh:
        probs = pickle.load(fh)
    prm = dict(config.decoder(run_meta["args"].get("hand_over", False)), **(args.decoder or {}))
    units = []
    for key, (p_dead, p_home, *rest) in sorted(probs.items()):
        p_switch = rest[0] if rest else None
        scenario, game, seed = key.rsplit("_", 2)
        _, labels, halves = unit_features(int(game), int(seed), scenario)
        pred = decode.decode(p_dead, p_home, halves, len(labels), prm, p_switch=p_switch)
        spell, macro, n_pred, n_gt = score(labels, pred)
        units.append(dict(scenario=scenario, game=int(game), seed=int(seed), spell=spell,
                          macro=macro, n_pred=n_pred, n_gt=n_gt))
    run_args = dict(run_meta["args"], decoder=prm, rescored_from=src)
    with open(os.path.join(EVAL_DIR, args.tag + ".json"), "w") as fh:
        json.dump(dict(args=run_args, code=CODE_HASH, units=units), fh, indent=1)
    with open(os.path.join(EVAL_DIR, args.tag + "_probs.pkl"), "wb") as fh:
        pickle.dump(probs, fh, protocol=4)
    summarize(units, args.tag)


def summarize(units, tag):
    print("\n{}".format(tag))
    for scenario in sorted({u["scenario"] for u in units}):
        sel = [u for u in units if u["scenario"] == scenario]
        s = np.array([u["spell"] for u in sel])
        m = np.array([u["macro"] for u in sel])
        print("  {:<7} spellF1 {:.4f} ± {:.4f} (sd over {} units)   macroF1 {:.4f} ± {:.4f}"
              .format(scenario, s.mean(), s.std(ddof=1) if len(s) > 1 else 0, len(s),
                      m.mean(), m.std(ddof=1) if len(m) > 1 else 0))


def compare(a, b):
    load = lambda t: json.load(open(os.path.join(EVAL_DIR, t + ".json")))["units"]
    ua = {(u["scenario"], u["game"], u["seed"]): u for u in load(a)}
    ub = {(u["scenario"], u["game"], u["seed"]): u for u in load(b)}
    for scenario in sorted({k[0] for k in ua} & {k[0] for k in ub}):
        keys = sorted(k for k in ua if k in ub and k[0] == scenario)
        for metric in ("spell", "macro"):
            d = np.array([ub[k][metric] - ua[k][metric] for k in keys])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print("  {:<7} {:<5} {} - {}: {:+.4f} ± {:.4f} (paired SE, {} units, {} better in {})"
                  .format(scenario, metric, b, a, d.mean(), se, len(d), b, int((d > 0).sum())))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="current")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(config.TRAIN_SEEDS))
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[101, 102, 103],
                    help="degradation seeds of the held-out games; 100 is reserved for tuning "
                         "the decoder in crossval.py")
    ap.add_argument("--scenarios", nargs="+", default=["default", "harsh"])
    ap.add_argument("--train-scenario", default="train",
                    help="degradation of the training games: 'train' randomizes the noise "
                         "per seed, 'default' pins it (the setting before randomization)")
    ap.add_argument("--exclude", help="regex of feature names to leave out")
    ap.add_argument("--spell-weighting", dest="spell_weighting", action="store_true",
                    default=config.SPELL_WEIGHTING,
                    help="possession-equal training weights for the team model (default: as "
                         "shipped, config.SPELL_WEIGHTING)")
    ap.add_argument("--no-spell-weighting", dest="spell_weighting", action="store_false")
    ap.add_argument("--hand-over", dest="hand_over", action="store_true", default=config.HAND_OVER,
                    help="train the hand-over detector and use it in decoding (default: as "
                         "shipped, config.HAND_OVER)")
    ap.add_argument("--no-hand-over", dest="hand_over", action="store_false")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--decoder", type=json.loads, default=None,
                    help='decoder costs as JSON, e.g. \'{"switch": 16}\' (default: config)')
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--rescore", metavar="RUN",
                    help="re-decode saved run RUN with --decoder / config costs, save as --tag")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.rescore:
        rescore(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

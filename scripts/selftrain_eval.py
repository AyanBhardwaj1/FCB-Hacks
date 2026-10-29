"""Does self-training on the target match help? Leave-one-game-out, paired against the base model.

For each held-out game and test unit: the base models label it and the decoder turns that into
possessions; those decoded labels become pseudo-labels (minus the frames near a decoded change
of state and the frames where the frame model disagrees with the decoder); the models are refit
on the training games plus the pseudo-labelled target, which then gets labelled again. No human
decides any pseudo-label, which RULES.md explicitly allows on the provided game.

    python scripts/selftrain_eval.py --tag st --test-seeds 101 102 103
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from common import CODE_HASH, score, unit_features  # noqa: E402
from mtah import config, decode, sample_games  # noqa: E402
from mtah.model import FrameModels  # noqa: E402


def pseudo_labels(pred, p_dead, p_home, margin, agree):
    """Decoded labels as training targets, 'X' (ignored by fit) where they are least reliable:
    within `margin` frames of a decoded change of state, and where the frame model's own call
    disagrees with the decoder by more than `agree`."""
    out = pred.copy()
    change = np.flatnonzero(pred[1:] != pred[:-1]) + 1
    near = np.zeros(len(pred), dtype=bool)
    for c in change:
        near[max(0, c - margin):c + margin] = True
    out[near] = "X"
    alive = np.isin(out, ["H", "A"])
    side = np.where(out == "H", p_home, 1 - p_home)
    out[alive & ((side < agree) | (p_dead > 0.5))] = "X"
    out[(out == "D") & (p_dead < 1 - agree)] = "X"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="selftrain")
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--scenarios", nargs="+", default=["default", "harsh"])
    ap.add_argument("--margin", type=int, default=25, help="frames ignored around a change")
    ap.add_argument("--agree", type=float, default=0.6)
    ap.add_argument("--repeat", type=int, default=2, help="copies of the pseudo-labelled target")
    args = ap.parse_args()
    dec = config.decoder()
    t0 = time.time()
    units = []
    for held in sample_games.GAMES:
        train = [unit_features(g, s, "train") + (g,)
                 for g in sample_games.GAMES if g != held for s in config.TRAIN_SEEDS]
        base = FrameModels(spell_weighting=config.SPELL_WEIGHTING).fit(train)
        for scenario in args.scenarios:
            for seed in args.test_seeds:
                feats, labels, halves = unit_features(held, seed, scenario)
                p_dead, p_home = base.predict(feats, halves)
                pred0 = decode.decode(p_dead, p_home, halves, len(labels), dec)
                pseudo = pseudo_labels(pred0, p_dead, p_home, args.margin, args.agree)
                kept = np.isin(pseudo, ["H", "A", "D"]).mean() / max(
                    np.isin(pred0, ["H", "A", "D"]).mean(), 1e-9)
                acc = (pseudo[np.isin(pseudo, ["H", "A", "D"])] ==
                       labels[np.isin(pseudo, ["H", "A", "D"])]).mean()
                adapted = FrameModels(spell_weighting=config.SPELL_WEIGHTING).fit(
                    train + [(feats, pseudo, halves, "target")] * args.repeat)
                q_dead, q_home = adapted.predict(feats, halves)
                pred1 = decode.decode(q_dead, q_home, halves, len(labels), dec)
                s0, m0, _, _ = score(labels, pred0)
                s1, m1, _, _ = score(labels, pred1)
                units.append(dict(scenario=scenario, game=held, seed=seed, base=s0, adapted=s1,
                                  base_macro=m0, adapted_macro=m1, kept=kept, pseudo_acc=acc))
                print("  {:<7} game {} seed {}: spellF1 {:.4f} -> {:.4f}  macroF1 {:.4f} -> "
                      "{:.4f}  (pseudo kept {:.0%}, {:.1%} right)  [{:.0f}s]".format(
                          scenario, held, seed, s0, s1, m0, m1, kept, acc, time.time() - t0),
                      flush=True)
        del train
    for scenario in args.scenarios:
        u = [x for x in units if x["scenario"] == scenario]
        d = np.array([x["adapted"] - x["base"] for x in u])
        print("{}: base {:.4f}  adapted {:.4f}  paired diff {:+.4f} ± {:.4f} (sem)  better in "
              "{}/{}".format(scenario, np.mean([x["base"] for x in u]),
                             np.mean([x["adapted"] for x in u]), d.mean(),
                             d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)))
    os.makedirs(os.path.join(config.CACHE_DIR, "eval"), exist_ok=True)
    with open(os.path.join(config.CACHE_DIR, "eval", args.tag + ".json"), "w") as fh:
        json.dump(dict(args=vars(args), code=CODE_HASH, units=units), fh, indent=1)


if __name__ == "__main__":
    main()

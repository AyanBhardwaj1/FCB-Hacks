"""Paths and the decoder settings chosen by leave-one-game-out validation on the sample games.

The decoder costs are the only tuned numbers in the pipeline, and they were chosen on Metrica's
public sample games (scripts/crossval.py), never on the competition match.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE_REPO = os.path.join(ROOT, "external", "more-than-a-hack-2027")
MODEL_PATH = os.path.join(ROOT, "models", "frame_models.pkl")
CACHE_DIR = os.path.join(ROOT, "cache")

GAME = "BAR-ATH"

# Chosen on degradation seed 100 for the possession-weighted model, under the rule
# switch <= stop + restart - 16 and a 1 s minimum stoppage (see mtah/decode.py): level with the
# best setting the rule allows (0.7426 vs 0.7427 spellF1). Reported results use fresh seeds.
# See docs/TECHNICAL.md.
DECODER = dict(a_dead=0.75, a_team=2.0, switch=64.0, stop=40.0, restart=40.0, min_dead=25)

# Possession-equal training weights for the team model (mtah/model.spell_weights): spellF1 scores
# every possession once, so training should too.
SPELL_WEIGHTING = True

# Train the hand-over detector (mtah/model.near_switch) and let it lower the decoder's switch cost
# where a possession change is likely (mtah/decode.switch_costs). Off: on the design seeds it did
# not improve spellF1 (docs/DEVLOG.md).
HAND_OVER = False

# Detector weights tuned on degradation seed 100; applied by decoder() whenever the detector is on.
HAND_DECODER = dict(hand_w=24.0, hand_floor=24.0, hand_q=0.05)


def decoder(hand_over=None):
    """The decoder settings to use: DECODER, plus HAND_DECODER when the hand-over detector is on
    (HAND_OVER unless overridden). One place, so turning the detector on can never leave
    decoding ignoring it."""
    on = HAND_OVER if hand_over is None else hand_over
    return dict(DECODER, **(HAND_DECODER if on else {}))

# Degradation seeds used to build the training set from each sample game. Four rather than two:
# more varied noise for the same three matches (see docs/TECHNICAL.md).
TRAIN_SEEDS = (1, 2, 3, 4)


def challenge_modules():
    """Import the challenge repository's games.py and evaluation modules (git submodule)."""
    for sub in ("", "evaluation"):
        path = os.path.join(CHALLENGE_REPO, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    if not os.path.isfile(os.path.join(CHALLENGE_REPO, "games.py")):
        raise IOError("challenge repository missing at {} -- run "
                      "`git submodule update --init`".format(CHALLENGE_REPO))
    import games
    import possession_metrics
    return games, possession_metrics

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

# Chosen by scripts/crossval.py on degradation seed 100: within 0.002 spellF1 of the best setting
# for both feature sets (plateau 0.730-0.735). Reported results use fresh seeds. See
# docs/TECHNICAL.md.
DECODER = dict(a_dead=1.0, a_team=2.0, switch=48.0, stop=32.0, restart=32.0)

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

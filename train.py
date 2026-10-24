"""Train the frame models on Metrica's public sample games and save them to models/.

    python train.py

Needs the sample data (fetched automatically into external/sample-data if missing, ~230 MB).
Deterministic: same data, same pinned library versions, same model file.
"""

import argparse
import os
import pickle
import subprocess
import sys
import time

from mtah import config, features, sample_games
from mtah.model import FrameModels

SAMPLE_REPO = "https://github.com/metrica-sports/sample-data"


def ensure_sample_data():
    if os.path.isdir(sample_games.DEFAULT_DIR):
        return
    target = os.path.dirname(sample_games.DEFAULT_DIR)
    print("fetching Metrica sample data into {} ...".format(target))
    subprocess.check_call(["git", "clone", "--depth", "1", SAMPLE_REPO, target])


def training_items(seeds):
    items = []
    for g in sample_games.GAMES:
        match, events = sample_games.load_game(g)
        labels = sample_games.labels_from_events(events, match.n_frames, match.halves)
        for s in seeds:
            deg = sample_games.degrade(match, labels, seed=s)
            items.append((features.compute(deg), labels, deg.halves, g))
            print("  sample game {} seed {}: {} frames".format(g, s, match.n_frames))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=config.MODEL_PATH)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.TRAIN_SEEDS))
    args = ap.parse_args()

    ensure_sample_data()
    t = time.time()
    items = training_items(args.seeds)
    models = FrameModels().fit(items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(models, fh, protocol=4)
    print("saved {} ({:.0f}s)".format(args.out, time.time() - t))


if __name__ == "__main__":
    sys.exit(main())

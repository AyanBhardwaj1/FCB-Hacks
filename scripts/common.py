"""Shared helpers for the validation scripts: scoring and cached degraded sample-game features."""

import hashlib
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtah import config, features, sample_games  # noqa: E402

_, pm = config.challenge_modules()

SCENARIOS = {
    "default": {},
    "harsh": dict(ball_missing=0.55, player_missing=0.15, ball_noise=0.5,
                  player_noise=0.25, spike_rate=0.006, fragment_every_s=15.0),
}

# Everything a cached (features, labels) file depends on: change any of these and the cache key
# changes with it, so a stale file can never be served.
_SOURCES = [os.path.join(ROOT, "mtah", n) for n in ("features.py", "sample_games.py", "io.py")]


def _code_hash():
    h = hashlib.sha256()
    for path in _SOURCES:
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def score(labels_true, labels_pred):
    """(spellF1, macroF1, n_pred, n_gt) exactly as the official scorer computes them."""
    gt, pred = pm.label_codes(labels_true), pm.label_codes(labels_pred)
    runs = pm.valid_runs(gt)
    spell = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
    return spell["mean_f1"], macro, spell["n_pred"], spell["n_gt"]


def unit_features(game, seed, scenario="default"):
    """(feats, labels, halves) for one degraded copy of a sample game, cached on disk."""
    kwargs = SCENARIOS[scenario]
    key = hashlib.sha256(json.dumps([_code_hash(), scenario, kwargs, game, seed],
                                    sort_keys=True).encode()).hexdigest()[:16]
    folder = os.path.join(config.CACHE_DIR, "sample")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "{}_g{}_s{}_{}.pkl".format(scenario, game, seed, key))
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    match, events = sample_games.load_game(game)
    labels = sample_games.labels_from_events(events, match.n_frames, match.halves)
    deg = sample_games.degrade(match, labels, seed=seed, **kwargs)
    out = (features.compute(deg), labels, deg.halves)
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out

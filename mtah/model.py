"""The frame-level classifiers, trained on other matches.

Stage 1, from the tracking features:
  dead model        P(ball dead | frame)            -- shared features + both team blocks
  possession model  P(home has it | ball alive)     -- one team block + shared features

Stage 2 (optional, off by default -- it did not beat stage 1 alone in leave-one-game-out
validation, 0.686 vs 0.690 spellF1), from stage 1's own output: the same two questions asked again of the stage-1
probabilities seen through windows of 0.2-8 s before, after and around each frame. A frame in
the middle of a possession and a frame at a hand-over can look identical on their own; their
neighbourhoods do not. Stage 2 is trained on out-of-fold stage-1 output (each training game
predicted by a stage-1 model that never saw it), so it learns how wrong stage 1 really is.

All models are gradient-boosted trees (scikit-learn's HistGradientBoostingClassifier: NaN-native,
deterministic with early stopping off and a fixed seed).

Symmetry is built in rather than hoped for. Each training frame is used twice, once as seen by
home and once as seen by away with the label flipped, and at prediction time

    P(H) = ( f(view_home) + 1 - f(view_away) ) / 2

so swapping the two teams' tracking swaps the answer exactly. The model cannot learn "home
usually has the ball" from three sample matches and carry it to Barcelona.
"""

import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import frames_since, matrices, rolling_mean, side_means

# Params
GB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=400,
                 l2_regularization=1.0, early_stopping=False, random_state=0)
STAGE2_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=400,
                     l2_regularization=1.0, early_stopping=False, random_state=0)
TRAIN_STRIDE = 4      # frames; neighbouring frames are near-duplicates, so subsample
SPELL_REF_S = 8.0     # s; possession-equal weights are 1 for a possession this long...
SPELL_W_LO, SPELL_W_HI = 0.25, 4.0   # ...and clipped to this range
CONTEXT_S = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
SIDE_S = (0.5, 1.0, 2.0, 4.0)
EPS = 1e-4


def _dead_design(xh, xa, xs):
    return np.hstack([xs, xh, xa]), np.hstack([xs, xa, xh])


def _poss_design(xh, xa, xs):
    return np.hstack([xh, xs]), np.hstack([xa, xs])


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def context(p_dead, p_home, halves, fps=25):
    """Stage-2 inputs: [T, k] for the dead question and [T, k] for the home question.

    Built per half so no window reaches across halftime. The home features are all odd in
    logit(p_home), so negating them is exactly "swap the teams".
    """
    T = len(p_dead)
    ld, lh = _logit(p_dead), _logit(p_home)
    blocks_d = [np.full(T, np.nan) for _ in range(2 + 2 * len(CONTEXT_S) + 4 * len(SIDE_S) + 2)]
    blocks_h = [np.full(T, np.nan) for _ in range(1 + len(CONTEXT_S) + 3 * len(SIDE_S))]
    for a, b in halves:
        d, h = ld[a:b], lh[a:b]
        out_d = [d, np.abs(h)]
        out_h = [h]
        for s in CONTEXT_S:
            w = max(1, int(round(s * fps / 2)))
            out_d.append(rolling_mean(d, w))
            out_d.append(rolling_mean(np.abs(h), w))
            out_h.append(rolling_mean(h, w))
        for s in SIDE_S:
            w = max(1, int(round(s * fps)))
            db, da = side_means(d, w)
            hb, ha = side_means(h, w)
            out_d += [db, da, np.abs(hb), np.abs(ha)]
            out_h += [hb, ha, ha - hb]
        # how long the dead model has been confident, either way
        conf_dead = (d > 0).astype(bool)
        out_d.append(frames_since(~conf_dead, 30 * fps) / fps)
        out_d.append(frames_since(conf_dead, 30 * fps) / fps)
        for blk, col in zip(blocks_d, out_d):
            blk[a:b] = col
        for blk, col in zip(blocks_h, out_h):
            blk[a:b] = col
    return np.column_stack(blocks_d).astype(np.float32), np.column_stack(blocks_h).astype(np.float32)


def spell_weights(labels, ref_frames, lo=SPELL_W_LO, hi=SPELL_W_HI):
    """Per-frame training weight that makes every possession count the same.

    spellF1 scores each possession once, however long it lasts, but a frame-level loss lets a
    30 s possession outweigh a 3 s one ten to one -- so the model learns to ignore short
    possessions, which are half of all possessions. Each frame of an H or A run of n frames gets
    ref_frames / n (a possession of ref_frames keeps weight 1), clipped to [lo, hi] so one
    two-frame blip cannot dominate. Dead-ball frames keep weight 1; X frames get 0.
    """
    labels = np.asarray(labels)
    w = np.where(labels == "X", 0.0, 1.0)
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if labels[a] in ("H", "A"):
            w[a:b] = np.clip(ref_frames / float(b - a), lo, hi)
    return w


class Stage1:
    def __init__(self, gb_params=None, stride=TRAIN_STRIDE, exclude=None, spell_weighting=False):
        self.gb_params = dict(GB_PARAMS, **(gb_params or {}))
        self.stride = stride
        self.exclude = exclude   # regex: feature names to leave out (ablations)
        self.names = None        # (team names, shared names) the models were trained on
        self.spell_weighting = spell_weighting   # possession-equal weights for the team model

    def _matrices(self, feats):
        """(xh, xa, xs), columns picked by name in the order the models were trained on.

        The first call (in fit) fixes the column lists, minus anything matching `exclude`.
        Later calls select those names and raise if any is missing, so a model can never be fed
        a feature set it was not trained on -- not even one with the same column count.
        """
        nt, xh, xa, ns, xs = matrices(feats)
        if getattr(self, "names", None) is None:
            pat = re.compile(self.exclude) if self.exclude else None
            self.names = ([n for n in nt if not (pat and pat.search(n))],
                          [n for n in ns if not (pat and pat.search(n))])
        want_t, want_s = self.names
        missing = sorted(set(want_t) - set(nt)) + sorted(set(want_s) - set(ns))
        if missing:
            raise ValueError("features missing for this model (retrain with train.py?): {}"
                             .format(", ".join(missing[:10])))
        it = {n: i for i, n in enumerate(nt)}
        js = {n: i for i, n in enumerate(ns)}
        cols_t = [it[n] for n in want_t]
        cols_s = [js[n] for n in want_s]
        return xh[:, cols_t], xa[:, cols_t], xs[:, cols_s]

    def fit(self, items):
        """items: [(feats, labels)] where labels is an array of 'H'/'A'/'D'/'X' per frame."""
        dX, dy, pX, py, pw = [], [], [], [], []
        for feats, labels in items:
            xh, xa, xs = self._matrices(feats)
            sel = np.flatnonzero(np.isin(labels, ["H", "A", "D"]))[::self.stride]
            lab = labels[sel]
            w_all = spell_weights(labels, SPELL_REF_S * 25) if self.spell_weighting else \
                np.ones(len(labels))
            d1, d2 = _dead_design(xh[sel], xa[sel], xs[sel])
            dead = (lab == "D").astype(np.int8)
            dX += [d1, d2]
            dy += [dead, dead]
            alive = lab != "D"
            p1, p2 = _poss_design(xh[sel][alive], xa[sel][alive], xs[sel][alive])
            home = (lab[alive] == "H").astype(np.int8)
            pX += [p1, p2]
            py += [home, 1 - home]
            pw += [w_all[sel][alive]] * 2
        self.dead = HistGradientBoostingClassifier(**self.gb_params).fit(
            np.vstack(dX), np.concatenate(dy))
        self.poss = HistGradientBoostingClassifier(**self.gb_params).fit(
            np.vstack(pX), np.concatenate(py), sample_weight=np.concatenate(pw))
        return self

    def predict(self, feats):
        xh, xa, xs = self._matrices(feats)
        d1, d2 = _dead_design(xh, xa, xs)
        p_dead = 0.5 * (self.dead.predict_proba(d1)[:, 1] + self.dead.predict_proba(d2)[:, 1])
        p1, p2 = _poss_design(xh, xa, xs)
        p_home = 0.5 * (self.poss.predict_proba(p1)[:, 1] + 1 - self.poss.predict_proba(p2)[:, 1])
        return p_dead, p_home


class FrameModels:
    """Stage 1 + stage 2. fit() takes (feats, labels, halves, group) items; `group` names the
    match, and stage 2 is trained on stage-1 predictions made by models that left that match out.
    """

    def __init__(self, gb_params=None, stride=TRAIN_STRIDE, stage2=False, exclude=None,
                 spell_weighting=False):
        self.gb_params = gb_params
        self.stride = stride
        self.use_stage2 = stage2
        self.exclude = exclude
        self.spell_weighting = spell_weighting

    def fit(self, items):
        self.stage1 = Stage1(self.gb_params, self.stride, self.exclude,
                             self.spell_weighting).fit([(f, l) for f, l, _, _ in items])
        if not self.use_stage2:
            return self
        groups = sorted({g for _, _, _, g in items})
        dX, dy, hX, hy, hw = [], [], [], [], []
        for g in groups:
            inner = Stage1(self.gb_params, self.stride, self.exclude, self.spell_weighting).fit(
                [(f, l) for f, l, _, gg in items if gg != g])
            for f, l, halves, gg in items:
                if gg != g:
                    continue
                p_dead, p_home = inner.predict(f)
                cd, ch = context(p_dead, p_home, halves)
                sel = np.flatnonzero(np.isin(l, ["H", "A", "D"]))[::self.stride]
                lab = l[sel]
                dead = (lab == "D").astype(np.int8)
                # swapping teams negates every home feature; the dead block holds |.| only
                dX.append(cd[sel])
                dy.append(dead)
                alive = sel[lab != "D"]
                home = (l[alive] == "H").astype(np.int8)
                hX += [ch[alive], -ch[alive]]
                hy += [home, 1 - home]
                w = spell_weights(l, SPELL_REF_S * 25)[alive] if self.spell_weighting else \
                    np.ones(len(alive))
                hw += [w, w]
        self.dead2 = HistGradientBoostingClassifier(**STAGE2_PARAMS).fit(
            np.vstack(dX), np.concatenate(dy))
        self.home2 = HistGradientBoostingClassifier(**STAGE2_PARAMS).fit(
            np.vstack(hX), np.concatenate(hy), sample_weight=np.concatenate(hw))
        return self

    def predict(self, feats, halves):
        """(p_dead, p_home) per frame. p_home is P(home has it | ball alive)."""
        p_dead, p_home = self.stage1.predict(feats)
        if not self.use_stage2:
            return p_dead, p_home
        cd, ch = context(p_dead, p_home, halves)
        q_dead = self.dead2.predict_proba(cd)[:, 1]
        q_home = 0.5 * (self.home2.predict_proba(ch)[:, 1] + 1 - self.home2.predict_proba(-ch)[:, 1])
        # outside the halves stage 2 has no input; keep stage 1 there
        inside = np.zeros(len(p_dead), dtype=bool)
        for a, b in halves:
            inside[a:b] = True
        return np.where(inside, q_dead, p_dead), np.where(inside, q_home, p_home)

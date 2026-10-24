"""From per-frame probabilities to possessions: a three-state Viterbi decode per half.

A possession is an interval, and the ranking metric scores intervals, so the last step has to
produce clean ones. Frame probabilities flicker; the decoder pays a fixed price for every change
of state and keeps the labelling that best explains the probabilities net of those prices. That
is the "controlled possession" rule expressed as a cost: a deflection -- a few frames of an
opponent near the ball -- does not buy a change of possession, sustained control does.

States 0 = H, 1 = A, 2 = D. Emission scores are tempered log-probabilities:

    H: a_dead * log(1 - p_dead) + a_team * log(p_home)
    A: a_dead * log(1 - p_dead) + a_team * log(1 - p_home)
    D: a_dead * log(p_dead)

Transition costs: switch (H <-> A), stop (H/A -> D), restart (D -> H/A).
"""

import numpy as np

LABELS = np.array(["H", "A", "D"])

DEFAULT = dict(a_dead=1.0, a_team=1.0, switch=12.0, stop=12.0, restart=12.0)


def emissions(p_dead, p_home, a_dead, a_team, eps=1e-4):
    pd_ = np.clip(p_dead, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    alive = a_dead * np.log(1 - pd_)
    return np.column_stack([alive + a_team * np.log(ph), alive + a_team * np.log(1 - ph),
                            a_dead * np.log(pd_)])


def viterbi(scores, switch, stop, restart):
    """Best state path for [T, 3] emission scores under the transition costs.

    A plain-float loop: with three states, numpy's per-call overhead costs more than the maths.
    """
    T = len(scores)
    sh, sa, sd = (scores[:, k].tolist() for k in range(3))
    back = bytearray(3 * T)
    h, a, d = sh[0], sa[0], sd[0]
    for t in range(1, T):
        # best predecessor of each state (ties go to staying put, then to the lower index)
        hh, ah, dh = h, a - switch, d - restart
        if hh >= ah and hh >= dh:
            nh, bh = hh, 0
        elif ah >= dh:
            nh, bh = ah, 1
        else:
            nh, bh = dh, 2
        ha, aa, da = h - switch, a, d - restart
        if aa >= ha and aa >= da:
            na, ba = aa, 1
        elif ha >= da:
            na, ba = ha, 0
        else:
            na, ba = da, 2
        hd, ad, dd = h - stop, a - stop, d
        if dd >= hd and dd >= ad:
            nd, bd = dd, 2
        elif hd >= ad:
            nd, bd = hd, 0
        else:
            nd, bd = ad, 1
        j = 3 * t
        back[j], back[j + 1], back[j + 2] = bh, ba, bd
        h, a, d = nh + sh[t], na + sa[t], nd + sd[t]
    path = np.zeros(T, dtype=np.int8)
    state = 0 if (h >= a and h >= d) else (1 if a >= d else 2)
    path[-1] = state
    for t in range(T - 1, 0, -1):
        state = back[3 * t + state]
        path[t - 1] = state
    return path


def decode(p_dead, p_home, halves, n_frames, params=None, extra=None):
    """Labels for every frame: H / A / D inside the halves, X outside.

    `extra` is an optional [T, 3] array of additional log-scores (e.g. Smart Tagging priors).
    """
    prm = dict(DEFAULT, **(params or {}))
    scores = emissions(p_dead, p_home, prm["a_dead"], prm["a_team"])
    if extra is not None:
        scores = scores + extra
    out = np.full(n_frames, "X", dtype="<U1")
    for a, b in halves:
        path = viterbi(scores[a:b], prm["switch"], prm["stop"], prm["restart"])
        out[a:b] = LABELS[path]
    return out

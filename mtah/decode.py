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

Minimum dead-ball duration (`min_dead` frames): a stoppage between two stretches of play must
last at least that long (one at the very start or end of a half is exempt). Real stoppages do:
in the sample games 1 in 850 is shorter than 1.1 s. Without it, whenever switch > stop + restart
the decoder hands possession over through a one-frame fake stoppage (H -> D -> A) instead of a
switch, so the switch cost silently stops meaning anything.
"""

import numpy as np

LABELS = np.array(["H", "A", "D"])

DEFAULT = dict(a_dead=1.0, a_team=1.0, switch=16.0, stop=16.0, restart=16.0, min_dead=1)

# A hand-over must be cheaper than stopping and restarting play by at least this much. At
# equality the transition costs no longer separate the two, and any hand-over where the dead
# model is mildly unsure (P(dead) above ~0.14 at the shipped weights) becomes a fake stoppage.
SWITCH_MARGIN = 16.0


def emissions(p_dead, p_home, a_dead, a_team, eps=1e-4):
    pd_ = np.clip(p_dead, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    alive = a_dead * np.log(1 - pd_)
    return np.column_stack([alive + a_team * np.log(ph), alive + a_team * np.log(1 - ph),
                            a_dead * np.log(pd_)])


def viterbi(scores, switch, stop, restart, min_dead=1):
    """Best state path for [T, 3] emission scores under the transition costs.

    A plain-float loop: with three states, numpy's per-call overhead costs more than the maths.
    """
    if min_dead > 1:
        if not np.all(np.isfinite(scores)):
            # C[t] - C[s] would be -inf - -inf = NaN and silently poison the path
            raise ValueError("emission scores must be finite for the min_dead decoder")
        return _viterbi_min_dead(scores, switch, stop, restart, int(min_dead))
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


def _viterbi_min_dead(scores, switch, stop, restart, k):
    """Exact Viterbi with D runs of at least k frames, in O(1) per frame.

    Besides H and A it tracks G, the best path that has been dead for at least k frames (or since
    the start of the half), and E[s], the best path entering D at frame s. A path that entered at
    s becomes eligible to restart k - 1 frames later, scored E[s] plus the dead emissions since
    (a running cumulative sum), so the k intermediate D states never need to exist.
    """
    T = len(scores)
    neg = float("-inf")
    sh, sa, sd = (scores[:, j].tolist() for j in range(3))
    C = np.cumsum(scores[:, 2]).tolist()          # cumulative dead emission
    E = [neg] * T                                 # entering D at s
    back_e = bytearray(T)                         # state before the entry: 0 H, 1 A
    back_h = bytearray(T)                         # 0 H, 1 A, 2 G
    back_a = bytearray(T)
    back_g = bytearray(T)                         # 0 stayed in G, 1 graduated from E[t-k+1]
    h, a, g = sh[0], sa[0], sd[0]                 # dead from the start is exempt
    for t in range(1, T):
        ch, ca, cg = h, a - switch, g - restart
        if ch >= ca and ch >= cg:
            nh, bh = ch, 0
        elif ca >= cg:
            nh, bh = ca, 1
        else:
            nh, bh = cg, 2
        ch, ca, cg = h - switch, a, g - restart
        if ca >= ch and ca >= cg:
            na, ba = ca, 1
        elif ch >= cg:
            na, ba = ch, 0
        else:
            na, ba = cg, 2
        if h >= a:
            E[t], back_e[t] = h - stop + sd[t], 0
        else:
            E[t], back_e[t] = a - stop + sd[t], 1
        stay = g + sd[t]
        s = t - k + 1
        grad = E[s] + C[t] - C[s] if s >= 1 else neg
        if stay >= grad:
            ng, back_g[t] = stay, 0
        else:
            ng, back_g[t] = grad, 1
        back_h[t], back_a[t] = bh, ba
        h, a, g = nh + sh[t], na + sa[t], ng
    # termination: H, A, G, or a stoppage still running at the end of the half (exempt)
    best, state, start = h, 0, -1
    if a > best:
        best, state = a, 1
    if g > best:
        best, state = g, 2
    for s in range(max(1, T - k + 1), T):
        v = E[s] + C[T - 1] - C[s]
        if v > best:
            best, state, start = v, 3, s
    path = np.zeros(T, dtype=np.int8)
    t = T - 1
    if state == 3:                                # open stoppage from `start` to the end
        path[start:] = 2
        t, state = start - 1, back_e[start]
    while t >= 0:
        if state == 0:
            path[t] = 0
            nxt = back_h[t]
        elif state == 1:
            path[t] = 1
            nxt = back_a[t]
        else:                                     # G
            if t == 0 or back_g[t] == 0:
                path[t] = 2
                nxt = 2
            else:
                s = t - k + 1
                path[s:t + 1] = 2
                t, state = s - 1, back_e[s]
                continue
        t -= 1
        state = nxt
    return path


def decode(p_dead, p_home, halves, n_frames, params=None, extra=None):
    """Labels for every frame: H / A / D inside the halves, X outside.

    `extra` is an optional [T, 3] array of additional log-scores (e.g. Smart Tagging priors).
    """
    prm = dict(DEFAULT, **(params or {}))
    if prm["switch"] > prm["stop"] + prm["restart"] - SWITCH_MARGIN:
        # Handing possession over must cost clearly less than stopping play and restarting it,
        # or the decoder answers uncertain hand-overs with fake stoppages: D frames belong to no
        # possession, which flatters spellF1 on paper while labelling live play dead.
        raise ValueError("switch cost {} must be at most stop + restart - {} ({} + {} - {}): "
                         "otherwise the decoder hands possession over through fake stoppages"
                         .format(prm["switch"], SWITCH_MARGIN, prm["stop"], prm["restart"],
                                 SWITCH_MARGIN))
    scores = emissions(p_dead, p_home, prm["a_dead"], prm["a_team"])
    if extra is not None:
        scores = scores + extra
    out = np.full(n_frames, "X", dtype="<U1")
    for a, b in halves:
        path = viterbi(scores[a:b], prm["switch"], prm["stop"], prm["restart"],
                       prm.get("min_dead", 1))
        out[a:b] = LABELS[path]
    return out

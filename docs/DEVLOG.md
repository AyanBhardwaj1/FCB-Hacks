# Development log: what was tried, measured, kept and dropped

The technical document (docs/TECHNICAL.md) summarises the shipped pipeline. This log keeps the
full record behind it: every round's diagnosis, the paired comparisons, and the negative results.
All numbers are on Metrica's public sample games, with leave-one-game-out validation and the
challenge's own scorer.

## Validation protocol and every round, newest first

`scripts/evaluate.py` runs leave-one-game-out: train on two sample games, then score the third
under three unseen degradation seeds and two scenarios with the challenge's own
`possession_metrics.py`. That makes 9 test units per scenario. `default` is calibrated to the
challenge ATD. `harsh` is a robustness check that is worse than we expect the real data to be:
55% ball missing (vs 36–46%), 15% player dropout (vs 7%), ball noise 0.5 m (vs 0.25),
player noise 0.25 m (vs 0.10), 3× false ball detections, track fragments every 15 s (vs 40 s).
Seeds are kept apart by role: 100 tunes decoder costs (`scripts/crossval.py`), 101–103 are for
choosing between designs, and each round's final numbers come from a seed range no decision has
looked at (201–203, then 301–303). Comparisons are **paired** on identical test units, and every
arm of a comparison is evaluated with the current degradation code.

**Round 6, the ball in tracker gaps (design seeds 101–103).** Measured first
(`scripts/gap_analysis.py`, live frames inside gaps, against the true ball): straight-line
interpolation between sightings places the ball within 0.3–0.9 m, with the right nearest team
88–95% of the time, for gaps up to 2 s. It drops to 2.8 m / 75% for 2–4 s gaps and is no better
than chance beyond 8 s. Snapping to the nearest player adds nothing.

- Extending interpolation from 0.4 s to 2 s gaps (with flags for interpolated frames and position
  in the gap) changed spellF1 by +0.001 ± 0.002 (default) and +0.004 ± 0.004 (harsh). Not shipped.
  The reason is that interpolation adds no information: it is a function of the last and next
  sightings, which the model already reads through its time-since/until features.
- **Ceiling: what real ball detections are worth.** The same pipeline trained and tested with
  the ball almost never lost (3%) scores 0.807 vs 0.748 (+0.059 ± 0.007, 9/9). The *shipped*
  model, unchanged, gains **+0.036 ± 0.009** (9/9) when 20% instead of 36–46% of the ball is
  missing at test time, and +0.063 ± 0.007 at 3%. No retraining is needed to profit from extra
  sightings. These simulated sightings have the tracker's noise but no false detections.
- So the lever is an *independent* signal: the video. `mtah/video_ball.py` + `run.py --video-ball`
  (optional, `requirements-video.txt`) search the tactical video for the ball, only inside
  tracker gaps and only near where it should be:
  - search centre and radius from the sightings around the gap, projected through the frame's
    homography;
  - a small, bright, low-saturation, round blob of the size a 22 cm ball has at that spot, mapped
    back to pitch metres;
  - the existing speed-spike filter downstream.

  It is classical OpenCV with no learned weights, so it is deterministic and inspectable.
  Candidates on the penalty or centre spots are ignored, and a detection only counts if a
  neighbouring frame confirms it. It **calibrates itself on the real video without any ground
  truth**. Stretches the tracker does see are hidden, with lengths drawn from the real gaps, and
  searched like gaps twice:
  - around the expected position, measuring how often and how accurately the ball is found when
    it is there;
  - around a point the ball is *not* near, measuring false positives (in real gaps the ball is
    often occluded, in the air or out of play).

  It **fails closed**: detections are merged only if the whole video was read and both probe
  kinds pass. That means at least 100 frames each, at least 15% found, at least 80% of finds
  within 1 m, and at most 5% false detections. Results are cached under `cache/` (derived from
  the licensed video, never committed). Tests run it on synthetic video (a perspective homography,
  coloured players, a white-shirted decoy, painted spots, occlusions, MJPG compression, a
  truncated file): at least 90% of visible-ball frames found within 0.5 m, no detections while
  the ball is hidden, only gap frames filled. Independent review found that the first version's
  calibration could not see false positives (a penalty spot would have passed as the ball), and
  that is what the ball-absent probes, marking rejection and confirmation rule fix.
- **Not validated on real footage**: there is no video in the sample data, and the match video
  cannot be used in a cloud session. Its effect on the real match is unknown until it runs
  locally, and the calibration report it prints is the evidence to judge it by.

**Round 5, short possessions: no change shipped (design seeds 101–103).** The shipped pipeline
recovers 52% of possessions under 8 s (IoU ≥ 0.5): 27% under 2 s, 41% at 2–4 s, 70% at 4–8 s
(`scripts/short_analysis.py`). The anatomy points at the tracking, not the model or decoder:
46% of short possessions contain no touch the detector can find for the possessing team, and
those are recovered 33% of the time, against 60–84% when it finds one or more. Ball visibility
under 30% inside the possession: 27%.

Two attempts, both measured paired against the shipped pipeline:
1. **Sequence features**: completed passes (consecutive same-team touches ≥ 5 m apart, 97% of
   them inside the passing team's own possession) and steals from the touch sequence, plus the
   nearest team's margin at the last and next ball sighting around a tracker gap. Result: spellF1
   +0.001 ± 0.003 (default), −0.000 ± 0.004 (harsh). The existing touch features already carry
   this, so the features were removed again.
2. **Hand-over detector**: a third classifier, P(direct possession change within ± 0.5 s),
   team-symmetric (P near true hand-overs 0.19–0.25 vs 0.03 elsewhere). The decoder's switch cost
   drops where it fires (`decode.switch_costs`, per-frame switch costs in an exact Viterbi).
   An oracle (cheap switches only near *true* hand-overs) shows +0.034 is available. The real
   detector, tuned on seed 100 (+0.010 there), raised recovery of possessions under 8 s from 52%
   to 58% and switches found from 62% to 68%, but invented 83 more switches. On the design seeds
   spellF1 was +0.002 ± 0.004 (default) and −0.006 ± 0.003 (harsh, worse on 8/9). Gentler settings
   were +0.003 ± 0.003 at best. It stays in the code as an opt-in (`config.HAND_OVER`,
   `--hand-over`), off by default. The shipped model and submission are unchanged.

**Round 4, possession hand-overs, on fresh seeds 401–403.** "Before" = the round-3 pipeline
(decoder `a_dead 0.75, a_team 2, switch 48, stop 32, restart 32`). "After" (shipped) = plus
possession-equal training weights, the 1 s minimum stoppage, and the margin rule, with decoder
`switch 64, stop 40, restart 40`, chosen on seed 100 under those rules:

| seeds 401–403 | default spellF1 | harsh spellF1 | default macroF1 | harsh macroF1 |
|---|---|---|---|---|
| before | 0.741 ± 0.017 | 0.681 ± 0.020 | 0.905 | 0.884 |
| **after (shipped)** | **0.758 ± 0.014** | **0.699 ± 0.015** | **0.904** | **0.884** |

Paired after − before: **+0.016 ± 0.003** spellF1 on default (better on 9/9 units) and
**+0.019 ± 0.005** on harsh (8/9). macroF1 is unchanged (−0.001 ± 0.001).

| default scenario, 2 s window | before | after |
|---|---|---|
| possession switches found / invented | 60% / 200 | **63%** / 241 |
| switches into a possession < 3 s / 3–8 s found | 32% / 57% | **38% / 62%** |
| fake stoppages at hand-overs (predicted D with no true D) | 61 | 65 |
| stops found / invented | 70% / 286 | 71% / 278 |

How the round went:
- Diagnosis: possessions over 8 s were already 96% right. On possessions under 8 s that were
  mostly mislabelled, the frame model gave the new team only 0.26–0.30, so the misses were the
  model's, not the decoder's. The loss/metric mismatch above explained it.
- The first decoder tune for the weighted model picked switch 96 on seed 100. It turned out to
  win by routing hand-overs through fake stoppages: 451 invented stops instead of about 280, and
  fewer direct switches found (50%). Hence the minimum stoppage and the margin rule. Independent
  review then showed the first version of the rule (equality allowed) still let a mildly raised
  P(dead) turn a hand-over into a stoppage (86 fake stoppages on the design seeds), so the
  margin of 16 was added.
- The decoder choice was made on seed 100 and the design seeds 101–103. The fresh-seed run was
  first scored with an intermediate decoder (`switch 64, stop/restart 32`: +0.015 / +0.020) and
  then re-decoded with the final one, with no retraining. The table shows the final one.

**Round 3, dead-ball boundaries, on fresh seeds 301–303.** "Before" = touch and turnover
features, fixed training noise, decoder `a_team 2, switch 48, stop 32, restart 32`. "After" (shipped)
= plus dead-ball boundary features, randomized training noise, decoder `a_dead 0.75` with the
same costs. Both use four training seeds per game:

| seeds 301–303 | default spellF1 | harsh spellF1 | default macroF1 | harsh macroF1 |
|---|---|---|---|---|
| baseline idea (nearest player, ±50-frame vote) | ≈ 0.20 | — | ≈ 0.46 | — |
| before | 0.726 ± 0.016 | 0.625 ± 0.029 | 0.900 | 0.844 |
| **after (shipped)** | **0.731 ± 0.012** | **0.687 ± 0.023** | **0.904** | **0.887** |

± is the spread (sd) over the 9 units. Paired differences after − before (mean ± standard error):
+0.005 ± 0.006 spellF1 on default (no measurable change, better on 4/9); **+0.061 ± 0.011** on
harsh (9/9). macroF1 +0.004 ± 0.002 on default (7/9), +0.043 ± 0.007 on harsh (9/9).

Dead-ball boundaries specifically (`scripts/error_analysis.py`, 2 s matching window):

| | default before → after | harsh before → after |
|---|---|---|
| stops found / invented | 66% / 319 → **70% / 277** | 51% / 435 → **64% / 330** |
| restarts found / invented | 75% / 244 → **76% / 227** | 49% / 452 → **70% / 281** |
| restart timing, \|offset\| p75 | 16 → **11** frames | 26 → **17** frames |
| possession switches found / invented | 59% / 232 → 60% / 228 | |
| spellF1, perfect team labels (oracle) | 0.925 → 0.930 | |
| spellF1, perfect dead-ball labels (oracle) | 0.793 → 0.794 | |

Reading it: dead-ball detection got measurably sharper, but on realistic noise each fixed
boundary moves possession overlap only slightly, so spellF1 barely moves. Most of the gain is
robustness. Randomizing the training noise alone was worth +0.057 ± 0.006 spellF1 on harsh with
no cost on default (design seeds 101–103), and the boundary features added about +0.014 on top.
On the design seeds the boundary features also showed +0.007 ± 0.003 on default. That gain
largely disappeared once review found they could key on the timing of a single synthetic
dead-ball template, which is now randomized (§2). Team hand-overs remain the big gap: 40% of
true possession switches are still missed.

History, for honesty:
- Round 2 (touch and turnover features) on fresh seeds 201–203: spellF1 0.720 vs 0.695 before
  them, +0.025 ± 0.005 (9/9), with no gain on harsh. That was the shipped pipeline before this
  round.
- Numbers in earlier versions of this document (0.688, then 0.734) came from degradations with
  label leaks: dropout clipped to alive/dead regions, then extra jitter on synthetic dead-ball
  paths. Both were found in independent review and are not comparable.

Other findings:
- The baseline's ≈0.20 here is close to its published 0.2433 on the real match. This is the
  main evidence that the degradation is realistic.
- Rejected after measurement: a stage-2 model over windowed stage-1 probabilities (no gain), and
  Smart Tagging priors (±0.01 depending on simulated tag noise).
- The public 5-minute window was **not** used to choose anything.


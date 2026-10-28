# Technical document: frame-level ball possession from automatic tracking

More than a Hack 2027, Phase 1. Draft: the final version should add the team name and the
validated full-match file hash.

## 1. Approach

The problem has two parts: **is the ball alive**, and **if so, which team controls it**. The
ranking metric (spellF1) scores whole possessions, so the output also has to be made of clean
intervals.

```
ATD (tracking) ──► per-frame features ──► P(dead), P(home | alive) ──► 3-state Viterbi ──► H/A/D
                     (mtah/features.py)     gradient-boosted trees       switch / stop /
                                            trained on OTHER matches     restart costs
```

1. **Features** (120 per frame: 71 seen from each team's side plus 49 shared, i.e. 191 model
   columns for the dead-ball model; team-level only, since the ATD has no player identities):
   - ball: seen / interpolated, length of the current gap, speed, distance outside the pitch
     lines, distance to touchline, corner and centre spot
   - proximity: each team's nearest player to the ball, players within 2 m and 5 m, and
     *control episodes* (a player within 1.5 m of a slow ball, with the nearest opponent at least
     0.5 m further away)
   - **touches**: a sharp change of ball velocity (≥ 5 m/s between the 3 frames before and the 3
     after, at its local peak) with one team's player within 2.5 m and clearly closest. These
     catch one-touch passes, first-time clearances and interceptions, which the slow-ball
     control rule misses. From them: last and next toucher, time since / until each team's last
     / next touch, and each team's touches in the previous and next 2 s (pass chains)
   - time since / until each team last / next controlled the ball: the strongest cue when the
     tracker loses the ball
   - team shape, each team in its own attacking frame: centroid, spread, most advanced and
     second-deepest player, forward velocity of the block, players in the opponent's half
   - **turnover signatures**: each block's forward velocity and spread over the next 2 s minus
     the previous 2 s (a team that has just lost the ball turns and contracts), and the ball's
     velocity towards each team's attacking goal before and after the frame
   - activity: mean player speed (1/3/6 s) relative to the match median, share of fast players,
     block motion; kickoff shape (both teams entirely in their own halves)
   - **dead-ball boundaries**, one-sided in time so the decoder can place a stop or restart to
     the frame instead of at the blurred edge of a centred window: time since / until the ball
     was last outside the lines; time since / until it was last still, with "still" judged from
     the displacement between half-second windows (a frame-to-frame speed reads 2–4 m/s for a
     resting ball on noisy tracking); *projected exit*, i.e. how long ago the ball's last-seen
     straight path crossed a line, which locates the stop even when the tracker loses the ball
     on its way out; ball speed over the next vs previous 1–2 s (a restart kicks a still ball);
     collective player speed over the next 1–2 s minus the previous (players ease off at the
     whistle)
   - centred 0.5/1/3 s rolling means of the main ones (offline, so non-causal windows are fine)
2. **Two classifiers** (scikit-learn `HistGradientBoostingClassifier`, NaN-native): P(dead)
   and P(home | alive). **Exact team symmetry**: each training frame is used from both teams'
   points of view (label flipped), and at prediction P(H) = ½·(f(home view) + 1 − f(away view)).
   Swapping the two teams' tracking swaps the answer exactly (unit-tested), so nothing about "the
   home side" carries over from training. Models store the feature names they were trained on and
   refuse any other feature set. **Possession-equal weights**: spellF1 counts every possession
   once, but a frame-level loss lets a 30 s possession outweigh a 3 s one ten to one, and short
   possessions are half of all possessions. So each training frame of the team model is weighted
   by 8 s ÷ its possession's length, clipped to [0.25, 4], and every possession counts about the
   same, as it does in the score. The dead-ball model stays unweighted.
3. **Decoder**: Viterbi over states H/A/D per half, with emissions from tempered
   log-probabilities and fixed costs for every H↔A switch, every stop (→D) and every restart
   (D→). This is the challenge's *controlled possession* rule written as a price. A deflection
   or a won header (a few frames of evidence for the other team) doesn't pay for a switch.
   Sustained control does. Outside the two graded halves we write `X`. Two structural rules
   keep the dead-ball state honest:
   - A stoppage lasts at least 1 s (`min_dead`; 1 in 850 real ones is shorter). This is an
     exact Viterbi, O(1) per frame, checked against brute force.
   - A hand-over must cost at least 16 less than stopping and restarting play
     (`switch ≤ stop + restart − 16`; `decode()` refuses other costs).

   Without these the decoder can hand possession over through a fake stoppage. D frames belong to
   no possession, so that flatters spellF1 on paper while labelling live play dead. One decoder
   sweep did exactly that: with switch 96 > 32 + 32 it made zero direct switches.

Why this rather than smoothing: the published baseline (nearest player + majority vote) has no
notion of a dead ball beyond "ball not seen" and no notion of an interval. Our validation shows
those are the two things that matter.

## 2. Training data: other matches only

RULES.md allows labelled data from other games in any amount, provided the organisers can
access it. We use **Metrica Sports' public sample data**
(<https://github.com/metrica-sports/sample-data>, pinned at commit `e706dd5`): three anonymised
matches with tracking and event data, ≈430k frames. Labels are derived **mechanically from the
published events**. No person labelled anything for this project:

- **possession**: the team of the latest controlling event (pass, carry, set piece, shot,
  recovery; the losing team keeps it after "ball lost" until the opponent's next controlling
  event). Challenges don't move possession.
- **controlled possession** (CHALLENGE.md): a spell stands only if the team completed a pass,
  shot or set piece in it, carried the ball ≥ 1 s, or kept it ≥ 2 s. Otherwise it is handed back
  to the previous team. This turns touches and deflections into "no change", as the challenge
  asks.
- **dead ball**: from ball out / foul / offside / shot out / goal until the next restart; each
  half is dead until its kick-off.

**Making clean sample tracking look like the challenge ATD** (`mtah/sample_games.degrade`):
- **ball dropout is one burst process over each half**, more likely when the ball is fast (in
  the air), crowded, off the pitch or far from every player, with the share missing over the
  halves drawn per seed in 36–46% (the challenge: 41%). Dropout **never depends on the labels**.
  An earlier version drew separate live / dead dropout masks clipped to the label regions. That
  switched ball visibility on or off exactly at about half of all stops and restarts, which the
  models could read as a label leak (found in independent review, see §3). Now visibility flips
  at label boundaries at the base rate (1.6–3.1% vs 1.9%), and a unit test guards it.
- the sample data stops tracking the ball whenever it is dead. Left as-is, a model would learn
  "no ball ⇒ dead ball", which is false for the challenge ATD. Each dead stretch therefore gets a
  plausible synthetic ball path (drifts on over the line, is carried to the restart spot, sits
  still there before the restart), and the few short gaps in live play are interpolated. Then all
  of it goes through the same label-free dropout. Each stretch gets its own shape: how fast the
  ball slows (time constant 0.2–1 s), how long it drifts (0.5–2.5 s), and whether it rests
  before the restart at all (no in 30% of stretches, else 1–4 s). With one fixed template, the
  one-sided "still ball" features could learn the template's timing instead of football
- players drop out in bursts, and tracks are cut into fragments packed into slots. Positional
  noise is added to players and ball, plus false ball detections. A speed channel is
  synthesised, since the challenge ATD has one
- **the noise level is randomized per training seed** (`sample_games.PARAM_RANGES`): ball noise
  0.15–0.50 m, player noise 0.05–0.25 m, player dropout 4–12%, false ball detections
  0.1–0.5% of frames, track fragments every 15–60 s. The real ATD's noise level is unknown, and a
  model trained at one fixed level learns thresholds (a still ball, a touch) that break when the
  data is noisier. The evaluation scenarios pin their values, so results stay comparable

Four degradation seeds per game form the training set.

## 3. Validation

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

## 4. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD | everything above |
| published half boundaries (`games.half_video_blocks`) | where to decode; `X` elsewhere |
| Smart Tagging | implemented as soft decoder priors (`--tags`), **off by default**: no measurable gain in simulation, and it hurt as tag noise grew |
| homography, both videos, broadcast ATD | not used |

## 5. Possession rules implemented

- A change of possession needs sustained evidence of control by the other team (decoder switch
  cost 64 at team weight 2, about 0.5–1 s of clear evidence).
- Stops and restarts cost 40 each, with the dead-ball evidence weighted 0.75. A stoppage lasts
  at least 1 s, and a hand-over is never routed through a stoppage to save cost (see §1).
- Loose and contested balls stay with the last team in control (no evidence ⇒ no switch).
- Dead ball: evidence from the ball outside the lines, a still and unattended ball, collective
  slowdown, long ball gaps and the kickoff shape. Restarts begin a new possession.

## 6. Safety checks in the pipeline

- The EPTS reader follows the metadata's `DataFormatSpecification` and **raises** on any line
  whose field count disagrees with it, on records outside every spec block or outside
  1..199,319, and on unknown players. It never silently shifts or drops data.
- `run.py` refuses to write a submission if graded-frame coverage is implausibly low (ball
  < 20%, < 10 players; the real ATD has 59% and 20.5), then runs the challenge's validator on
  the file it wrote.

## 7. Limitations

- Training labels come from event data on three matches of another competition. The challenge's
  ground truth is derived from Metrica's corrected tracking, so the semantics differ at the margins.
- Ball dropout and dead-ball paths in training are simulated. The real ATD's error pattern
  (identity switches, duplicates, systematic loss in aerial play) is only approximated.
- Short possessions are still the weak spot (52% of those under 8 s recovered), and round 5
  suggests the sample-game tracking often holds no evidence of them: nearly half contain no
  detectable touch by the possessing team. The video detector (round 6) targets exactly that,
  but it is unvalidated on real footage.
- The video detector looks for a white ball. A differently coloured ball, heavy motion blur, or
  a camera view where the ball spans only one or two pixels will lower its recall. The
  calibration report shows that, and the fail-closed gate then keeps it out.
- The dead-ball paths are simulated (randomized, but simulated). How well the boundary
  features transfer depends on how the real ATD sees an out-of-play ball, which the sample data
  cannot show.
- Next steps: possession hand-overs (still the largest gap: perfect team labels would give
  ≈ 0.93), a hand-over detector precise enough to beat its false switches (the oracle says
  +0.03 is there), a touch detector
  whose threshold adapts to the measured ball noise, pseudo-labelling the challenge match itself
  (allowed, as long as no person decides labels), ball detection in tracker gaps from video, and
  a duration-aware (semi-Markov) decoder.

## 8. Reproduction

```bash
git clone --recursive <repo> && cd FCB-Hacks
pip install -r requirements.txt                       # pinned
export MTAH_DATA_DIR=/path/to/challenge/bundle        # or unpack into ./data
python run.py --team <team_name>                      # -> submissions/<team_name>.csv
python train.py                                       # optional: rebuild models/ from public data
python scripts/evaluate.py --tag mine                 # optional: the validation in section 3
python scripts/crossval.py                            # optional: the decoder-cost sweep
```

Deterministic: fixed seeds, no early-stopping randomness, pinned versions and pinned sample data.
Two runs give byte-identical files (checked by SHA-256).

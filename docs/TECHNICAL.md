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
   refuse any other feature set.
3. **Decoder**: Viterbi over states H/A/D per half, with emissions from tempered
   log-probabilities and fixed costs for every H↔A switch, every stop (→D) and every restart
   (D→). This is the challenge's *controlled possession* rule written as a price. A deflection
   or a won header (a few frames of evidence for the other team) doesn't pay for a switch.
   Sustained control does. Outside the two graded halves we write `X`.

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

**Current pipeline vs the previous one, on fresh seeds 301–303.** "Before" = touch and turnover
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
  cost 48 at team weight 2, about 0.5–1 s of clear evidence).
- Stops and restarts cost 32 each, with the dead-ball evidence weighted 0.75.
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
- Short possessions (< 4 s) are mostly missed, partly by design: the switch cost suppresses them.
- The dead-ball paths are simulated (randomized, but simulated). How well the boundary
  features transfer depends on how the real ATD sees an out-of-play ball, which the sample data
  cannot show.
- Next steps: possession hand-overs (still the largest gap to the oracle), a touch detector
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

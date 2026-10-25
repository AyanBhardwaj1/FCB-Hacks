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

1. **Features** (108 per frame: 71 seen from each team's side plus 37 shared, i.e. 179 model
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
  of it goes through the same label-free dropout
- players drop out in bursts (≈20 of 22 tracked on average, vs 20.5 in the challenge), and tracks
  are cut into fragments packed into slots. Positional noise is added to players and ball, plus
  0.2% false ball detections. A speed channel is synthesised, since the challenge ATD has one

Four degradation seeds per game form the training set.

## 3. Validation

`scripts/evaluate.py` runs leave-one-game-out: train on two sample games, then score the third
under three unseen degradation seeds and two scenarios with the challenge's own
`possession_metrics.py`. That makes 9 test units per scenario. `default` is calibrated to the
challenge ATD. `harsh` is a robustness check that is worse than we expect the real data to be:
55% ball missing (vs 36–46%), 15% player dropout (vs 7%), ball noise 0.5 m (vs 0.25),
player noise 0.25 m (vs 0.10), 3× false ball detections, track fragments every 15 s (vs 40 s).
Seeds are kept apart by role: 100 tunes decoder costs (`scripts/crossval.py`, run for each
feature set), 101–103 were used to choose between designs, and 201–203 report the final numbers.
Comparisons are **paired** on identical test units.

Final numbers, on **fresh degradation seeds 201–203**, which no decision ever looked at. Both
arms use the same decoder (`a_team 2, switch 48, stop 32, restart 32`; it is within 0.002 of each
arm's own best on seed 100) and four training seeds per game:

| leak-free degradation, seeds 201–203 | default spellF1 | harsh spellF1 | default macroF1 |
|---|---|---|---|
| baseline idea (nearest player, ±50-frame vote) | ≈ 0.20 | — | ≈ 0.46 |
| v1 features (no touches / turnover signatures) | 0.695 ± 0.019 | 0.612 ± 0.033 | 0.890 |
| **v2 features (shipped)** | **0.720 ± 0.015** | **0.615 ± 0.032** | **0.899** |

± is the spread (sd) over the 9 units. Paired differences v2 − v1 (mean ± standard error):
**+0.025 ± 0.005** spellF1 on default, better on 9 of 9 units; +0.004 ± 0.005 on harsh, which is
no measurable gain. The touch detector thresholds a change in ball velocity, so doubled ball
noise drowns it. Making the threshold adapt to the measured noise is an obvious next step.

How the choices were made, honestly: the v2 feature set and the four training seeds were picked
by comparing arms on seeds 101–103 (v2 − v1 there: +0.040 ± 0.004; 4 − 2 seeds: +0.002 on
default, +0.012 on harsh). Those runs still had the dead-ball jitter leak described in §2, which
flattered the touch features. The fresh-seed table above is the number to quote.

Error analysis (`scripts/error_analysis.py`, default scenario, 2 s matching window):

| | v1 | v2 |
|---|---|---|
| possession switches found / invented | 53% / 244 | **57% / 231** |
| median timing of found switches | +5 frames | **+2 frames** |
| stops found / invented | 66% / 389 | 68% / 342 |
| restarts found / invented | 77% / 298 | 76% / 275 |
| spellF1 with perfect team labels (oracle) | 0.920 | 0.923 |
| spellF1 with perfect dead-ball labels (oracle) | 0.768 | 0.785 |

Team hand-overs remain where most of the score is: 43% of true switches are still missed.

Other findings:
- The baseline's ≈0.20 here is close to its published 0.2433 on the real match. This is the
  main evidence that the degradation is realistic.
- Rejected after measurement: a stage-2 model over windowed stage-1 probabilities (no gain), and
  Smart Tagging priors (±0.01 depending on simulated tag noise).
- Numbers in earlier versions of this document (0.688, then 0.734) came from degradations with
  label leaks: dropout clipped to alive/dead regions, then extra jitter on synthetic dead-ball
  paths. Both were found in independent review. They are not comparable to the table above.
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
- Next steps: dead-ball detection (the oracle says ≈ +0.08 is available there), pseudo-labelling
  the challenge match itself (allowed, as long as no person decides labels), ball detection in
  tracker gaps from video, and a duration-aware (semi-Markov) decoder.

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

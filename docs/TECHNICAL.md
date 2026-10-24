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

1. **Features** (≈100 per frame, team-level only, since the ATD has no player identities):
   - ball: seen / interpolated, length of the current gap, speed, distance outside the pitch
     lines, distance to touchline, corner and centre spot
   - proximity: each team's nearest player to the ball, players within 2 m and 5 m, and
     *control episodes* (a player within 1.5 m of a slow ball, with the nearest opponent at least
     0.5 m further away)
   - time since / until each team last / next controlled the ball: the strongest cue when the
     ball is lost by the tracker
   - team shape, each team in its own attacking frame: centroid, spread, most advanced and
     second-deepest player, forward velocity of the block, players in the opponent's half
   - activity: mean player speed (1/3/6 s) relative to the match median, share of fast players,
     block motion; kickoff shape (both teams entirely in their own halves)
   - centred 0.5/1/3 s rolling means of the main ones (offline, so non-causal windows are fine)
2. **Two classifiers** (scikit-learn `HistGradientBoostingClassifier`, NaN-native): P(dead)
   and P(home | alive). **Exact team symmetry**: each training frame is used from both teams'
   points of view (label flipped), and at prediction P(H) = ½·(f(home view) + 1 − f(away view)).
   Swapping the two teams' tracking swaps the answer exactly (unit-tested), so nothing about "the
   home side" carries over from training.
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
(<https://github.com/metrica-sports/sample-data>): three anonymised matches with tracking and
event data, ≈430k frames. Labels are derived **mechanically from the published events**. No person
labelled anything for this project:

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
- the ball goes missing in bursts (log-normal lengths), more often when it is fast or crowded.
  Live and dead frames get separate missing shares (live 28–42%, dead 40–80%, drawn per seed),
  which brackets the challenge's 41% of graded frames
- the sample data stops tracking the ball whenever it is dead. Left as-is, a model would learn
  "no ball ⇒ dead ball", which is false for the challenge ATD. Each dead stretch therefore gets a
  plausible synthetic ball path (drifts on over the line, is carried to the restart spot, sits
  still there before the restart), which is then subject to the same dropout
- players drop out in bursts (≈20 of 22 tracked on average, vs 20.5 in the challenge), and tracks
  are cut into fragments packed into slots. Positional noise is added to players and ball, plus
  0.2% false ball detections. A speed channel is synthesised, since the challenge ATD has one

Two degradation seeds per game form the training set.

## 3. Validation

Leave-one-game-out: train on two sample games, score the third (degraded with an unseen seed)
with the challenge's own `possession_metrics.py`.

| | spellF1 | macroF1 |
|---|---|---|
| baseline idea (nearest player, ±50-frame vote) on the same degraded data | 0.21 | 0.55 |
| frame models + decoder (`switch 16, stop 32, restart 32`) | **0.688** | **0.889** |
| + stage-2 context model (probabilities through 0.2–8 s windows) | 0.686 | 0.888 |
| + Smart Tagging priors, simulated tags | ±0.01 depending on tag noise | |

- The baseline's 0.21 on degraded sample games is close to its published 0.2433 on the real
  match. This is the main evidence that the degradation is realistic.
- **Error analysis (oracles)**: with perfect dead-ball labels spellF1 rises to ≈0.75. With perfect
  team labels it rises to ≈0.89. Team hand-overs are where most of the remaining score is.
- Decoder costs sit on a broad plateau (0.687–0.690 over a range of settings). We picked its middle
  rather than its maximum, since differences below ~0.04 spellF1 are noise per EVALUATION.md.
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
  cost ≈ 16 log-units, about 0.5–1 s of clear control).
- Loose and contested balls stay with the last team in control (no evidence ⇒ no switch).
- Dead ball: evidence from the ball outside the lines, a still and unattended ball, collective
  slowdown, long ball gaps and the kickoff shape. Restarts begin a new possession.

## 6. Limitations

- Training labels come from event data on three matches of another competition. The challenge's
  ground truth is derived from Metrica's corrected tracking, so the semantics differ at the margins.
- Ball dropout and dead-ball paths in training are simulated. The real ATD's error pattern
  (identity switches, duplicates, systematic loss in aerial play) is only approximated.
- Short possessions (< 4 s) are mostly missed, partly by design: the switch cost suppresses them.
- With more time: pseudo-labelling the challenge match itself (allowed, as long as no person
  decides labels), using the video for ball detection in tracker gaps, and a duration-aware
  (semi-Markov) decoder.

## 7. Reproduction

```bash
git clone --recursive <repo> && cd FCB-Hacks
pip install -r requirements.txt                       # pinned
export MTAH_DATA_DIR=/path/to/challenge/bundle        # or unpack into ./data
python run.py --team <team_name>                      # -> submissions/<team_name>.csv
python train.py                                       # optional: rebuild models/ from public data
python scripts/crossval.py                            # optional: the validation in section 3
```

Deterministic: fixed seeds, no early-stopping randomness, pinned versions. Two runs give
byte-identical files (checked by SHA-256).

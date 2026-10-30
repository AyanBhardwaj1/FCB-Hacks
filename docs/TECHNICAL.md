# Frame-level ball possession from automatic tracking

**More than a Hack 2027, Phase 1** · Team: **TEAM_NAME**

| | |
|---|---|
| possession file | `TEAM_NAME.csv`, SHA-256 `CSV_SHA256` |
| produced by | `RUN_COMMAND` at commit `GIT_COMMIT`, RUN_PLATFORM |
| repository | <REPO_URL> (public); regenerate with the command above, see §7 |

## Summary

Gradient-boosted frame classifiers read team-level tracking features and give, per frame,
P(ball dead) and P(home has it | alive). A three-state Viterbi decoder turns those into possession
intervals, paying a price for every change of state. That price is the challenge's
"controlled possession" rule. **No labels from the competition match are used anywhere.** The
models learn from Metrica's three public sample games, whose tracking is degraded to look like
the challenge's automatic ATD.

On those games (leave one game out, unseen degradation seeds, scored with the challenge's own
`possession_metrics.py`) the pipeline reaches **spellF1 0.768 ± 0.009** (macroF1 0.910). The
challenge's baseline method scores ≈ 0.20 on the same data, against 0.2433 on the real match.
That match is the main evidence that the degraded games are a fair stand-in. On tracking worse
than we expect the real data to be, spellF1 is 0.715 ± 0.024. These are not competition scores.

## 1. Approach

```
ATD ──► per-frame features ──► P(dead), P(home | alive) ──► 3-state Viterbi ──► H / A / D (X outside halves)
          mtah/features.py        gradient-boosted trees        mtah/decode.py
                                  trained on OTHER matches
```

**Features** (120 per frame, team-level only, since the ATD has no player identities):

- **Ball and proximity:** whether the ball is seen or interpolated (gaps up to 0.4 s), how long
  the current gap is, ball speed, distance outside the lines; each team's nearest player and
  players within 2 m / 5 m; *control episodes* (a player within 1.5 m of a slow ball, the
  nearest opponent at least 0.5 m further).
- **Touches:** a sharp change of ball velocity (≥ 5 m/s) with one team's player clearly
  closest. From them, the last and next toucher, the time since and until each team's last
  touch, and short pass chains. Plus the time since and until each team last controlled the ball.
  These are the strongest cues when the tracker loses the ball.
- **Team shape**, each team in its own attacking frame: centroid, spread, most advanced and
  second-deepest player, forward velocity of the block.
- **Turnover signatures:** each team's forward velocity and spread over the next 2 s minus the
  previous 2 s, and the ball's velocity towards each goal.
- **Dead-ball cues:** collective player speed against the match median, kickoff shape, and
  one-sided boundary features. Those are the time since and until the ball was last out or
  still (judged from half-second windows, robust to jitter), the projected time the
  last-seen ball path crossed a line, and ball and player speed over the next 1–2 s minus the
  previous.

**Classifiers** (scikit-learn `HistGradientBoostingClassifier`, which handles missing values
natively):

- **Two models.** P(dead) and P(home | alive). Both use **exact team symmetry**: every training
  frame is seen from both teams' points of view, and at prediction
  P(H) = ½·(f(home view) + 1 − f(away view)). Swapping the teams' tracking swaps the answer
  exactly, so nothing about "the home side" carries over from training.
- **Settings:** 600 boosting rounds, 31 leaves, learning rate 0.05, at least 400 frames per leaf.
  Capacity was chosen by paired leave-one-game-out comparison (round 8 in `docs/DEVLOG.md`).
  300 rounds underfit, while larger trees, fewer leaves and denser training frames did not help.
- **Possession-equal weights.** spellF1 counts every possession once, so the possession model
  weights each training frame by 8 s ÷ the length of its possession (clipped to [0.25, 4]).
  Without this, short possessions, half of all possessions, are drowned out.

**Decoder:** Viterbi over H, A, D for each half.

- **Emissions:** tempered log-probabilities.
- **Costs:** a fixed cost for every switch (64), stop (40) and restart (40). A deflection or a
  won header is a few frames of evidence and doesn't pay for a switch; sustained control does.
- **Two structural rules keep "dead" honest:**
  - a stoppage lasts at least 1 s (2 of the 281 stoppages in the sample games are shorter);
  - a hand-over always costs at least 16 less than a stop plus a restart.

  Without them the decoder can hand possession over through fake stoppages, which flatters
  spellF1 while labelling live play dead. This is exact (O(1) per frame) and checked against
  brute force.

**Missing and noisy tracking** is the core difficulty (the ball is absent on 41% of graded frames),
and it is handled at every stage rather than patched once:

- **No identities.** The ATD's 46 slots are tracklets, not players, so every feature is
  team-level: nearest player per team, counts within a radius, team shape. Nothing depends on
  following one player.
- **Ball gaps.** Gaps up to 10 frames (0.4 s) are interpolated. Longer ones stay missing, and the
  trees route missing values natively instead of trusting an invented position. Features say
  how long the current gap is and what happened at its edges: who touched the ball last and
  next, where the last-seen path would have left the pitch, whether play slowed down.
- **Evidence that survives gaps.** Touches, control episodes and team-shape turnovers are
  measured where the ball is visible and carried across the gap as "time since / until", so a
  possession change hidden in a gap is still bracketed by evidence on both sides.
- **Training on the same kind of damage.** The models only ever see tracking degraded like the
  challenge's (ball missing in bursts, dropouts, fragments, noise, false detections), with the
  noise level randomized per copy because the real level is unknown (§3).
- **The decoder smooths what is left.** One noisy frame cannot pay for a change of state.
- **Bad input stops the run.** `run.py` refuses files that are blank or wrongly formatted, and
  tracking far below the published coverage (ball on < 20% of graded frames or < 10 players).

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (tracking with ball) | everything above |
| published half boundaries (`games.half_video_blocks`) | where to decode; `X` elsewhere |
| tactical video + homography | **optional** (`run.py --video-ball`): ball detection inside tracker gaps, see §4. Off unless asked for, and then used only if its self-calibration on the video passes |
| Smart Tagging | implemented as soft decoder priors (`--tags`), **off by default**: no measurable gain, and it hurt as tag noise grew |
| broadcast video and ATD | not used (not synchronised to the tactical timeline) |
| public 5-minute ground truth | pipeline check only (printed by `run.py`); **never** used to choose or tune anything |

## 3. Training data

The rules allow labelled data from other games in any amount, provided the organisers can access
it. We use **Metrica Sports' public sample data**
(<https://github.com/metrica-sports/sample-data>, pinned at commit `e706dd5`). It is three
anonymised professional matches with tracking at 25 fps and event data:

| | frames | minutes | events | possessions | stoppages |
|---|---|---|---|---|---|
| Sample Game 1 | 145,006 | 97 | 1,745 | 267 | 97 |
| Sample Game 2 | 141,156 | 94 | 1,935 | 253 | 95 |
| Sample Game 3 | 143,761 | 96 | 3,620 | 295 | 96 |
| **total** | **429,923** | **287** | **7,300** | **815** | **288** |

- **Who labelled what.** The event data are Metrica Sports' own annotations of these three
  other matches (human-produced, as far as we know), published openly by Metrica. RULES.md allows
  labelled data from other games in any amount. Our team labelled nothing: frame labels are
  derived from those events **mechanically**, by code in `mtah/sample_games.py`:
  - **possession:** the team of the latest controlling event (pass, carry, set piece, shot,
    recovery). Challenges don't move possession.
  - **controlled possession:** a spell stands only with a completed pass, shot or set piece, a
    carry of at least 1 s, or 2 s on the ball. Otherwise it goes back to the previous team, as
    the challenge's definition asks.
  - **dead ball:** from ball out, foul, offside, shot out or goal until the next restart.
- **Making the clean sample tracking look like the challenge ATD** (`degrade()`):
  - the ball goes missing in bursts over each half, never depending on the labels (36–46% of
    frames; the challenge reports 41%);
  - synthetic ball paths fill the stretches where the sample data stopped tracking a dead ball,
    each with its own randomized shape;
  - players drop out and tracks fragment;
  - noise and false ball detections are added;
  - a speed channel is synthesised.

  The noise level is randomized for each training copy, since the real ATD's level is unknown.
  Each game is degraded with eight seeds: about 3.4 M frames, of which every 4th is used
  (neighbouring frames are near-duplicates), each seen from both teams' points of view.

## 4. Validation

`scripts/evaluate.py` trains on two sample games and scores the third, three unseen
degradation seeds each, with the challenge's scorer. That gives 9 test units per scenario. Seeds
are kept apart by role:
- 100 tunes decoder costs;
- 101–103 compare designs;
- a fresh range that no decision looked at reports each round's final result.

All comparisons are **paired** on identical test units. Two independent code reviews ran in every
round. Among other things they found two label leaks in an early degradation, both fixed before
any number here was produced.

| fresh seeds 501–503 | spellF1 | macroF1 | spellF1, harsh tracking |
|---|---|---|---|
| baseline idea (nearest player, ±2 s vote) | ≈ 0.20 | ≈ 0.46 | — |
| **this pipeline (shipped)** | **0.768 ± 0.009** | **0.910** | **0.715 ± 0.024** |

What moved the score (paired, on fresh seeds, details in `docs/DEVLOG.md`):

| change | spellF1 |
|---|---|
| frame models + decoder vs the baseline idea | +0.50 |
| touch and turnover features | +0.025 ± 0.005 |
| randomized training noise + dead-ball boundary features | +0.005 (realistic), **+0.061 ± 0.011 (harsh)** |
| possession-equal weights, 1 s stoppages, margin rule | **+0.016 ± 0.003** (9/9) |
| 600 boosting rounds and 8 noise copies per game (was 300 and 4) | **+0.014 ± 0.003** (9/9), harsh **+0.016 ± 0.003** (9/9) |

Measured and **not** shipped: a second-stage context model, Smart Tagging priors, pass and
steal sequence features, a learned hand-over detector (opt-in `--hand-over`; it raised short
possession recall but invented as many switches), longer ball interpolation, and self-training
on the target match with algorithmic pseudo-labels (+0.001 ± 0.002; the pseudo-labels are wrong
exactly where the model is). None improved spellF1 beyond noise.

**Error profile** (shipped pipeline, 2 s matching window):
- 65% of true possession switches are found;
- 73% of stops and 79% of restarts are found;
- with perfect team labels spellF1 would be ≈ 0.93, and with perfect dead-ball labels ≈ 0.83, so
  possession hand-overs are where the remaining score is.
- **Ceiling for ball detection:** the shipped model, unchanged, gains **+0.036 ± 0.009** (9/9)
  when 20% instead of 36–46% of the ball is missing. That is why the optional video detector
  exists.

**Reading the real files.** The FIFA/EPTS reader (`mtah/io.py`) follows the metadata's
`DataFormatSpecification`. On Metrica's own EPTS export (Sample Game 3) it agrees with `kloppy`,
the library the challenge's starter code uses, on all 440,000 player and 14,060 ball positions
checked (differences only at float32 rounding), with no disagreement on when the ball is missing
and the same home team. It gives the identical file whether lines end in LF or CRLF, carry
trailing spaces or a byte-order mark, arrive out of order, or write a missing player or ball as one
bare `NaN` instead of one per channel. The output is also identical with 1 to 4 CPU threads.
`run.py` refuses blank or wrongly formatted files and implausibly low tracking coverage, so a bad
download cannot silently produce a submission.

**The optional video ball detector** (`mtah/video_ball.py`):
- **What it does.** It runs classical OpenCV with no learned weights, inside tracker gaps only,
  near where the ball should be (projected through each frame's homography). It looks for a
  small, round, white blob of ball size, never on a penalty or centre spot, and confirmed by a
  neighbouring frame.
- **How it checks itself.** It **calibrates on the real video without any ground truth**.
  Tracked stretches are hidden and searched like gaps, once around the expected position
  (accuracy) and once where the ball is not (false positives).
- **When it is used.** Only if at least 80% of finds land within 1 m, at most 5% of ball-absent
  searches find something, and the whole video was read. Otherwise it is ignored.
- **Status.** Tested on synthetic video only: there is no video in the sample data, and the
  match video stays off cloud services.

## 5. Possession rules implemented

- **Change of possession:** needs sustained evidence of control by the other team, about
  0.5–1 s of clear evidence at the shipped costs. Touches without control, deflections and
  won-but-uncontrolled aerial duels don't pay for a switch.
- **Loose or contested ball:** stays with the last team in control.
- **Dead ball:** recognised from the ball outside the lines, a still and unattended ball,
  collective slowdown, long tracker gaps and the kickoff shape. Every stoppage lasts at least
  1 s, and each restart starts a new possession.
- **Frame alignment:** the ATD's `frame_id` 1 is video frame 0, so row *i* of the file is ATD
  record *i* + 1 (the starter code's convention). Halves come from the published
  `games.half_video_blocks`.
- **Abstaining:** frames outside the two halves are `X`. Inside them the decoder always commits
  to `H`, `A` or `D`: spellF1 would replace an `X` with the previous label anyway, and macroF1
  counts it as a miss, so abstaining never helps.

## 6. Limitations

- **Label semantics.** Training labels come from event data on three matches of another
  competition. The ground truth comes from Metrica's corrected tracking, so the two differ at the
  margins.
- **Simulated errors.** Tracking errors and dead-ball ball paths are simulated, not measured.
  The real ATD's error pattern is only approximated.
- **Short possessions.** Only about half of the possessions under 8 s are recovered, often
  because the tracking holds no touch evidence for them.
- **Video detector.** It is unvalidated on real footage, and a non-white ball or heavy blur will
  lower its recall. Its calibration report shows this, and the fail-closed gate then keeps it out.
- **Next steps:** a hand-over detector precise enough to beat its false switches (an oracle
  shows +0.03 available), noise-adaptive touch detection, and a duration-aware decoder.
  Self-training on the match was tried and gave nothing (§4).

## 7. Reproduction

Tested environment: **Python 3.13** on Linux, with the exact package versions in
`requirements.txt` (`run.py` warns if yours differ and records the environment next to the file).

```bash
git clone --recursive REPO_URL && cd FCB-Hacks
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt               # + requirements-video.txt for --video-ball
export MTAH_DATA_DIR=/path/to/challenge/bundle  # or unpack it into ./data
RUN_COMMAND                                   # the exact command that produced the file
```

- **Same-output check without match data:** `python scripts/make_fake_bundle.py`, then
  `python run.py --data-dir cache/fake_data --out check.csv`. That builds a stand-in bundle in the
  challenge's format from Metrica's public sample data. The file's SHA-256 should start
  `b296386c9bad3790` on Linux x86-64 with the pinned environment. The previous model's output
  also matched byte for byte on macOS x86-64.
- **Determinism:** fixed seeds, no randomness at prediction time, pinned packages and pinned
  sample data. Two runs give byte-identical files. When the model was committed, retraining it
  from scratch (`python train.py`) gave the same output byte for byte (compare SHA-256 of the
  CSV before and after). With `--video-ball` the file also depends on how the installed OpenCV
  decodes the video, so it is pinned in `requirements-video.txt` and checked like the rest.
- **Validation:** `scripts/evaluate.py` and `scripts/crossval.py` reproduce §4.
- **Tests:** `pip install -r requirements-dev.txt && python -m pytest -q tests`.

**Rules compliance.**
- No hand labels, and no human-produced data about the competition match.
- The private ground truth is never used, and the public 5 minutes are never used for choices.
- No match data, or anything derived from it, is in the repository: `data/`, `cache/` and
  `submissions/` are gitignored, and `scripts/package_submission.py` refuses to package if any
  is tracked.
- The match data was never sent to a hosted LLM. The pipeline was developed against public
  sample data only.

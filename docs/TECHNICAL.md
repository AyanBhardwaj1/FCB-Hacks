# Frame-level ball possession from automatic tracking

**More than a Hack 2027, Phase 1** · Team: **TEAM_NAME**

| | |
|---|---|
| possession file | `TEAM_NAME.csv`, SHA-256 `CSV_SHA256` |
| produced by | `RUN_COMMAND` at commit `GIT_COMMIT` |
| repository | <REPO_URL> (public); regenerate with the command above, see §7 |

## Summary

Gradient-boosted frame classifiers read team-level tracking features and give, per frame,
P(ball dead) and P(home has it | alive). A three-state Viterbi decoder turns those into possession
intervals, paying a price for every change of state. That price is the challenge's
"controlled possession" rule. **No labels from the competition match are used anywhere.** The
models learn from Metrica's three public sample games, whose tracking is degraded to look like
the challenge's automatic ATD.

On those games (leave one game out, unseen degradation seeds, scored with the challenge's own
`possession_metrics.py`) the pipeline reaches **spellF1 0.758 ± 0.014** (macroF1 0.904). The
challenge's baseline method scores ≈ 0.20 on the same data, against 0.2433 on the real match.
That match is the main evidence that the degraded games are a fair stand-in. On tracking worse
than we expect the real data to be, spellF1 is 0.699 ± 0.015. These are not competition scores.

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
- **Possession-equal weights.** spellF1 counts every possession once, so the possession model
  weights each training frame by 8 s ÷ the length of its possession (clipped to [0.25, 4]).
  Without this, short possessions, half of all possessions, are drowned out.

**Decoder:** Viterbi over H, A, D for each half.

- **Emissions:** tempered log-probabilities.
- **Costs:** a fixed cost for every switch (64), stop (40) and restart (40). A deflection or a
  won header is a few frames of evidence and doesn't pay for a switch; sustained control does.
- **Two structural rules keep "dead" honest:**
  - a stoppage lasts at least 1 s (1 in 850 real stoppages is shorter);
  - a hand-over always costs at least 16 less than a stop plus a restart.

  Without them the decoder can hand possession over through fake stoppages, which flatters
  spellF1 while labelling live play dead. This is exact (O(1) per frame) and checked against
  brute force.

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

- **Who labelled what.** The events were produced and published by Metrica Sports. We did not
  label anything: frame labels are derived from those events **mechanically**, by code in
  `mtah/sample_games.py`:
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
  Each game is degraded with four seeds, giving about 1.7 M training frames.

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

| fresh seeds 401–403 | spellF1 | macroF1 | spellF1, harsh tracking |
|---|---|---|---|
| baseline idea (nearest player, ±2 s vote) | ≈ 0.20 | ≈ 0.46 | — |
| **this pipeline (shipped)** | **0.758 ± 0.014** | **0.904** | **0.699 ± 0.015** |

What moved the score (paired, on fresh seeds, details in `docs/DEVLOG.md`):

| change | spellF1 |
|---|---|
| frame models + decoder vs the baseline idea | +0.50 |
| touch and turnover features | +0.025 ± 0.005 |
| randomized training noise + dead-ball boundary features | +0.005 (realistic), **+0.061 ± 0.011 (harsh)** |
| possession-equal weights, 1 s stoppages, margin rule | **+0.016 ± 0.003** (9/9) |

Measured and **not** shipped: a second-stage context model, Smart Tagging priors, pass and
steal sequence features, a learned hand-over detector (opt-in `--hand-over`; it raised short
possession recall but invented as many switches), and longer ball interpolation. None improved
spellF1 beyond noise.

**Error profile** (shipped pipeline, 2 s matching window):
- 63% of true possession switches are found;
- 71% of stops and 76% of restarts are found;
- with perfect team labels spellF1 would be ≈ 0.93, and with perfect dead-ball labels ≈ 0.79, so
  possession hand-overs are where the remaining score is.
- **Ceiling for ball detection:** the shipped model, unchanged, gains **+0.036 ± 0.009** (9/9)
  when 20% instead of 36–46% of the ball is missing. That is why the optional video detector
  exists.

**Reading the real files.** The FIFA/EPTS reader (`mtah/io.py`) follows the metadata's
`DataFormatSpecification`. On Metrica's own EPTS export (Sample Game 3) it agrees with `kloppy`,
the library the challenge's starter code uses, on all 440,000 player and 14,060 ball positions
checked (differences only at float32 rounding), with no disagreement on when the ball is missing
and the same home team. `run.py` refuses blank or wrongly formatted files and implausibly low
tracking coverage, so a bad download cannot silently produce a submission.

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
- **Outside the halves:** frames outside the two graded halves are `X`.

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
  shows +0.03 available), noise-adaptive touch detection, pseudo-labelling the challenge match
  itself, and a duration-aware decoder.

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

- **Determinism:** fixed seeds, no randomness at prediction time, pinned packages and pinned
  sample data. Two runs give byte-identical files. When the model was committed, retraining it
  from scratch (`python train.py`) gave the same output byte for byte (compare SHA-256 of the
  CSV before and after). With `--video-ball` the file also depends on how the installed OpenCV
  decodes the video, so it is pinned in `requirements-video.txt` and checked like the rest.
- **Validation:** `scripts/evaluate.py` and `scripts/crossval.py` reproduce §4.
- **Tests:** `python -m pytest -q`.

**Rules compliance.**
- No hand labels, and no human-produced data about the competition match.
- The private ground truth is never used, and the public 5 minutes are never used for choices.
- No match data, or anything derived from it, is in the repository: `data/`, `cache/` and
  `submissions/` are gitignored, and `scripts/package_submission.py` refuses to package if any
  is tracked.
- The match data was never sent to a hosted LLM. The pipeline was developed against public
  sample data only.

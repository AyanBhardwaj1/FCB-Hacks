# FCB-Hacks: More than a Hack 2027, Phase 1 (ball possession)

Our entry to [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(Metrica Sports × FC Barcelona Innovation Hub × Mobile World Capital Barcelona). The task is to
label every one of the 199,319 frames of FC Barcelona vs Athletic Club as `H` (Barça has the ball),
`A` (Athletic has it), `D` (dead ball) or `X` (no answer). Entries are ranked on **spellF1**, which
counts how many of the match's real possessions are recovered as whole possessions.

**How it works, in one paragraph.** Gradient-boosted frame classifiers read team-level tracking
features: ball proximity, touches detected from ball kinematics, who last and next touched or
controlled the ball, team shape and how it turns over, collective speed, ball out of bounds,
when the ball last left the pitch or came to rest, kickoff shape. One classifier gives P(dead)
and the other gives P(Barça | alive), trained so every possession counts equally, as it does
in the score. A three-state Viterbi decoder then turns them into clean possession intervals,
paying a fixed cost for every change of state. That cost is the challenge's "controlled
possession" rule written as a price: a deflection doesn't pay for a change of possession,
sustained control does. **Training uses no labels from the competition match.** The models learn from Metrica's three public
sample games. Their events give frame labels, and their tracking is degraded to look like the
challenge's automatic ATD (ball missing ~41% of the time in bursts, track fragments, noise). Full
write-up: [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Results so far

Leave-one-game-out on the sample games (3 held-out games × 3 fresh degradation seeds that no
design decision looked at), scored with the challenge's own `possession_metrics.py`
(`python scripts/evaluate.py`):

| method | spellF1 | macroF1 | spellF1, harsh degradation |
|---|---|---|---|
| challenge baseline idea (nearest player, ±2 s majority vote) | ~0.20 | ~0.46 | — |
| **this pipeline** | **0.758 ± 0.014** | **0.904** | **0.699 ± 0.015** |

The baseline's ~0.20 here is close to its 0.2433 on the real match, which suggests the degraded
sample games are a fair stand-in. These are **not** competition scores. The real number only
exists once the organisers score the private full-match ground truth. Details, ablations and error
analysis are in [docs/TECHNICAL.md](docs/TECHNICAL.md#4-validation); the round-by-round history
is in [docs/DEVLOG.md](docs/DEVLOG.md).

## Reproduce

```bash
git clone --recursive <this repo> && cd FCB-Hacks      # --recursive pulls the challenge repo
python3.13 -m venv .venv && source .venv/bin/activate   # Python 3.13 (see .python-version)
pip install -r requirements.txt                         # exact pins: the file must regenerate byte for byte

# 1. put the challenge data where the challenge's DATA.md says (never commit it):
#      data/tactical/BAR-ATH_tactical_atd.xml, ..._atd.txt, ..._smart_tagging.xml
#    or export MTAH_DATA_DIR=/path/to/bundle

# 2. produce the submission (uses the committed model in models/)
python run.py --team <team_name>        # -> submissions/<team_name>.csv (+ .run.json manifest), validated

# 3. package the deliverables (refuses unless the file is valid, unchanged, made by the pushed,
#    clean commit, and no match data is tracked) -> submissions/<team_name>.{csv,md}
git push
python scripts/package_submission.py --team <team_name>

# optional: also search the tactical video for the ball inside tracker gaps (runs locally,
# ~20-40 min once, cached; merged only if its self-calibration on the video passes)
pip install -r requirements-video.txt
python run.py --team <team_name> --video-ball

# optional: retrain the model from the public sample data (fetched automatically)
python train.py
# optional: the validation behind the numbers above
python scripts/evaluate.py --tag mine          # held-out scores, both scenarios
python scripts/crossval.py                     # decoder-cost sweep (degradation seed 100)
```

`run.py` runs the challenge's own validator on the file it writes. It also scores the file
against the public 5-minute window, as a check that frames line up, **not** for tuning. The
rules forbid tuning on that window.

No match data? `python scripts/make_fake_bundle.py` builds a stand-in bundle in the challenge's
exact format from a public sample game, and `python run.py --data-dir cache/fake_data` exercises
the whole path.

## Layout

```
run.py                  challenge data -> submissions/<team>.csv (validated) + run manifest
train.py                public sample games -> models/frame_models.pkl
mtah/io.py              FIFA/EPTS reader (follows the metadata's DataFormatSpecification), writers
mtah/features.py        per-frame features from team-level tracking
mtah/model.py           gradient-boosted frame models (optional stage 2), exact team symmetry
mtah/decode.py          three-state Viterbi with switch / stop / restart costs
mtah/tags.py            optional Smart Tagging priors (off by default, see the write-up)
mtah/sample_games.py    Metrica sample games: event-derived labels, ATD-like degradation
scripts/evaluate.py     leave-one-game-out evaluation with paired comparisons (--compare A B)
scripts/crossval.py     decoder-cost sweep on out-of-fold probabilities
scripts/error_analysis.py   transition-level error breakdown and oracle upper bounds
scripts/dead_analysis.py    dead-ball breakdown: what stopped play, restart type, duration
scripts/handover_analysis.py  possession-switch breakdown: how the ball changed hands, spell length
scripts/short_analysis.py   short-possession anatomy: touches, passes, ball visibility, recovery
scripts/gap_analysis.py     how well the ball can be placed inside tracker gaps (vs the truth)
mtah/video_ball.py          optional ball detection from the tactical video, inside tracker gaps
scripts/make_fake_bundle.py   challenge-format stand-in data for smoke tests
scripts/package_submission.py  pre-upload checks; fills the technical document's placeholders
mtah/env.py             pinned-environment check recorded in every run manifest
docs/TECHNICAL.md       the short technical document (submission deliverable)
docs/DEVLOG.md          every validation round, newest first
external/more-than-a-hack-2027   the challenge repo (git submodule): scorer, games.py
tests/                  python -m pytest -q
```

## Rules we hold ourselves to

- **No hand labels on the competition match, and no use of its private ground truth.** The only
  labels are mechanical ones from public events of *other* matches (allowed by RULES.md).
- **Match data never goes into this repository** (`data/`, `cache/`, `submissions/` are
  gitignored) and **never goes to a hosted LLM**, as the terms of use require. The pipeline was
  developed against public sample data only.
- **Deterministic.** Pinned dependencies and Python version, fixed seeds, no early-stopping
  randomness: the same data and repo give the same file. (With `--video-ball` that also needs
  the pinned OpenCV, since a different video decoder can see slightly different pixels.) `run.py` warns if the environment
  drifts from the pins and records it next to the file.

## Team

**O(1)Burger.** Teams must have 3–5 members to be eligible for the Innovation Challenge
and the final.

# CLAUDE.md

Entry to More than a Hack 2027, Phase 1 (frame-level ball possession for FC Barcelona vs Athletic
Club). README.md explains the method; docs/TECHNICAL.md is the submission's technical document.

## Data licence: hard rules for any Claude session in this repo

The match data is licensed for the competition only, and its terms forbid making the data, or
any extract of it, available to a hosted LLM. So:

- **Never read, open, print, `head`, `cat`, `grep`, sample or summarise** anything in `data/`,
  `cache/` or `submissions/`, or any match video, homography, ATD or Smart Tagging file. That
  includes the possession CSV the pipeline writes and its per-frame probabilities.
- Run the pipeline's commands and read **only what they print**: counts, coverage, scores,
  hashes, paths and error messages. If a traceback quotes a line of data, don't repeat it.
- Never commit any of those files (they are gitignored; keep it that way).
- `cache/fake_data/` is built from Metrica's *public* sample data and may be inspected freely.

## Producing the submission (macOS / Linux)

Python 3.13 with the exact pins in requirements.txt. With conda (the user's Mac has it):

```bash
conda create -y -n fcb python=3.13 && conda activate fcb
pip install -r requirements.txt
git submodule update --init          # the challenge repo, if the clone was not --recursive
```

1. **Same-output check (public data only).** Builds a stand-in bundle from Metrica's public
   sample game (fetches ~230 MB on first run) and runs the pipeline on it:

   ```bash
   python scripts/make_fake_bundle.py
   python run.py --data-dir cache/fake_data --out /tmp/check.csv
   shasum -a 256 /tmp/check.csv       # expect 7a51203b824eba14... (same on Linux and macOS x86-64)
   ```

   A different hash with an otherwise clean run means this machine's floating point differs
   from the reference one; report it to the user rather than "fixing" code to match.

2. **Match data.** The user places the challenge bundle so that `data/tactical/BAR-ATH_tactical_atd.xml`
   and `data/tactical/BAR-ATH_tactical_atd.txt` exist (`ls data/tactical` to check names only),
   or exports `MTAH_DATA_DIR` to the folder containing `tactical/`.

3. **The possession file.** With the user's exact registered team name:

   ```bash
   python run.py --team "<team name>"
   ```

   Sanity checks on what it prints: ball seen on roughly 59% of graded frames and ~20 players
   tracked (it refuses below 20% / 10); no environment-drift warning; the public 5-minute
   window spellF1 should be well above the official baselines on it (0.29-0.42). Near zero means
   misaligned frames. That window is a check only: never tune anything on it (challenge rules).

4. **Package.** The repository must be clean and pushed (run.py's output is gitignored):

   ```bash
   python scripts/package_submission.py --team "<team name>"
   ```

   It writes `submissions/<team>.csv`, `<team>.md` and `<team>_manifest.json`, and prints the
   form fields. The user uploads the .csv and .md and pastes the repository URL.

## Development

- Tests: `python -m pytest -q tests`; lint: `python -m pyflakes run.py train.py mtah scripts tests`.
- Changes to code under `mtah/` change the submitted file: commit and push before step 3, or
  packaging refuses.
- Don't change `models/frame_models.pkl`, `mtah/config.py` costs or features to "improve" the
  match result. Anything tuned on the match would break the rules.

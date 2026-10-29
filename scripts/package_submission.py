"""Check and package the three deliverables before uploading them to the submission form.

    python run.py --team <team_name>                   # writes submissions/<team_name>.csv
    git push                                           # the repository must hold that code
    python scripts/package_submission.py --team <team_name>

It refuses, with the reason, unless all of this holds:

  - the team name is a usable file name (the board names each row after the file)
  - the CSV passes the challenge's own validator, unchanged since run.py wrote it (its SHA-256
    matches the run manifest)
  - the CSV was produced by the commit the repository is at now, with no uncommitted changes,
    and that commit is on origin (fetched now) -- otherwise the public repository cannot
    regenerate the file
  - no match data (ATD, video, homography, Smart Tagging, caches, submissions) is tracked in git

Then it writes, in submissions/ (gitignored):

  <team>.csv             the possession file to upload
  <team>.md              the technical document, with team, file hash, commit and command filled
  <team>_manifest.json   hash, commit, command, repository URL, environment -- for the record
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mtah import config  # noqa: E402

TOKENS = ("TEAM_NAME", "CSV_SHA256", "GIT_COMMIT", "RUN_COMMAND", "REPO_URL", "RUN_PLATFORM")
DATA_PATTERNS = (r"^data/", r"^cache/", r"^submissions/", r"\.mp4$", r"\.npz$", r"_atd\.",
                 r"homography\.csv$", r"smart_tagging")


def check_team_name(team):
    """The registered team name, as long as it is a plain file name (letters in any script,
    digits, spaces, '-', '_' and parentheses; no path separators, dots or edge spaces)."""
    if not re.fullmatch(r"(?!\s)[\w \-()]{1,64}(?<!\s)", team or ""):
        raise ValueError("team name {!r}: use 1-64 letters, digits, spaces, '-', '_' or "
                         "parentheses -- it becomes the file names the board is labelled "
                         "with".format(team))
    return team


def tracked_data_files(files):
    """Tracked paths that look like match data or anything derived from it."""
    return [f for f in files if any(re.search(p, f) for p in DATA_PATTERNS)]


def fill_doc(text, team, sha, commit, command, repo="", platform="", required=TOKENS):
    """The technical document with its placeholders filled in one pass (a value that happens to
    contain a token is left alone); every required placeholder must be there."""
    missing = [t for t in required if t not in text]
    if missing:
        raise ValueError("technical document lacks placeholders {}".format(missing))
    values = dict(zip(TOKENS, (team, sha, commit, command, repo, platform)))
    return re.sub("|".join(TOKENS), lambda m: values[m.group(0)], text)


def repo_url(remote):
    """A browsable https URL for the origin remote (credentials and .git stripped)."""
    m = re.match(r"^git@([^:]+):(.+)$", remote)
    url = "https://{}/{}".format(*m.groups()) if m else re.sub(r"//[^/@]+@", "//", remote)
    return re.sub(r"\.git$", "", url)


def git_state(root=ROOT):
    def git(*args):
        return subprocess.check_output(["git", "-C", root] + list(args)).decode().strip()
    try:
        git("fetch", "--quiet", "origin")       # judge "pushed" by origin now, not stale refs
    except subprocess.CalledProcessError:
        sys.exit("could not fetch origin to confirm the commit is pushed -- check the network")
    return {"commit": git("rev-parse", "HEAD"),
            "clean": git("status", "--porcelain") == "",
            "pushed": git("branch", "-r", "--list", "origin/*", "--contains", "HEAD") != "",
            "tracked": tracked_data_files(git("ls-files").splitlines()),
            "remote": repo_url(git("remote", "get-url", "origin"))}


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def package(team, csv, doc, out_dir, git):
    """Run every check and write the deliverables; SystemExit with the reason on any failure."""
    try:
        check_team_name(team)
    except ValueError as exc:
        sys.exit(str(exc))
    if not os.path.isfile(csv):
        sys.exit("no possession file at {} -- run `python run.py --team {}` first".format(csv, team))
    games, pm = config.challenge_modules()
    error = pm.validate_submission(pm.read_label_csv(csv), games.source_frames("tactical",
                                                                              config.GAME))
    if error:
        sys.exit("INVALID possession file: {}".format(error))
    man_path = os.path.splitext(csv)[0] + ".run.json"
    if not os.path.isfile(man_path):
        sys.exit("no run manifest {} -- regenerate the file with run.py".format(man_path))
    with open(man_path) as fh:
        manifest = json.load(fh)
    sha = _sha256(csv)
    if manifest.get("sha256") != sha:
        sys.exit("{} does not match its run manifest (changed since run.py wrote it?) -- "
                 "regenerate it".format(csv))
    if manifest.get("commit") != git["commit"]:
        sys.exit("the file was produced by commit {} but the repository is at {} -- the public "
                 "repository could not regenerate it; re-run run.py".format(
                     manifest.get("commit"), git["commit"]))
    if not manifest.get("clean"):
        sys.exit("run.py ran with uncommitted changes to tracked files, so no commit holds the code "
                 "that made {} -- commit, push and re-run run.py".format(csv))
    if not git["clean"]:
        sys.exit("the repository has uncommitted changes or untracked files (`git status`) -- "
                 "commit and push them, or delete them, then re-run run.py")
    if not git["pushed"]:
        sys.exit("commit {} is not pushed -- the organisers cannot see it; git push".format(
            git["commit"]))
    if git["tracked"]:
        sys.exit("match data is tracked in git (the data licence forbids publishing it): "
                 "{}".format(git["tracked"][:5]))

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, team + ".csv")
    if os.path.abspath(out_csv) != os.path.abspath(csv):
        shutil.copyfile(csv, out_csv)
    with open(doc) as fh:
        envr = manifest.get("environment") or {}
        text = fill_doc(fh.read(), team, sha, git["commit"], manifest["command"], git["remote"],
                        "Python {} on {}".format(envr.get("python", "?"),
                                                 envr.get("platform", "?")))
    out_doc = os.path.join(out_dir, team + ".md")
    with open(out_doc, "w") as fh:
        fh.write(text)
    out_man = os.path.join(out_dir, team + "_manifest.json")
    with open(out_man, "w") as fh:
        json.dump({"team": team, "csv": os.path.basename(out_csv), "sha256": sha,
                   "commit": git["commit"], "repository": git["remote"],
                   "command": manifest["command"], "environment": manifest.get("environment"),
                   "packaged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                  fh, indent=1)
    return [out_csv, out_doc, out_man]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", required=True)
    ap.add_argument("--csv", help="default submissions/<team>.csv")
    ap.add_argument("--doc", default=os.path.join(ROOT, "docs", "TECHNICAL.md"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "submissions"))
    args = ap.parse_args()
    csv = args.csv or os.path.join(ROOT, "submissions", args.team + ".csv")
    git = git_state()
    paths = package(args.team, csv, args.doc, args.out_dir, git)
    print("all checks passed. For the submission form:")
    print("  Team Name                   {}".format(args.team))
    print("  Possession file             {}".format(paths[0]))
    print("  Public code repository URL  {}  (check it opens in a private browser window)".format(
        git["remote"]))
    print("  Short technical document    {}  (.md is accepted; export to PDF if you prefer, "
          "named {}.pdf)".format(paths[1], args.team))
    print("  record                      {}".format(paths[2]))


if __name__ == "__main__":
    main()

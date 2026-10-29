"""Submission tooling: environment pinning, run manifests, packaging checks.

    python -m pytest -q tests/test_submission.py
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from mtah import env  # noqa: E402


def test_pinned_versions_are_read_from_requirements(tmp_path):
    req = tmp_path / "r.txt"
    req.write_text("# comment\nnumpy==2.5.3\npandas==3.0.5  # inline\n-r other.txt\nfoo>=1\n")
    assert env.pinned(str(req)) == {"numpy": "2.5.3", "pandas": "3.0.5"}


def test_mismatches_report_every_difference():
    pins = {"numpy": "2.5.3", "pandas": "3.0.5", "scikit-learn": "1.9.1"}
    got = {"numpy": "2.5.3", "pandas": "3.1.0"}
    out = env.mismatches(pins, got, python=(3, 12), want_python=(3, 13))
    assert "pandas 3.1.0 (pinned 3.0.5)" in out
    assert "scikit-learn not installed (pinned 1.9.1)" in out
    assert any(m.startswith("python 3.12") for m in out)
    assert not any(m.startswith("numpy") for m in out)


def test_video_pins_add_opencv_to_the_base_pins():
    base, video = env.pinned(), env.pinned(env.REQUIREMENTS, env.REQUIREMENTS_VIDEO)
    assert set(video) - set(base) == {"opencv-python-headless"}
    assert all(video[k] == v for k, v in base.items())


# -- run manifest -------------------------------------------------------------------------------------

def _valid_csv(path):
    from mtah import config
    from mtah.io import write_submission
    games, _ = config.challenge_modules()
    n = games.source_frames("tactical", config.GAME)
    labels = np.full(n, "X")
    labels[11185:90652] = "H"
    labels[110808:190198] = "A"
    return write_submission(str(path), labels)


def test_run_writes_a_manifest_that_pins_the_file(tmp_path):
    import hashlib
    import run
    csv = _valid_csv(tmp_path / "t.csv")
    man = run.write_manifest(csv, ["run.py", "--team", "t", "--data-dir", "/my data"])
    data = json.load(open(man))
    assert data["sha256"] == hashlib.sha256(open(csv, "rb").read()).hexdigest()
    assert data["command"] == "python run.py --team t --data-dir '/my data'"   # shell-quoted
    assert data["commit"] and "python" in data["environment"]
    assert isinstance(data["clean"], bool)


# -- packaging -----------------------------------------------------------------------------------------

def test_team_names_must_be_safe_file_names():
    import package_submission as ps
    for good in ("fcb_hacks", "Team-7", "AyanBots", "FCB Hacks", "Barça", "O(1)Burger"):
        assert ps.check_team_name(good) == good
    for bad in ("", " lead", "trail ", "../x", "a/b", "a.b", "x" * 65):
        with pytest.raises(ValueError):
            ps.check_team_name(bad)


def test_tracked_match_data_is_caught():
    import package_submission as ps
    files = ["mtah/io.py", "data/tactical/BAR-ATH_tactical_atd.txt", "x/BAR-ATH_tactical.mp4",
             "docs/a_homography.csv", "y/BAR-ATH_tactical_smart_tagging.xml", "models/f.pkl",
             "cache/video_ball_1.npz", "submissions/t.csv"]
    assert ps.tracked_data_files(files) == files[1:5] + files[6:]


def test_document_placeholders_are_filled_and_none_left():
    import package_submission as ps
    text = "Team: TEAM_NAME\nsha CSV_SHA256 at GIT_COMMIT via RUN_COMMAND, REPO_URL\n"
    out = ps.fill_doc(text, "t", "abc", "deadbeef", "python run.py --team t", "https://x/y")
    assert out == "Team: t\nsha abc at deadbeef via python run.py --team t, https://x/y\n"
    # one pass: a value that contains a token is not filled again
    assert ps.fill_doc(text, "GIT_COMMIT", "a", "c", "r", "u").startswith("Team: GIT_COMMIT\n")
    with pytest.raises(ValueError):
        ps.fill_doc("Team: TEAM_NAME", "t", "abc", "d", "c", "u",
                    required=("TEAM_NAME", "GIT_COMMIT"))


def test_repo_url_is_browsable():
    import package_submission as ps
    assert ps.repo_url("git@github.com:a/b.git") == "https://github.com/a/b"
    assert ps.repo_url("https://tok@github.com/a/b.git") == "https://github.com/a/b"
    assert ps.repo_url("https://github.com/a/b") == "https://github.com/a/b"


def _git(commit="c0ffee", clean=True, pushed=True, tracked=()):
    return {"commit": commit, "clean": clean, "pushed": pushed, "tracked": list(tracked),
            "remote": "https://github.com/x/y"}


def _package(tmp_path, run_clean=True, **kw):
    import package_submission as ps
    import run
    csv = _valid_csv(tmp_path / "team1.csv")
    run.write_manifest(csv, ["run.py", "--team", "team1"], commit="c0ffee", clean=run_clean)
    doc = tmp_path / "doc.md"
    doc.write_text("Team: TEAM_NAME / CSV_SHA256 / GIT_COMMIT / RUN_COMMAND / REPO_URL\n")
    args = dict(team="team1", csv=csv, doc=str(doc), out_dir=str(tmp_path / "out"),
                git=_git())
    args.update(kw)
    return ps.package(**args)


def test_package_produces_the_named_files(tmp_path):
    out = _package(tmp_path)
    names = sorted(os.path.basename(p) for p in out)
    assert names == ["team1.csv", "team1.md", "team1_manifest.json"]
    md = open([p for p in out if p.endswith(".md")][0]).read()
    assert "team1" in md and "c0ffee" in md and "TEAM_NAME" not in md


def test_package_refuses_a_stale_or_unreproducible_state(tmp_path):
    with pytest.raises(SystemExit, match="commit"):
        _package(tmp_path, git=_git(commit="other"))            # file made by another commit
    with pytest.raises(SystemExit, match="uncommitted"):
        _package(tmp_path, git=_git(clean=False))
    with pytest.raises(SystemExit, match="uncommitted"):
        _package(tmp_path, run_clean=False)                     # run.py ran on modified code
    with pytest.raises(SystemExit, match="pushed"):
        _package(tmp_path, git=_git(pushed=False))
    with pytest.raises(SystemExit, match="match data"):
        _package(tmp_path, git=_git(tracked=["data/tactical/BAR-ATH_tactical_atd.txt"]))


def test_package_refuses_a_file_that_does_not_match_its_manifest(tmp_path):
    import package_submission as ps
    import run
    csv = _valid_csv(tmp_path / "team1.csv")
    run.write_manifest(csv, ["run.py"], commit="c0ffee", clean=True)
    with open(csv, "a") as fh:
        fh.write("")
    lines = open(csv).read().splitlines()
    lines[20000] = lines[20000][:-1] + "A"                     # edited after the run
    open(csv, "w").write("\n".join(lines) + "\n")
    doc = tmp_path / "doc.md"
    doc.write_text("TEAM_NAME CSV_SHA256 GIT_COMMIT RUN_COMMAND REPO_URL")
    with pytest.raises(SystemExit, match="manifest"):
        ps.package("team1", csv, str(doc), str(tmp_path / "out"), _git())


# -- input preflight ----------------------------------------------------------------------------------

def test_preflight_names_a_blank_or_wrong_input_file_without_printing_it(tmp_path):
    import run
    xml, txt = tmp_path / "a_atd.xml", tmp_path / "a_atd.txt"
    xml.write_bytes(b"\xef\xbb\xbf  <?xml version='1.0'?><Main/>")
    txt.write_text("1:NaN,NaN;0.5,0.5:0.4,0.6,0\n")
    assert run.check_inputs(str(xml), str(txt)) is None                     # BOM and spaces ok

    xml.write_bytes(b"\0" * 5000)                                          # failed download
    with pytest.raises(SystemExit, match="blank.*download"):
        run.check_inputs(str(xml), str(txt))
    xml.write_bytes(b"")
    with pytest.raises(SystemExit, match="empty"):
        run.check_inputs(str(xml), str(txt))
    xml.write_bytes(b"\x1f\x8b\x08secret")                                 # not XML
    with pytest.raises(SystemExit, match="not an XML") as exc:
        run.check_inputs(str(xml), str(txt))
    assert "secret" not in str(exc.value)

    xml.write_bytes(b"<Main/>")
    txt.write_text("PRIVATE header line\n1:...\n")                          # not frame records
    with pytest.raises(SystemExit, match="frame records") as exc:
        run.check_inputs(str(xml), str(txt))
    assert "PRIVATE" not in str(exc.value)

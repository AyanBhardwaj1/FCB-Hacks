"""The software environment a submission is made in, and how it differs from the pinned one.

The challenge asks that the repository regenerate the submitted file exactly ("not
approximately -- the same file"). The pipeline is deterministic, but a different scikit-learn
can load the pickled model differently and a different numpy can round a feature differently, so
run.py reports any drift from requirements.txt and records the environment it ran in.
"""

import os
import platform
import re
import sys
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")
REQUIREMENTS_VIDEO = os.path.join(ROOT, "requirements-video.txt")   # adds OpenCV for --video-ball
PYTHON = (3, 13)          # the interpreter the shipped model and results were produced with


def pinned(*paths):
    """{package: version} for every `name==version` line of the requirements files (default
    requirements.txt; `-r` includes are not followed, so pass every file that applies)."""
    out = {}
    for path in paths or (REQUIREMENTS,):
        with open(path) as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                m = re.match(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.+\-]+)$", line)
                if m:
                    out[m.group(1)] = m.group(2)
    return out


def installed(names):
    """{package: installed version} for the given names (missing ones left out)."""
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def mismatches(pins, got, python=None, want_python=PYTHON):
    """Human-readable differences between the pinned and the actual environment."""
    python = tuple(sys.version_info[:2]) if python is None else tuple(python)
    out = []
    if python != tuple(want_python):
        out.append("python {} (pinned {})".format(".".join(map(str, python)),
                                                 ".".join(map(str, want_python))))
    for name, version in sorted(pins.items()):
        if name not in got:
            out.append("{} not installed (pinned {})".format(name, version))
        elif got[name] != version:
            out.append("{} {} (pinned {})".format(name, got[name], version))
    return out


def describe(*requirements):
    """The environment, for a run manifest, against the given requirements files."""
    pins = pinned(*requirements)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": installed(pins), "drift": mismatches(pins, installed(pins))}

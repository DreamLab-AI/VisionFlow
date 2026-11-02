#!/usr/bin/env python3
"""Print a diagram panel with the source lines it cites, read at the topic's
declared revision, so a reviewer without repository access can judge it.

Usage: python3 -I panel-evidence.py <worktree> <panel-id>...
A panel id is like NF-03.4. Citations are `file:N` or `file:N-M`; each is
resolved against the topic's `sources:` by path suffix, and the owning
repository's `verified_commit` entry picks the revision.
"""
import os
import re
import subprocess
import sys

wt = sys.argv[1]
ws = os.path.dirname(os.path.dirname(wt))  # /home/devuser/workspace
diagrams = os.path.join(wt, "docs/diagrams")
CITE = re.compile(r"([A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)(?:-(\d+))?")


def front(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    return m.group(1) if m else ""


def topic_for(panel):
    prefix = panel.split(".")[0]
    for area in sorted(os.listdir(diagrams)):
        d = os.path.join(diagrams, area)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".md"):
                continue
            p = os.path.join(d, f)
            head = open(p).read(400)
            if re.search(r"^id: %s$" % re.escape(prefix), head, re.M):
                return p
    return None


for panel in sys.argv[2:]:
    path = topic_for(panel)
    if not path:
        print("=== %s: topic not found" % panel)
        continue
    text = open(path).read()
    fm = front(text)
    sources = re.findall(r"^\s*-\s*(\S+)\s*$", fm.split("verified_commit")[0], re.M)
    commits = dict(re.findall(r"([A-Za-z0-9_.-]+):\s*([0-9a-f]{7,40})", fm))
    single = re.search(r"^verified_commit:\s*([0-9a-f]{7,40})\s*$", fm, re.M)
    m = re.search(r"^## %s\b.*?(?=^## |\Z)" % re.escape(panel), text, re.S | re.M)
    body = m.group(0) if m else ""
    print("=" * 70)
    print("PANEL %s  (%s)" % (panel, os.path.relpath(path, wt)))
    print(body.strip())
    print("--- cited source lines at the declared revision")
    seen = set()
    for cpath, a, b in CITE.findall(body):
        a = int(a)
        b = int(b) if b else a
        hits = [s for s in sources if s.endswith(cpath) or s.endswith("/" + cpath)]
        if not hits:
            continue
        src = hits[0]
        if src.startswith("../project/agentbox/"):
            # agentbox is a submodule of VisionClaw: read it from agentbox
            # itself, at the topic's agentbox revision.
            rel = src[len("../project/agentbox/"):]
            rev = commits.get("agentbox") or (single.group(1) if single else None)
            repo_dir = os.path.join(ws, "agentbox")
        elif src.startswith("../"):
            repo = src.split("/")[1]
            rel = src.split("/", 2)[2]
            # The VisionClaw checkout lives in `project/` but topics key its
            # revision as `visionclaw`.
            alias = {"project": "visionclaw"}.get(repo, repo)
            rev = (commits.get(repo) or commits.get(alias)
                   or (single.group(1) if single else None))
            repo_dir = os.path.join(ws, repo)
        else:
            repo, rel = "VisionFlow", src
            rev = commits.get("VisionFlow") or (single.group(1) if single else None)
            repo_dir = wt
        key = (src, a, b)
        if key in seen or not rev:
            continue
        seen.add(key)
        r = subprocess.run(["git", "-C", repo_dir, "show", "%s:%s" % (rev, rel)],
                           capture_output=True, text=True)
        if r.returncode:
            print("%s:%d  [unreadable at %s]" % (src, a, rev))
            continue
        lines = r.stdout.split("\n")
        # A wide window: a few lines either side cannot establish multi-step
        # claims, so give the reviewer the surrounding function body.
        pad = int(os.environ.get("EVIDENCE_PAD", "1"))
        lo, hi = max(1, a - pad), min(len(lines), b + pad)
        print("%s @ %s lines %d-%d:" % (src, rev[:12], lo, hi))
        for n in range(lo, hi + 1):
            print("  %5d  %s" % (n, lines[n - 1][:200]))

#!/usr/bin/env python3
"""Per-topic backlog for given diagram areas, plus unassessed ADRs per repo.

Usage: python3 -I topic-backlog.py <snapshot.json> <area>... [--repos repo,...]
"""
import json
import sys

snap = json.load(open(sys.argv[1]))
args = sys.argv[2:]
repos = []
if "--repos" in args:
    i = args.index("--repos")
    repos = args[i + 1].split(",")
    args = args[:i]
for area in args:
    rows = []
    for t in snap["topics"]:
        if t["path"].split("/")[2] != area:
            continue
        left = [p["id"] for p in t["panels"] if p["review"] == "unreconciled"]
        rows.append((t["path"].split("/", 3)[3], len(t["panels"]), len(left)))
    total = sum(r[2] for r in rows)
    print(f"== {area}: {len(rows)} topics, {total} panels left")
    for name, n, left in rows:
        if left:
            print(f"   {left:3}/{n:<3} {name}")
for repo in repos:
    left = [r["path"] for r in snap["adr_records"]
            if r["repo"] == repo and r.get("category") == "decision-candidate"
            and r["review"] == "unreconciled"]
    print(f"== ADRs {repo}: {len(left)} left")
    for p in left:
        print("   ", p)

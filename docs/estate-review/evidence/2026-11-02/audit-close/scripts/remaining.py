#!/usr/bin/env python3
"""Per-repository remaining work from a ledger snapshot.

Usage: python3 -I remaining.py <snapshot.json> [repo-or-area-substring]
A unit counts as assessed when its `review` is anything other than
"unreconciled". With a filter, also lists the unassessed units.
"""
import collections
import json
import sys

snap = json.load(open(sys.argv[1]))
only = sys.argv[2] if len(sys.argv) > 2 else None
values = collections.Counter()

adr = collections.defaultdict(lambda: [0, 0])
left_adrs = collections.defaultdict(list)
for r in snap["adr_records"]:
    if r.get("category") != "decision-candidate":
        continue
    values["adr:" + r["review"]] += 1
    adr[r["repo"]][1] += 1
    if r["review"] != "unreconciled":
        adr[r["repo"]][0] += 1
    else:
        left_adrs[r["repo"]].append(r["path"])

pan = collections.defaultdict(lambda: [0, 0])
left_panels = collections.defaultdict(list)
for t in snap["topics"]:
    area = t["path"].split("/")[2]
    for p in t["panels"]:
        values["panel:" + p["review"]] += 1
        pan[area][1] += 1
        if p["review"] != "unreconciled":
            pan[area][0] += 1
        else:
            left_panels[area].append(p["id"])

print("review values:", dict(values))
print("ADRs (assessed/total) by repo:")
for repo, (a, n) in sorted(adr.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True):
    print(f"  {repo:28} {a:4}/{n:<4} left {n - a}")
print("Panels (assessed/total) by area:")
for area, (a, n) in sorted(pan.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True):
    print(f"  {area:28} {a:4}/{n:<4} left {n - a}")
if only:
    print("Unassessed ADRs:", [p for r, ps in left_adrs.items() if only in r for p in ps])
    print("Unassessed panels:", [p for r, ps in left_panels.items() if only in r for p in ps])

#!/usr/bin/env python3
"""Build one findings index from every remediation queue and wave 4 data file.

Usage: python3 -I findings-index.py <worktree> > index.md

Queues use several shapes (`records`, `issues`, `findings`, `review_findings`,
`rows`); each record is normalised to id, repository, severity, status,
verification and summary. The parallel-implementations queue is a pointer to a
static 134-finding report, so it is listed as one line and never recounted.
"""
import glob
import json
import os
import sys

wt = sys.argv[1]
rem = os.path.join(wt, "docs/estate-review/remediation")
w4 = os.path.join(wt, "docs/estate-review/evidence/2026-11-02/audit-wave4")

LIST_KEYS = ("records", "issues", "findings", "review_findings", "rows")


def pick(r, *keys):
    for k in keys:
        v = r.get(k)
        if v not in (None, "", []):
            return v
    return ""


def norm(r, queue, default_repo):
    summary = pick(r, "summary", "title", "slug", "finding", "description", "claim")
    if isinstance(summary, dict):
        summary = json.dumps(summary)[:160]
    verified = r.get("lead_verified")
    if verified is None:
        verified = bool(r.get("verified_by"))
    return {
        "id": str(pick(r, "id", "finding_id", "row", "key") or "?"),
        "repo": str(pick(r, "repo", "repository") or default_repo),
        "severity": str(pick(r, "severity", "class") or "-"),
        "status": str(pick(r, "status", "disposition", "now") or "open"),
        "verified": "lead" if verified is True else "worker",
        "summary": " ".join(str(summary).split())[:220],
        "queue": queue,
    }


rows = []
pointers = []
for f in sorted(glob.glob(os.path.join(rem, "*.json"))):
    name = os.path.basename(f)
    d = json.load(open(f))
    default_repo = "solid-pod-rs" if name.startswith("solid-") else ""
    if name.startswith("parallel-implementations"):
        pointers.append("- `%s`: %s findings in a static source report; not recounted here." %
                        (name, d.get("issue_count", "?")))
        continue
    lists = [d] if isinstance(d, list) else [d[k] for k in LIST_KEYS if isinstance(d.get(k), list)]
    for lst in lists:
        for r in lst:
            if isinstance(r, dict):
                rows.append(norm(r, "remediation/" + name, default_repo))
for f in sorted(glob.glob(os.path.join(w4, "w4-*.json"))):
    d = json.load(open(f))
    for r in d.get("findings", []):
        if isinstance(r, dict):
            rows.append(norm(r, "audit-wave4/" + os.path.basename(f), ""))

print("| ID | Repository | Severity | Status | Verified by | Summary | Queue |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print("| %s | %s | %s | %s | %s | %s | `%s` |" % (
        r["id"], r["repo"] or "-", r["severity"], r["status"], r["verified"],
        r["summary"].replace("|", "/"), r["queue"]))
print()
for p in pointers:
    print(p)
print()
print("Total indexed findings: %d" % len(rows))

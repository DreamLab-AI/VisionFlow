#!/usr/bin/env python3
"""Merge worker claim proposals into the ledger's claims.json.

Usage: python3 -I merge_claims.py <snapshot.json> <proposal.json>...

The lead runs this after reading each worker's report. For every proposed
claim it checks that the source file is a ledger record (an ADR candidate or a
diagram topic), that every named panel exists in that topic, and that the
report exists. It then computes source_sha256 from the snapshot's current
record and report_sha256 from the report bytes, replaces any existing claim for
the same (kind, repo, path) only when the new one names the same or more
panels, and writes claims.json back. It prints what it added, replaced and
rejected; it never invents a claim.
"""
import hashlib
import json
import os
import sys

WT = "/home/devuser/workspace/wt/VisionFlow-audit-wave3"
CLAIMS = os.path.join(WT, "docs/estate-review/completion/claims.json")


def main(snapshot_path, proposals):
    snap = json.load(open(snapshot_path))
    adrs = {(r["repo"], r["path"]): r for r in snap["adr_records"]}
    topics = {(t["repo"], t["path"]): t for t in snap["topics"]}
    claims = json.load(open(CLAIMS))
    index = {(c["kind"], c["repo"], c["path"]): i for i, c in enumerate(claims)}
    added = replaced = 0
    rejected = []
    for prop_path in proposals:
        for p in json.load(open(prop_path)):
            key = (p["kind"], p["repo"], p["path"])
            rec = (adrs if p["kind"] == "adr" else topics).get((p["repo"], p["path"]))
            report = os.path.join(WT, p["report"])
            if rec is None:
                rejected.append((prop_path, key, "not a ledger record"))
                continue
            if not os.path.exists(report):
                rejected.append((prop_path, key, "report missing"))
                continue
            entry = {
                "kind": p["kind"],
                "repo": p["repo"],
                "path": p["path"],
                "source_sha256": rec["sha256"],
                "report": p["report"],
                "report_sha256": hashlib.sha256(open(report, "rb").read()).hexdigest(),
                "coverage": p["coverage"],
            }
            if p["kind"] == "topic":
                known = {x["id"] for x in rec["panels"]}
                panels = list(dict.fromkeys(p.get("panels", [])))
                unknown = [x for x in panels if x not in known]
                if unknown or not panels:
                    rejected.append((prop_path, key, f"unknown or empty panels {unknown}"))
                    continue
                entry["panels"] = panels
            entry["basis"] = p["basis"]
            if key in index:
                old = claims[index[key]]
                if p["kind"] == "topic" and not set(old.get("panels", [])) <= set(entry["panels"]):
                    # Keep both: the older claim's panels stay credited by its own report.
                    claims.append(entry)
                    added += 1
                else:
                    claims[index[key]] = entry
                    replaced += 1
            else:
                index[key] = len(claims)
                claims.append(entry)
                added += 1
    with open(CLAIMS, "w") as f:
        json.dump(claims, f, indent=2, ensure_ascii=True)
        f.write("\n")
    print(f"added {added}, replaced {replaced}, rejected {len(rejected)}")
    for r in rejected:
        print("  rejected:", r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

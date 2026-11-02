#!/usr/bin/env python3
"""Print panel coverage per topic for one worker's claim proposal.

Usage: python3 -I claim-coverage.py <proposal.json>
"""
import json
import sys

for claim in json.load(open(sys.argv[1])):
    panels = sorted(claim.get("panels", []))
    print(claim.get("key") or claim.get("path"), len(panels), panels)

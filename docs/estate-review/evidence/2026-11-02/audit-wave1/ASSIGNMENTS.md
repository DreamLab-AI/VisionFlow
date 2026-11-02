# Audit wave 1: assignments

Each worker owns exactly the files in its row. Read `CONTRACT.md` in this folder first. Topic paths are under `docs/diagrams/`, ADR paths are in the named repository.

| Worker | Slug | Assigned (write only these, plus your report and claims file) |
| --- | --- | --- |
| w-solid-adr | solid-forge-adrs | ADRs in solid-pod-rs: `crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md`, `ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md`, `ADR-2011-extensions-are-compile-time-crates-route-ownership-is-the-exemption.md` (full section assessment); classify `crates/solid-pod-rs/docs/adr/PREAMBLE.md`, `README.md`, `TEMPLATE.md`, `crates/solid-pod-rs/docs/archive/adr/README.md`; historical ADR-057 to ADR-061 in `crates/solid-pod-rs/docs/archive/adr/` (classify as historical and assess every section against the code: still true, superseded by which ADR, or contradicted). No topic edits. Today's SP-GIT-02 repair (`docs/estate-review/2026-11-02-solid-mirror-publication.md` in the worktree) is directly relevant to ADR-2009 and ADR-2010: read it, but re-check its claims at the pinned revision. |
| w-solid-a | solid-composition-and-boot | Topics `solid-pod-rs/01-workspace-composition.md` (12 panels) and `solid-pod-rs/02-server-boot-and-routes.md` (11 panels). |
| w-solid-b | solid-ldp-and-wac | Topics `solid-pod-rs/03-ldp-request-lifecycle.md` and `solid-pod-rs/04-wac-authorisation.md`: assess only the unreconciled panels (03: 7 of 11; 04: 6 of 13). Find which with the snapshot command below. Do not change the already-credited panels' meaning; if you must restamp the topic, re-check them too and say so. |
| w-solid-c | solid-identity-and-storage | Topics `solid-pod-rs/05-identity-nip98-did-oidc.md` (10 of 18 unreconciled) and `solid-pod-rs/06-storage-quota-and-git-pods.md` (9 of 11). Same rule as w-solid-b. |
| w-solid-d | solid-payments-and-release | Topics `solid-pod-rs/07-provenance-and-payments.md` (5 of 19 unreconciled) and `solid-pod-rs/09-ci-release-and-consumer-pins.md` (9 of 9). Same rule as w-solid-b. |
| w-solid-e | solid-federation-forge-mcp | Topic `solid-pod-rs/08-federation-notifications-forge-mcp.md` (17 panels). Today's forge mirror repair changed this area: read `docs/estate-review/2026-11-02-solid-mirror-publication.md` and SP-14 first. |
| w-solid-f | solid-failure-modes | Topic `solid-pod-rs/10-failure-modes-and-invariants.md` (12 panels). |
| w-stale | stale-claim-rebind | The 11 stale ADR claims listed in `STALE.md` in this folder. For each: find the ADR revision that matches the claim's `source_sha256` (search `git log --format=%H -- <path>` in its repo and hash `git show <sha>:<path>`), diff it against the pinned revision, and decide whether the named report's assessment still holds for the new text. Write one report covering all 11; propose a claim only where the report's assessment still covers every changed section (your report then becomes the claim's report, and you must assess each changed section yourself). No ADR edits. |
| w-reconcile | earlier-review-reconciliation | Read-only sweep: find panel IDs (`XX-NN.N`) and ADR sections given an explicit disposition in existing dated reports under `docs/estate-review/*.md` (excluding `2026-11-02-*`), whose panel or ADR is still `unreconciled` in the snapshot. For each, check the report really assesses that unit and its source is unchanged since, else drop it. Write a report listing what qualifies and what does not, and claim proposals (the named report is the earlier report, not yours, when it qualifies unchanged). Do not edit topics or ADRs. |
| w-loom-dm | loom-and-dream-machine-adrs | All numbered ADR candidates in loom (8) and dream-machine (8). Find them with the snapshot command below. Full section assessment for each. |

Unreconciled units for your assignment come from the lead's fresh snapshot:

```sh
python3 -I - <<'EOF'
import json
s=json.load(open('/home/devuser/workspace/wt/VisionFlow-audit-wave1/docs/estate-review/evidence/2026-11-02/audit-wave1/snapshot-wave1-start.json'))
for t in s['topics']:
    for p in t['panels']:
        if p['review']=='unreconciled': print(t['path'], p['id'])
for r in s['adr_records']:
    if r['review']=='unreconciled' and r['category']=='decision-candidate': print(r['repo'], r['path'])
EOF
```

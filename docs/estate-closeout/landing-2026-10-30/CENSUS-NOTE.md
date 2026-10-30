# Census correction (manual)

`adr-census.sh` reproduces `census-after.tsv` and its summary, which read **187 open / 23 closed**.
The published figures are **188 open / 22 closed / 9 moved**, and the difference is one manual correction:

- ontomark `ADR-OM-003` uses a single bullet status line (`- Status: Accepted`). The script treats a
  single-axis record as closed when that line reads Accepted, but this record's body says activation is
  **staged** (installed-Obsidian acceptance not recorded). It is counted open.

Found by the cross-vendor audit (`codex-audit-r1.md`, claim 1). A rerun of the script will still print
187/23 until single-axis records are read for an activation statement in the body.

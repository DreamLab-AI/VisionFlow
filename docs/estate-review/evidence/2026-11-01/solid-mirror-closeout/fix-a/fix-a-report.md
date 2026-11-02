# Commit A fix round — report (uncommitted; worktree untouched otherwise)
All 7 gates exit 0 (run 2). Run 1 failed gates 4/5/6 on my own regressions (bare `repo::` under cfg(anchoring) at merge.rs:441; clippy type_complexity), fixed.
Deviation: item 8 listing OMITS an unmarked mirror instead of 503ing the whole /api/repos and home index; summary route returns 503. One-line flip available.
Baseline test module diff: fix-a-test-baseline.diff (only item 13, prepare delegation, fix_a opener). Reconstructed from 31fe898 + impl-a.patch (5151 lines, matches start).
Red provenance: item 5 over-cap red and item 7 record red appeared only after an intermediate change (absolute --git-dir); items 11/12/13 green pre-fix, proven by mutations M1-M3 (fix-a-mutations.txt).
MSRV 1.89 unverified (only rustc 1.99.1 offline).
Details: fix-a-red.txt, fix-a-gates.txt, gate-logs/.

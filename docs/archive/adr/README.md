# ARCHIVED — ADR (VisionFlow canon)

**Frozen:** 2026-08-31. **Do not add or edit records here.**

These seven ADR records (ADR-001..007) sat loose in `docs/` and drifted from the
code — most sharply ADR-001, whose Rust/WASM website premise no code in this repo
implements. They were retired in the archive cut of 2026-08-31 and are kept
read-only for history and to resolve inbound cross-references.

The living decision surface is now:

- What this repo is & runs today ... [`docs/BASELINE-visionflow.md`](../../BASELINE-visionflow.md)
- New ADR ledger ................... [`docs/adr/`](../../adr/)

New decisions go in `docs/adr/` using [`docs/adr/TEMPLATE.md`](../../adr/TEMPLATE.md).
The cut itself is recorded in
[`docs/adr/ADR-2001-corpus-consolidation.md`](../../adr/ADR-2001-corpus-consolidation.md).

## Archived records

| Legacy | Title | Legacy status | Where its truth lives now |
|--------|-------|---------------|---------------------------|
| ADR-001 | visionflow.info website technology stack | Accepted (D2/D4 superseded) | `BASELINE-visionflow.md` — website is static HTML/CSS/JS, no WASM |
| ADR-002 | Ecosystem alignment governance | Accepted | `BASELINE-visionflow.md` (canon role, Invariant 4); `docs/architecture/compatibility-matrix.md` |
| ADR-003 | Judgment broker distributed architecture | Accepted (amended) | Cross-repo (nostr-rust-forum/agentbox/VisionClaw); `BASELINE-visionflow.md` divergences |
| ADR-004 | Gap-close sprint governance | Accepted | `docs/registers/gap-register-*.md`; `BASELINE-visionflow.md` |
| ADR-005 | Gap-close canon decisions | Proposed | Drift/diagram gates in `scripts/` + `.github/workflows/`; `BASELINE-visionflow.md` |
| ADR-006 | Terminology canon & feedback loop | Accepted | `docs/terminology.md`; cross-repo bridge findings |
| ADR-007 | Close the governance loop | Proposed | Cross-repo; `BASELINE-visionflow.md` (open item) |

**Not in this cut:** the `docs/engineering/` ADR sequence (its own ADR-004/005)
was left in place — those are a distinct namespace and were not loose in `docs/`.

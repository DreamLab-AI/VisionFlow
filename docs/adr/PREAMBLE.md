**How to work against this pack** (engineering / build-with-quality agents start here):

The ADR pack for this repo is **its living governing document in `docs/` plus the
ledger records below that amend it**. The living doc is normative — its
*Invariants* section is the compliance surface and its *Change process* section
says how to amend it:

| Domain | Governing document |
|---|---|
| What VisionFlow is, the static website, the canon/governance role, CI gates | [`../BASELINE-visionflow.md`](../BASELINE-visionflow.md) |

**Lookup order:** governing doc → its `file:line` citations into code/config →
the ledger records below → `docs/archive/adr/` **only for rationale and history —
never as authority** (the archive is the pre-2026-08-31 ADR-001..007 corpus, frozen
precisely because it drifted from the code; see
[`../archive/adr/README.md`](../archive/adr/README.md) for the legacy-record map).

**Making a decision:** copy [`TEMPLATE.md`](TEMPLATE.md) to `ADR-NNNN-slug.md`
(next free number), fill the three-axis status honestly, update
`BASELINE-visionflow.md` **in the same change**, and regenerate this index
(`node scripts/adr-index-gen.cjs docs/adr` — it fails CI on invalid frontmatter).

# ADR status and evidence contract

Effective closeout interpretation, 2026-09-07. This makes the existing three-axis ledger schema explicit; it does not ratify proposed decisions or promote deployed state.

| Axis | Values | Question answered |
|---|---|---|
| Decision | proposed, accepted, rejected, superseded | Has the architecture decision been adopted? |
| Implementation | none, partial, complete | How much of the stated implementation scope exists? |
| Activation | inactive, staged, live | Where has that implementation been activated? |

The axes are independent, not one ordered completion scale. Accepted/partial/live is valid when an adopted design has only part of its scope running. Proposed/partial/inactive is valid for a candidate implementation. Complete/staged does not establish live acceptance. A rejected or superseded design can leave implementation or deployed residue; record that residue and its removal requirement instead of rewriting its status to make the tuple look tidy.

`verified_commit` and `verified_paths` bind source claims. A source test cannot promote activation. A runtime claim needs the effective profile, loaded artefact identity, target and an observed acceptance result. A failed check remains evidence even when a later repair passes. Missing deployment evidence remains unknown; it is not an inferred failure or an inferred success.

A supersession edge means the successor replaces the predecessor's decision scope. A `lineage` mention means ancestry or rationale and must not be converted automatically into supersession. A partial amendment needs a section-level mapping. Frozen archives remain historical under the corpus-consolidation decisions. An empty `supersedes` list is not itself a defect; missing disposition for an applicable old obligation is.

The ledger generator validates enum membership and required fields. Estate source-review tables, historical routing maps and test/runtime receipts establish the additional meaning. The [2026-09-07 audit](../estate-review/2026-09-07-estate-audit.md) supplies the current mappings and limits. The [master board](../../../project/docs/TODO-unified.md) retains unresolved section-level lineage and deployment obligations.

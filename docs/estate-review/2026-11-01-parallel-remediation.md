---
title: Parallel audit intake and remediation evidence
date: 2026-11-01
status: in-progress
type: explanation
---

# Parallel audit intake and remediation evidence

Your [parallel-implementation report](2026-11-01-parallel-implementations.md) is merged into VisionFlow. Its original `d2b2a19` was rebased from `b205687` onto actual main `bc97713`, producing **`9c2ac2c`**, then fast-forwarded. Report bytes are unchanged. Nothing was pushed. Its 134 distinct issues are preserved individually in the [remediation queue](remediation/parallel-implementations-2026-11-01.json), including merged IDs, original severity, source status, anchors and proposed canonical path.

Importing this report does not close its issues or certify every proposed deletion. Its reachability evidence is static and pinned to the report's source revisions. Its diagram census of 121 topics is that audit's baseline; it does not replace the whole-estate completion ledger. Eighteen rejected/folded hypotheses remain in the report, including the corrected vault-core dependency finding. RuView stays excluded.

## ABI-05: signature rejection fixed, consolidation still open

At Agentbox `9dfc27b16`, `RelayConsumer._verifySig` accepted id/sig strings whenever loading `nostr-tools` or calling its verifier threw. Injected-verifier exceptions escaped entirely. This helper guards local inbound events, forum governance events and fetched governance requests. A roster check after it cannot establish that an event was signed by the claimed roster member.

**Agentbox commit `0ec2956434f9c61eaf40477df58acc40b1b679c3` is merged locally.** The helper now requires literal `true`, catches errors from either verifier path, logs the verification failure once and returns false. Tests explicitly inject their verifier. No structural fallback remains. The existing `authority-consumer` already uses strict success and fail-closed exceptions; this patch aligns the relay path's rejection semantics with it. It does not replace its rosters or decision stores.

Five new regression tests exercise the actual loaded consumer. Four fail against the original code and all five pass after the fix. Failure cases prove that no governance notification, disk record, dedup entry or orchestrator dispatch occurs. Controls use a real signature from a known synthetic key, altered content and a malformed event. Fourteen existing forum-governance tests pass, including roster, deduplication, cursor and dispatch behaviour. No sockets, live relay or production key are used.

Evidence: [before](evidence/2026-11-01/parallel-intake/relay-before.txt), [after](evidence/2026-11-01/parallel-intake/relay-after.txt), [existing suite](evidence/2026-11-01/parallel-intake/relay-forum-tests.txt). The committed test is `tests/sovereign/relay-signature-failure.test.cjs`; the command is `npm run test:relay-signatures`. Agentbox's governance baseline records the repair and its activation limit.

Deployment is outstanding. A source merge does not update a running image. The wider ABI-05 issue remains **partially remediated**: establish one authoritative ingress and roster contract, remove the unused competing responsibility only after tracing all callers, and exercise the deployed relay-to-orchestrator path. The existing authority consumer remains in place because deleting it without migrating its signing/dashboard responsibilities would break a different path.

## Diagram disposition

[AB-40.1](../diagrams/agentbox/40-relay-signature-rejection.md) records the repaired check at the exact Agentbox commit. AB-13.7 now explicitly shows that its older declared revision's exception path accepted string-shaped signatures; it links the current repair rather than presenting the old invariant as enforced. AB-13's other sources have changed since its declared revision, so the topic is not broadly re-stamped or counted as newly assessed. Whole-topic refresh remains queued.

The [Solid payment/provenance audit](2026-11-01-solid-payment-delivery.md) adds independent runtime evidence: private provenance metadata bypass, historical sidecar mismatch, repeated anchor charging, incomplete receipt publication and canonical-state read fallback. These findings complement EST-08's receipt-contract comparison. They remain open product work, with precise acceptance criteria in that report. VisionClaw's VCS-10 is a different caller and is not closed by either the Solid tests or this Agentbox fix.

## Next remediation obligations

Revalidate VCS-10 against the current VisionClaw branch and keep springs work isolated. Route authenticated payment operations through the established URL, replay and body-binding verifier, then test altered-body and replay rejection at the HTTP boundary. Verify FWP-04's deletion ordering and failure recovery before changing KV/D1 ownership. Stop ABM-01's vectorless upsert through the governed memory API contract; no raw database repair is authorised or needed for code inspection. Keep the crypto-category severity of ABI-02/ABR-11 distinct from their limited exposure when prioritising deletion.

This queue preserves all 134 distinct issues. Only ABI-05's exception/structural fallback is remediated in this intake batch; its full record stays partial. The other 133 retain their imported status pending scoped reconciliation, including the two already resolved at the parallel audit's stamp. No findings are silently converted into closed tasks. VisionClaw source and springs remain untouched; its Agentbox submodule checkout now contains the explicitly requested repair, with no host gitlink commit or container rebuild.

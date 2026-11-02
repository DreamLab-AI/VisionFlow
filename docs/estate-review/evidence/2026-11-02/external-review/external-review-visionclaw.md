# VisionClaw and Codescape external review, 2026-11-02

Source revisions: VisionClaw e78210c4d; standalone Codescape
58108c0357bd0dbf6588f8b6adf6e044f4c04556. Source tracing only; no GPU,
headset or deployed-service test repeated.

Full 17-row disposition: VisionClaw
`docs/reference/external-review-2026-11-02.md`, merged main `221d8b559`.
Eight ADR notes: 2028, 2058, 2061, 2141, 2164, 2166, 2169, 2172.
Governors GPU-wire-abi and XR-client link the scoped review. Status axes and
verified_commit stamps deliberately unchanged.

Corrections to external claims:

- ABI212 and 53 manifest fields supported, but size assertions alone cannot
  establish field/type identity. Actual field-layout checks exist.
- GPU oracle deltas are fixture observations, not proofs. 3.4e-11 PageRank is
  canonical_live_scale; two_clique records4.3e-10. LOF4.759e-7 is corrected
  bounded-neighbour fixture. Earlier receipts retained, no repeat here.
- Current link pack is Rust. GDScript caps512 link records; deduplicated endpoint
  markers are extra (up to1024). Total allowance mismatch remains, no fixed
  buffer-overflow evidence.
- @ontology provenance producer is VisionClaw XR link_index.rs281-285, not
  Codescape ingest. repo empty and revisionNone remain.
- GPU hold messages carry holder identity; remaining node-only bookkeeping
  is orchestrator NodeInteractionMessage, not whole hold system.
- AppState actor list in external summary is not a current exhaustive roster;
  VoiceInterfaceActor/ElevationActor are not boot symbols there.

Open source findings rechecked: web throttle resolves unapplied chunk then ack;
drag acceptance lacks ordered post-release observer samples; XR URL-only upgrade
then post-connect auth conflicts with release Origin/upgrade credential gate;
topology generation does not cancel grabs; portal rearm fixed world0.3 ignores
arrangement scale; derived ontology route power_user-only lacks signed envelope.
Three emulated volumes/two backend connections and frame budget100k/reserve95k
supported, headset acceptance remains open. Ack-paced independent ClientStream
supported; zero position traffic only for still acknowledged graph, not all traffic.

Codescape main `4db47ba`: ADR0001 and BASELINE add local presence findings.
`Membership::relay` xr_presence.rs668-677 lacks current local_id membership check;
Inner::send_all521-524 uses same try_send bounded queue for poses and roster.
No cross-repo claims added to its owner baseline.

Validation: VisionClaw ADR generator passes146 records, regenerated index is
unchanged; git diff --check passes in both repos; invisible-Unicode inspection of
new source review finds0 suspicious codepoints. No code changed.
Owned WTs designated for teardown: visionclaw-external-adr-review and
codescape-external-adr-review. Primary dirty operator files preserved.
Estate ledger hashes referencing changed ADR/governing docs need reconciliation;
do not claim a new full-ADR assessment or refresh broad verified_commit.

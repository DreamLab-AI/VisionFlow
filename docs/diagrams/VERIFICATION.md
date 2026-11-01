<!-- Historical generated citation receipt, supplemented by scoped verified evidence on 2026-11-01. -->
# Citation verification

Historical full-corpus results from the previous local `--cite-check` run. A citation counts as
verified only when its bytes were read from the revision its topic declares
(`git show <sha>:<path>`). Anything else was read from the working tree, which
means it was judged against whatever happens to be checked out, not against a
commit. That distinction is invisible in a warning count, which is why it is
stated here: a green run reports what it checked, not merely that nothing objected.

- **25768** verified at the declared revision
- **0** unverified (read from the working tree)
- **0** unresolvable (never checked at all)
- **9** allowlisted (unverifiable), waived by [citation-allowlist.json](citation-allowlist.json)

An allowlisted citation names a file that exists in no commit, so no declared
revision can ever contain it. It is waived by name, one entry per citation, with
a reason; it is never counted as verified. The line checks still applied to it.

<!-- declared-revisions: 09ec90f26aa573731460292d0e80343914cb43d9 0ec2956434f9c61eaf40477df58acc40b1b679c3 1b27f79d08537b61b3d5a5d02cac9ee5999b79ed 28af4365a68d9aacfd7c04f63299cc96ad717ce3 4d361b58ce2b3ee17e1570361eaf78270b630012 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6 560e50c1a35bb9486b9f43ebf074b816c1e01349 5fa0e1325a2f903c175cfe67bbc6d644fcf38267 6500582dfae23bf8a38a6942916d208154e43625 7b95fb20d50a9d06c64fb6d6cc393b16edbbc35d 815f963b1cbd6999e78144ce5475005ae1dd1d8e 8692b12ba987ecb87165566beb7c320a2446e4b4 RuView@5728b90949e60e3f908d996997d453147c1794e0 VisionFlow@4d361b58ce2b3ee17e1570361eaf78270b630012 WasmVOWL@51a1484301901e817757c4cce17b1fece371652a a67cb2a8d474f4ba2e79bdbf55fad2e87bfd7e98 agentbox@6500582dfae23bf8a38a6942916d208154e43625 agentbox@9dfc27b16e82e9786535a55e790710493dd51fdf c39f41f104b114d6a3fa29013daaf6fa58b45080 codescape@13e056d9aa90d6f1b4aaf3b72d69a8cb29446ad1 codescape@58108c0357bd0dbf6588f8b6adf6e044f4c04556 diagram-ir@15cba163da7086a3ebbb5a059c8a8d0ab4ba9706 dream-engine@59a3e215e826aa1e9ebe157d49c1ac504f7f39c9 dream-engine@815f963b1cbd6999e78144ce5475005ae1dd1d8e dream-machine@b77093e564ef689923fb523b0e0b18d2bb4a21a4 dreamlab-ai-website@a67cb2a8d474f4ba2e79bdbf55fad2e87bfd7e98 dreamlab-ai-website@b5eb1ecf96599ebe154ebf23bbe7c703d2e653a5 knowledgegraph@1edcfc93de90ccff564073ef82cb198c422f75e3 knowledgegraph@4ed9ac159daf402b4fb252ce559dbeb894a8d91e loom@5ad860863b588c0cdb2290e31c69e39d82755aeb loom@ba21a689a57dc67dc68ad22e52b058da34a7fb05 nostr-rust-forum@0b7979c404f41414cfdbefffa28fb6d5ae1b1242 nostr-rust-forum@1b27f79d08537b61b3d5a5d02cac9ee5999b79ed nostr-rust-forum@9c36b6032640c89879ae52fe25da99493979a0b6 nostr-rust-forum@c39f41f104b114d6a3fa29013daaf6fa58b45080 ontomark-obsidian@2a6c44e024121ec055329d37915e0ce5a05f75f2 ontomark@0caae5d4b695cb720942b9b9212017c00a80ad41 ontomark@4b9d15df48056971321a62b8c4e905f4536c74db ontomark@560e50c1a35bb9486b9f43ebf074b816c1e01349 prose-sanitiser@b261f65466b12bef2fd7c4957cee463cbe06190f sidestr-rs@9c6de86c657a25ec2f2a189baf63db7dedad40eb sidestr-rs@d8ace3aad2f23eb1d11bd72af4ab0b74d599f618 solid-pod-rs@09ec90f26aa573731460292d0e80343914cb43d9 unmute@e348e567c7020d40020412024f7aa03c38a3bc3e visionclaw@2257ee7fa68a9e56cfedd175cfb4d3a14481d894 visionclaw@531df5ef74453a2ed320bff968c203a9b91faed7 visionclaw@75b72416bac176ca8fde715cac3bd7c77657b3e7 visionclaw@8692b12ba987ecb87165566beb7c320a2446e4b4 visionclaw@d6c091f0ffb6d1af04ce011836c45ace6f4dafd9 visionclaw@e4ba5eb35ee6715cfd7f2daf1cd9af05e8a18cce visionflow@4d361b58ce2b3ee17e1570361eaf78270b630012 visionflow@9cc2266a3f797321e2314741557dbc06061cb4b6 visionflow@baae32a visionflow@c146f8b90f698bde23c75c58a40fc1eba57e346d visionflow@e7a9182 visiongraph@556a0156b5c326fb61141ed2ab1760dd8d9c4dd6 visiongraph@7b0875f2c26b9e69dc5c97e9c88166acc14c47dc visiontouch-rs@7b95fb20d50a9d06c64fb6d6cc393b16edbbc35d vowl-wasm@28af4365a68d9aacfd7c04f63299cc96ad717ce3 -->

The comment above combines the historical revision set with the scoped AB-40 and VC-55
receipts below. The historical totals above are unchanged; they are not fresh
whole-corpus totals. `--check-verification` checks revision-set parity only.

## Scoped supplement — 1 November 2026

[AB-40](agentbox/40-relay-signature-rejection.md) adds revision
`0ec2956434f9c61eaf40477df58acc40b1b679c3`. Its [local gate receipt](../estate-review/evidence/2026-11-01/parallel-intake/new-diagram.txt)
resolves all 11 citations at that revision, with zero fallback, unresolved or
allowlisted citations. The [composition receipt](../estate-review/evidence/2026-11-01/parallel-intake/verification-supplement.json)
binds the topic, log and previous report bytes.

The existing SP-07 and AB-13 topics were also rechecked at their unchanged
stamps: [217 citations](../estate-review/evidence/2026-11-01/solid-payments/render.txt)
and [375 citations](../estate-review/evidence/2026-11-01/parallel-intake/historical-diagram.txt),
respectively. Three AB-13 semantic advisories remain in the audit queue.
No whole-corpus source run was repeated: excluded checkouts remain outside
this audit. Historical references below do not re-admit them.

## Payment authentication supplement — 1 November 2026

[VC-55](visionclaw/55-payment-authentication.md) adds revision
`5fa0e1325a2f903c175cfe67bbc6d644fcf38267`. Its [scoped receipt](../estate-review/evidence/2026-11-01/pay-nip98/diagram-new.txt)
resolves 19 citations at that revision. The historical [VC-04 receipt](../estate-review/evidence/2026-11-01/pay-nip98/diagram-historical.txt)
resolves 207 citations at its unchanged stamp. Both have zero warnings,
fallbacks or unresolved citations. VC-04 received a 120-citation JEV advisory
sample, not complete semantic acceptance. Both topics rendered; VC-04.27 and
VC-55.1 were visually inspected and assessed. Historical totals above remain
unchanged. The [composition receipt](../estate-review/evidence/2026-11-01/pay-nip98/verification-supplement.json)
binds the previous report, topics and logs; no excluded source was accessed.

## Historical exceptions

| Topic | unverified | unresolvable | allowlisted |
|---|---|---|---|
| [visionflow/02-website-build-and-deploy.md](visionflow/02-website-build-and-deploy.md) | 0 | 0 | 4 |
| [visionflow/07-pitch-presentation-reports.md](visionflow/07-pitch-presentation-reports.md) | 0 | 0 | 5 |

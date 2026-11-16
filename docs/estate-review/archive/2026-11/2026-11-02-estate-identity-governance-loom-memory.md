---
title: Estate identity, governance, Loom and memory diagrams
date: 2026-11-02
status: assessed
type: explanation
---

# Estate identity, governance, Loom and memory diagrams

I assessed all 35 panels in four estate topics by source tracing at the wave-2 pins: ES-04 (identity mesh), ES-05 (governance loop), ES-06 (Loom and email privacy path) and ES-07 (memory and embedding). 5 are accurate, 19 corrected, 11 accurate with limits, and none inaccurate-unfixed. All four topics are restamped to the pins, with every citation re-resolved by symbol. The most important finding is that ES-04.4 claimed an agent's Solid pod write falls back to unsigned when the signer is unavailable. That is false at the pin and was already false at the old stamp. With `sign_requests` on, the pods adapter throws `SigningUnavailable` before any byte is sent, and the boot aborts on a failed identity mint. The panel now draws the fail-closed path. Most other corrections are line drift in high-churn agentbox files (`agentbox.toml`, `entrypoint-unified.sh`, `ruvector-mcp.cjs`, `INGRESS-identity.md`, ADR-2122). A few are substantive:

- ES-07.1: the memory server no longer reads `RUVECTOR_PG_CONNINFO`.
- ES-07.10: when the ruflo child is gone, the error comes from the proxy, not from the local stub.
- ES-05.9: receipt idempotency was overclaimed.
- ES-06.6: a Loom Jev arm was called configured when its cited baseline says live.
- ES-06.11: one lesson was over-read from its ADR.

## Findings

**F1 (ES-04.4): the unsigned pod-write fallback does not exist.**
- **Trigger:** the signer is unavailable while `[integrations.solid_pod_rs].sign_requests = true` (`../project/agentbox/agentbox.toml:590`).
- **Consequence:** the old panel told readers that a degraded boot silently writes unsigned under a placeholder identity. That steered attention away from the real, fail-closed behaviour.
- **Evidence:**
  - `buildPodNip98` returns `null` only when signing is off (`management-api/lib/pod-signer.js:73`).
  - The adapter takes `requireSigned` from the same flag (`management-api/adapters/index.js:74`).
  - A null header throws `SigningUnavailable` (`management-api/adapters/pods/_solid-http-base.js:105`).
  - The entrypoint exits 1 on a failed or inconsistent mint (`config/entrypoint-unified.sh:1424-1433`).
  - All of these are at agentbox c57d9c0. `pod-signer.js` is byte-identical at 6500582, and the entrypoint already aborted there (`:1422`).
- **Closed in the topic:** the panel now draws the throw, and a new identity-port note cites `pod-signer.js:76-77`.
- **Residual:** the module comment at `management-api/lib/agent-identity.js:41-43` still says the `did:nostr:local` fallback "runs unconditionally". That comment is stale; the code is correct. Closing it needs a product edit, which I did not make.

**F2 (ES-04.1): a third, identity-less credential on agent events (ADR-2164).**
- **Trigger:** a running task posts its own agent actions with its per-task token.
- **Consequence:** those actions reach the journal with `did: null`. The verification mesh no longer covers every agent-event path.
- **Evidence:**
  - `management-api/lib/agent-event-auth.js:143-145` accepts the token under every policy.
  - `docs/INGRESS-identity.md:338-344` (invariant 12) describes the token.
- **Recorded as:** a Drift note in the topic.
- **To close:** draw the task-token path in the mesh if attribution through tasks matters.

**F3 (ES-07.1): the memory server's connection source changed.**
- **Evidence:** `ruvector-mcp.cjs` now builds its pool from `RUVECTOR_PG_HOST/PORT/DATABASE/USER` plus a 0600 `RUVECTOR_PG_PASSWORD_FILE` (`mcp/servers/ruvector-mcp.cjs:52-55`, defaults in `management-api/lib/pg-config.js:34-37`). It no longer uses `RUVECTOR_PG_CONNINFO`, which VisionClaw's memory cloud still reads (`src/services/memory_cloud_service.rs:43`).
- **Consequence:** the two RuVector clients now take their connection from different configuration.
- **Closed:** corrected in the node.
- **Open:** whether VisionClaw should follow the single-credential-file rule (ADR-2164 amendment 2) is not decided in either repository.

**F4 (ES-07.10): unreachable child versus a stub.**
- **Evidence:** a forwarded tool whose ruflo child is gone gets the proxy's own error (`mcp/servers/ruvector-mcp.cjs:643-646`). The `unimplemented` stub (`:655-657`) answers only tools that are not forwarded: `task_*`, `coordination_*`, or anything with the gate off.
- **Closed:** the panel now draws both branches. The existing Debt note (a manifest boolean with no probe) still stands.

**F5 (ES-05.9): idempotency overclaim.**
- **Evidence:** `mintSpendReceipt` and `mintSpendActivity` derive a content-addressed URN from their inputs and keep no state (`management-api/lib/receipt-minter.js:33-34,45-56`). Nothing in the module stops a double spend.
- **Closed:** the note now says the URN is deterministic and that deduplication is the consumer's job.
- **Open:** identify which consumer, if any, refuses a repeated receipt URN.

**F6 (ES-06.6): a live/staged inversion.**
- **Evidence:** the Open note said the Loom Jev arm was configured and would take effect on the next activation. The baseline it cites says the compose System One consumers are live since the 2026-10-30 recreate (`../project/docs/BASELINE-architecture.md:43-54`). The passage is unchanged since 8692b12; it only moved down one line. Meanwhile the `loom` service is profile-gated (`docker-compose.unified.yml:453-454`).
- **Closed:** the note now says that source cannot show whether the arm serves anywhere.

**F7 (ES-06.11): JEV advisory confirmed by hand.** The L3 node said ADR-2084 explains what the passthrough assertion catches. Lines 26-27 only name the assertion as one of loom-draft's lessons. I reworded the node to what `:26` says. The other advisory, on ES-06.6 at `ADR-142…:54-56`, is a false positive: line 54 is the "first served Ontomark bundle" heading and line 56 is the served record.

**F8 (ES-06.9): Tension added.**
- **Evidence:** the swap runbook asks for `max_tokens >= 2048` (`skills/email-search/SKILL.md:109`). The façade floors at 1536 (`loom crates/loom-facade/src/config.rs:24`).
- **Consequence:** the two numbers answer different questions (a direct model smoke test versus the façade floor) but read as one rule.

**F9 (ES-04.6): the doc drift persists.** `agentbox/docs/SECURITY-profiles.md:11` still says the live mirror hook composes unredacted text. Lines 68-69 and the code (`config/hooks/nostr-live-mirror.cjs:453`) say redaction precedes egress. The topic already carries this as DOC-DRIFT.

**F10 (ES-07.9): the doc drift persists.** `agentbox/docs/LEARNING-memory.md:263-266` (D4) still tells readers to update a README learning-loop row that no longer exists (`README.md:85`).

**F11 (ES-05.14 and ES-05.15): records unchanged.**
- **Forum versions:** forum main at c39f41f has the same k256 0.14.0, p256 0.14.0, solid-pod-rs 0.5.0-alpha.18 and nostr-bbs-core 1.0.0-beta.19 as 1b27f79. The edge kit `9c36b603` still locks k256 0.13.4 and crypto-bigint 0.5.5. The text now names c39f41f.
- **ER-6:** still decided but not live. No agent-governance key exists, and `config/custody/identity-port-acl.json:40` grants none (ADR-2122 `:645-652`).

**F12: wording tied to the old stamp.**
- **What it was:** "resolved at this stamp", "at the earlier stamp" and "VERIFIED AT HEAD" in ES-04.1, ES-05.12 and ES-07.9 pointed at a revision the restamp would have silently changed.
- **Fix:** each now names the revision it means.

## Panel dispositions

| Panel | Disposition | Evidence |
|---|---|---|
| ES-04.1 | corrected | `agent-event-auth.js:192-194` (was 184-186). ADR-2153 `:228`, `:231`. INGRESS `:69-71,244-247`. New ADR-2164 drift at `agent-event-auth.js:143-145`. All other nodes are unchanged and hold. |
| ES-04.3 | accurate | `nip98.rs` is unchanged. Signature verification (`:559-565`) precedes the replay claim (`:573`). TTL is 2×, at `:178`. Simplification: the panel omits the `BodyBinding::Required` → `PayloadHashMissing` branch (`:540-541`). |
| ES-04.4 | corrected | F1. |
| ES-04.5 | accurate | `session-plane-acl.json:6,8-11`, `api.rs:32`, `proxy.mjs:167`, `agentbox.toml:160`. K1's legacy env fallback: `nostr-pod-bridge/src/lib.rs:135-137` (uncited in the topic). D5: ADR-2027 `:136`. |
| ES-04.6 | accurate-with-limits | Hook order (`nostr-live-mirror.cjs:405-414,278-291,453`) and kinds (`:61`) hold. Relay acceptance, Amethyst readability and fail-open are runtime. F9. |
| ES-04.7 | corrected | All INGRESS cites re-resolved (`:18,148-152,254-255,492-495,501-502,506-508,519-524`). Still PROPOSED in the doc. |
| ES-05.2 | accurate | `events.rs:130,140`, `client.rs:267`. |
| ES-05.6 | accurate-with-limits | `broker-bridge.js:435`, `authority.js:226`, `governance-application-receipts.js:26`. VisionClaw's acknowledgement and live responders are not in source. |
| ES-05.7 | corrected | `agentbox.toml:906-907`, `:922` (were 902-903, 918). Extractor, gate and publisher lines unchanged and hold. |
| ES-05.8 | accurate-with-limits | Uncited state model, consistent with ES-05.6 and ES-05.10. Remote states need witnessed receipts. |
| ES-05.9 | corrected | F5. Mandate functions `mandate.js:99,137,163,191,209` hold. |
| ES-05.10 | accurate-with-limits | Uncited summary. Its limits are runtime by its own statement. |
| ES-05.11 | accurate | `exec-record.js:63`. ADR-2071 is accepted/complete/live, `:5-7`. `:193` is the Acceptance heading. |
| ES-05.12 | corrected | "VERIFIED AT HEAD" now names visionclaw 86cdd76. Every cite holds at the pins: `events.rs:91-101,211-217,308-323`, `governance.rs:525,769`, `wrangler.toml:61`, `gate_tier :382-390`, ADR-2087 `:38,50,59,69,86-88`. |
| ES-05.13 | accurate | `enrichment_proposals_handler.rs:366-368,382-409,473,492-494,505-508,516`. `nip_handlers.rs:1344,1360-1379`. `governance.rs:515,525,716`. `broker-bridge.js:488-497`. |
| ES-05.14 | corrected | F11. ADR-2087 `:7,13,166` (exits 1 on the running image, per its review_trigger). ADR-2010/2011 forum lines hold. `workers-deploy.yml:44,287-299` hold. |
| ES-05.15 | corrected | ADR-2122 `:253-254`, `:645-650`, `:652` (were 244-245, 636-641, 643). `acl.rs:71-73,84,295-305` and `identity-port-acl.json:34-44` hold. |
| ES-05.16 | accurate-with-limits | dream-engine read at its stamp 59a3e21: 5 crates, 0.4.0 workspace, dream-gate 0.3.0. Adapter `engine.rs:180,399,1710,1774` holds at c57d9c0. I appended a re-read note. The GitHub forge path is not exercised. |
| ES-06.1 | corrected | ADR-2023 `:64` (was 55). Compose `:376-385,453-454` holds. Model identity (Qwen3.8-27B, per `SKILL.md:95`) and the HP rail/DNAT are documentation, not observed. |
| ES-06.2 | accurate-with-limits | Matches `skills/email-search/SKILL.md:20,25`. Gateway source is not in any pinned repository. |
| ES-06.3 | accurate-with-limits | Matches `SKILL.md:48-50,149-173,223-224`. Gateway source is outside the estate. |
| ES-06.4 | accurate-with-limits | Matches `SKILL.md:31-33,202`. |
| ES-06.5 | accurate-with-limits | Loom constants hold: `config.rs:24,29,35`, `tuning.rs:10-11,24`, compose `:408,416-419`. Jev wiring `routes/mod.rs:383-384,470-476`, `jev.rs:232-234,353`. The benchmark figures are runtime. |
| ES-06.6 | corrected | F6. Recon cites `entrypoint-unified.sh:1221-1229`, `agentbox.toml:1181-1189` (`loom_sync = "full"` at `:1189`). ADR-142 cite widened to `:54-56` (F7). Reload binary lines hold. |
| ES-06.7 | corrected | ADR-2023 `:64`. `SKILL.md:79,109,210-216` hold. |
| ES-06.8 | accurate-with-limits | `flake.nix:4387-4396` and `agentbox.toml:1396-1401` confirm the timeouts (uncited in the panel). Client behaviour is runtime. |
| ES-06.9 | accurate-with-limits | Runbook matches `SKILL.md:101-111`. F8 Tension added. Operational, not source-checkable. |
| ES-06.10 | corrected | `agentbox.toml:1772,1774-1775,1782,2224,2612`. `flake.nix:2246-2248,2902,3964`. `entrypoint-unified.sh:2304-2324,2328`. ADR-2023 `:64-65`. The Resolved note's ADR-2153 bare ref moved from `:82` to `:91`. ADR-2080 `:5-7,13,129-132` and compose `:395` hold. |
| ES-06.11 | corrected | F7. Other ADR-2084 cites (`:21-28,31-33,37-48,52-54`) and `explainer-tools.nix:11-16` hold. |
| ES-07.1 | corrected | F3. Tool lines `251,265,277,288`. `:86-88`, `:152`. `agentbox.toml:685-690`. No `RuVectorAdapter` in VisionClaw `src/` or `crates/` (grep at 86cdd76). |
| ES-07.2 | corrected | `entryId :124`, fail-closed log `:160`. |
| ES-07.3 | corrected | `:239-243`, `:160`. Harness probe `:163-164`. `agentbox.toml:894,914`. `system_one.rs:460`. ADR-2155 is still `staged`. |
| ES-07.7 | corrected | Harness `:81,83-84`. Verdict now in `verdictFromMedians` (`:217-226`). Band 175/102 unchanged (`:32`, fixture `:492-494`). |
| ES-07.9 | corrected | `agentbox.toml:479-480,492-495,504,510`. Feeder `:327-331,517-518,558,637`. LEARNING-memory `:263-266`. F10. |
| ES-07.10 | corrected | F4. `:32,538,564,612,615,618,304-309`. `agentbox.toml:454-455`. |

## Verification

- **Evidence class:** source tracing at the pinned revisions only. No product test, runtime probe or deployment was run, and none is claimed.
- **Pins read:**
  - agentbox `c57d9c0508f4044915f9aecea4b9aa5629cc6a35`, read through the `../project/agentbox` path the generator resolves.
  - VisionClaw `86cdd76355823a51bb64d503c005fe277726e724`.
  - loom `ba21a689a57dc67dc68ad22e52b058da34a7fb05`.
  - nostr-rust-forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`.
  - dreamlab-ai-website `0e1572801d03a1af6d8bc8861f928da087d69eed`.
  - VisionFlow `5f8b1da92ce5d9e6b32a0bd90ec3630c66619083`, the worktree HEAD.
- **dream-engine (ES-05 only):** it has no wave-2 pin, so it stays at its declared stamp `59a3e215e826aa1e9ebe157d49c1ac504f7f39c9`, and its citations were re-read there.
- **Method:**
  - I diffed every `sources:` file between its old stamp and the pin.
  - I read every cited span at the pin.
  - For changed files, I re-derived each line with `git grep -n` on the cited symbol or text, never by shifting.
  - I hand-read short `:NNN` references, which the generator does not check.
- **Gate:** run from the worktree root. Each command was run with its output redirected, and `$?` was taken from the generator itself:
  ```sh
  node scripts/diagram-index-gen.cjs docs/diagrams --only 04-identity-mesh-did-nostr.md --check --cite-check --strict-citations --render   # exit 0; 71 citations verified, 6/6 rendered
  node scripts/diagram-index-gen.cjs docs/diagrams --only 05-governance-loop-across-estate.md --check --cite-check --strict-citations --render   # exit 0; 113 verified, 12/12 rendered
  node scripts/diagram-index-gen.cjs docs/diagrams --only 06-loom-and-email-privacy-path.md --check --cite-check --strict-citations --render   # exit 0; 60 verified, 11/11 rendered; one JEV advisory (ADR-142:54-56), checked by hand and false (F7)
  node scripts/diagram-index-gen.cjs docs/diagrams --only 07-memory-and-embedding-estate.md --check --cite-check --strict-citations --render   # exit 0; 54 verified, 6/6 rendered
  ```
- **Not verified:**
  - The email gateway's own code, which is not in any pinned repository. ES-06.2 to ES-06.4 and ES-06.8 rest on agentbox's skill document.
  - The model actually served behind the Loom. Source names Qwen3.8-27B; the operator's environment notes name a different model. Only source counts here.
  - Live relay, forum-edge or responder behaviour.
  - SONA accumulation, recall figures and rerank activation.
  - The GitHub forge path of the dream engine.
- **Pin mismatch:** VisionClaw 86cdd76's own agentbox gitlink is `c3746299`, not the contract's agentbox pin. I followed the contract.

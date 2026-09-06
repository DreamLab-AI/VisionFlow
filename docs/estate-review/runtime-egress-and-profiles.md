---
title: Runtime profiles and session egress
status: source-and-isolated-probe-verified
date: 2026-09-04
type: explanation
---

# Runtime profiles and session egress

Profiles route harness settings and credentials to the intended provider. They share an OS user and filesystem authority, so directory separation is a configuration mechanism, not a security boundary between mutually untrusted processes. Session mirroring creates a second concern: where selected session text leaves that profile. The implementation has useful off-switches and encryption, but neither establishes a complete content-egress policy.

The [reproducible probes](evidence/runtime-egress-probes.py) and [receipt](evidence/runtime-egress-probes.json) use invented input, temporary profiles, a stub provider executable and mirror dry-run mode. No real credentials, transcripts, provider requests or relay messages were used.

## Provider routing validates a substring

The [OpenRouter](../../../project/agentbox/config/harness-wrappers/openrouter.sh) and [Z.AI](../../../project/agentbox/config/harness-wrappers/zai.sh) wrappers require a profile/settings file, nonempty redirect and token. They pin HOME and CLAUDE_CONFIG_DIR before launching the harness. However, their redirect assertion uses shell pattern `*EXPECTED_HOST*`; it does not parse and compare the URL hostname.

Both actual wrappers launched the stub for an expected URL and for an unrelated hostname containing the expected string (`openrouter.ai.example.invalid` or `api.z.ai.example.invalid`). They rejected a wholly unrelated URL. This contradicts ADR-2007's complete off-target redirect guarantee, so its implementation status becomes partial. No misbilling or real credential transmission was reproduced.

Require explicit scheme/host/port policy, negative fixtures for suffixes, user-info and path/query substrings, and profile-launch receipts that expose effective provider identity without secrets. Validate configuration separation separately from any desired OS isolation.

## Live mirror: selected raw text inside a gift wrap

The [live hook](../../../project/agentbox/config/hooks/nostr-live-mirror.cjs) selects lifecycle lines, user prompts and the last assistant text. It caps the composed message at 4,000 characters and adds a provenance reference where available. This is selected turn text, not an unconditional full transcript upload.

`AGENTBOX_LIVE_MIRROR=0` exits before composition. Without a derivable child key or an explicit recipient it also exits. Otherwise it defaults to a derived child self-DM when a root key is available; an explicit recipient uses the legacy path. An omitted off-switch is not an opt-in gate when those identity inputs exist. Recipient syntax is checked, but the hook does not enforce a separate enumerated recipient allowlist. The relay's admission rules are a different boundary.

The dry-run preserves an invented `password=` sentinel in composed prompt text for both explicit-recipient and child-key configurations. There is no redaction stage between body selection and wrapping. Source wraps the resulting body using NIP-59 before transport; encryption does not make the content redacted. Dry-run writes the composed body to stderr, which matters for diagnostic retention. The hook uses a cloud-relay default and accepts a `ws:` or `wss:` override; it does not fan out over NOSTR_RELAYS.

The exit contract is best-effort: transport rejection and expiry do not fail the calling session. Exit zero therefore cannot serve as proof of delivery or proof that a configured mirror is disabled. This review has not established remote retention or a live privacy incident.

## Digest: a separate provider and publication path

The Rust [session-summary path](../../../project/agentbox/services/nostr-pod-bridge/src/session_summary.rs) requires bridge configuration and a summarisation-provider key. It flattens and trims user/assistant transcript text, sends that text to the configured Z.AI request builder, then assembles a curated digest. The inspected sequence has no redaction step before the provider request. Curating the output does not remove the provider's receipt of the input.

[Digest publication](../../../project/agentbox/services/nostr-pod-bridge/src/lib.rs) signs kind 30840 with rendered summary content. Its local EffectiveMessage has `gift_wrapped: false`; it is not the live hook's NIP-59 path. Actual recipient visibility depends on the relay/read policy and must be verified separately. The Rust run path gates on its own bridge/provider configuration, not AGENTBOX_LIVE_MIRROR. Call-site policy may add further gates; setting the live-hook switch alone does not establish that every mirror path is disabled.

## Closeout requirements

CP-04/07/08 require a shared, explicit content-egress policy covering live messages, summariser input, signed digests and logs. Decide allowed content and recipients; enforce redaction before any external provider/transport; identify encryption and retention per path; and expose disabled, attempted, accepted and failed states without storing sensitive bodies in diagnostics. Verify each path independently and exercise the complete off-switch matrix with network-denial fixtures. Keep ADR-2026 proposed until the owning maintainers adopt and verify that policy.

Secret custody and rotation in ADR-2027 remain a separate open dependency. Record roles and secret references without copying secret values into the documentation. Current endpoint parsing, content scope and gating findings do not establish deployed configuration or imply that all session data has left this container.

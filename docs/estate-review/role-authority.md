---
title: Role authority, revocation and enforcement
status: in-progress
date: 2026-09-04
type: explanation
---

# Role authority, revocation and enforcement

VisionClaw centralises route requirements and persists pubkey-bound roles. These are useful foundations, but transaction boundaries, fallback roles and deployment modes determine the authority actually exercised. The [receipt](evidence/rbac-snapshot.json) records a source trace, not a runtime test.

## Role mutation transaction scope

[RoleStore](../../../project/src/services/role_store.rs) resolves explicit assignments first, then Admin for legacy power users, then the configured default for other authenticated users. Errors resolve to Viewer. Viewer is a reduced permission level, not universal denial: read operations can remain available. Removing an assignment restores this fallback policy; it is not account revocation and can increase a previously restricted user's effective authority.

The checked assignment/revocation methods transact target-role lookup, lattice checks, last-Owner count and mutation. This prevents interleaving those target checks within that transaction. However, the [handlers](../../../project/src/handlers/admin_rbac_handler.rs) resolve the caller's role beforehand and pass its value into the transaction. The transaction does not re-read the caller's current role. A concurrent caller demotion between resolution and mutation is therefore an acceptance case; the broad no-TOCTOU claim is not established for caller authority. No race was executed in this pass.

Successful changes log after commit. Assigned-by metadata is not an immutable history of every role transition; deletion and a successful log call do not establish a durable, correlated audit receipt. Closeout needs explicit mutation/authority/audit consistency and response-loss recovery, alongside the already implemented last-Owner safeguard.

## Central route policy and execution mode

[The API scope](../../../project/src/main.rs) installs [RbacGate](../../../project/src/middleware/rbac_gate.rs). Whole-segment matching prevents similar names inheriting a prefix. Public prefixes are evaluated before method requirements, so auth, client-logs and health families bypass this gate for all methods. Admin routes require Admin; other writes require WriteSettings under settings or WriteGraph elsewhere. Public safe reads are a separate setting. Every newly added route under a public prefix inherits that exception.

Report mode changes enforcement: a debug build accepts the requested mode directly, while a non-debug build can accept it when the acknowledgement equals today's UTC date at construction. The mode is stored in the middleware; this source does not re-evaluate the date on each request. A dated acknowledgement is thus an activation check, not an automatic expiry. Report mode forwards denied requests and even the missing-NostrService case. These are explicit branches, not proof of current deployment configuration.

The whoami handler describes itself as available to any authenticated caller, but its /api/admin/rbac path encounters the central Admin requirement first in enforce mode. Handler-local requirements alone cannot describe effective access. A complete route contract must account for middleware order and public-prefix exceptions.

## Closeout

CP-01/04/05/08 requires caller authority bound to the mutation transaction or another explicitly chosen revocation boundary; simultaneous caller demotion and target changes; last-Owner contention; error/read-only semantics; and removal-versus-revocation tests. Pair each committed transition with recoverable audit and idempotency evidence. Test all API methods and registered prefixes, whoami access, missing services, report-mode activation/restart/date rollover and release profile assertions. ADR-2010 is partial against its broad atomic-authority guarantee; ADR-2011 retains its central-gate implementation with enforcement modes and exceptions qualified. No production state was inspected or changed.

## Development bypass and release identity

The [actual extracted build matrix](evidence/dev-auth-probe.json) compiles the bypass and boot-hygiene helpers with their original conditional attributes. Nine executions distinguish non-debug without dev-auth, non-debug with dev-auth, and debug builds, each with the dev-mode variable absent, zero or one. Without dev-auth, either present value exits 2; absent succeeds with bypass false. Both development-capable variants accept zero and enable bypass for one. [Reproducer](evidence/dev-auth-probe.py) uses fresh synthetic environments, not real process configuration.

This means `--release` alone does not establish the production security boundary. The current Dockerfile build lines omit dev-auth for production compilation and Cargo defaults omit it, but that source is not an attestation of the shipped binary or its feature closure. ADR-2037 still proposes the image/binary assertion. ADR-2039 must not cite it as an already guaranteed CI control.

In the full source, verify_access checks the full bypass before normal role resolution, returning its sentinel identity for any required level. The WebSocket path invokes it on the authenticate message; it is not automatic authorisation on socket connection. Neither branch restricts peers when this full dev bypass is armed. This is separate from the loopback-gated dev-session token. Source and helper evidence do not certify headset writes, complete provenance handling of the sentinel or network confinement.

The named-profile assertion in ADR-2038 remains a separate proposed control. Refusing selected development variables does not validate the combined public-read, visibility, default-role and report-mode posture. CP-01/04/06/08 requires shipped artefact/feature identity, pre-listener failure tests, dev-versus-production promotion controls and full REST/WebSocket journeys. Include zero-valued forbidden variables, dev-auth release builds, report-mode combinations, proxy/SNAT paths and sentinel attribution. No full image or service was built or started in this pass.

## Profile claims and effective policy

The related ADR-2012/2026/2027 records now use the same evidence boundary as the release proposals. ADR-2012 is partial: its earlier title claimed release unreachability for report mode, and its consequences claimed midnight expiry. Source instead permits current-date activation in non-debug builds and caches the mode at construction. The literal dev token uses debug-or-dev-auth compilation, runtime opt-in and observed-loopback checks; it is not the peer-agnostic full bypass.

ADR-2026 retains its scoped control evidence, qualified to non-debug builds without dev-auth. NODE_ENV=development triggers that paired hygiene check only when DOCKER_ENV is also present. Role lookup failure produces Viewer, and unknown authenticated users can receive the Editor default. These mechanisms cannot establish a universal no-authority-gain theorem for all absent settings.

ADR-2027's four compose defaults are public reads on, ownerless boot allowed, Editor default and visibility filtering on. Their named profile is not an attestation of the running process. Profile acceptance must additionally account for report mode, feature set, bypasses, public prefixes and power-user fallback. [Hash revalidation](evidence/security-profile-snapshot.json) confirms the previous nine helper cases still reference current source; no new deployment test is implied. CP-01/04/08 needs one effective-policy matrix and a release-bound pre-listener check rather than treating each document's scoped check as whole-system proof.

## Request realms and deferred delegation

The [current source and test receipt](evidence/auth-realms-snapshot.json) reaches ADR-2009's browser migration review trigger. The API interceptor now signs NIP-98 in ordinary authenticated mode; settings endpoints and the LDP client also call signRequest. A search of current client/src finds no X-Nostr-Token header use. Eleven existing interceptor tests pass with a mocked signer. This establishes current source behaviour and header construction, not deployed browser coverage or cryptographic acceptance.

The server still accepts X-Nostr-Pubkey plus X-Nostr-Token. The Nostr-prefixed branch returns an error on validation failure before reaching that fallback. validate_session compares token equality and age against the user's mutable last_seen; it does not update the timestamp itself. Other service operations, including NIP-98 user retrieval, update last_seen. Consequently, the expiry description must identify activity semantics, rather than imply an absolute lifetime from token issuance. No session lifecycle or restart test ran in this pass.

ADR-2009 retains complete/live for the scoped coexistence implementation. Its former claim that the current interceptor requires legacy headers is superseded by this review. Retirement still needs a census of browser, socket, agent and external consumers; named compatibility policy; refresh/logout/revocation and persisted-session receipts; and a tested rollback. The prior body-binding and role/profile requirements also apply. Signing a client request does not establish that every server route checks its body.

ADR-2013 remains none/inactive for enterprise federation and delegated-user authority. Its former sole-realm title conflated a public-key identity foundation with request credentials. In the inspected NIP-98 path, authority comes from the verified event pubkey. The bridge verifies the incoming signature, then signs a new event with its own key and a source_event reference. That reference preserves correlation; it does not delegate the source user's permissions. Client delegation fixtures and agentbox's separate inbox policy do not implement delegation in this VisionClaw path.

CP-01/04/05/08 requires an explicit decision to keep federation deferred or activate a specified issuer/delegation contract. Before activation, define principal mapping, audience, permitted operations, grant expiry, revocation, key custody, audit correlation and restart behaviour. Exercise the complete signed grant → authorised mutation → receipt → revocation → denied retry journey. Source searches and these client mocks do not certify enterprise SSO absence throughout the estate or acceptance of a proposed delegation design.

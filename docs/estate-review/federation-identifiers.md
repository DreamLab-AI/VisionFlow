---
title: Federation identifier parity and crossing coverage
status: source-and-cross-language-probe-verified
date: 2026-09-04
type: explanation
---

# Federation identifier parity and crossing coverage

Cross-repository identity needs agreement on bytes, grammar and supported translations. The current agentbox and VisionClaw implementations agree on the tested hash function, while their supported kinds and precomputed-address validation differ. A shared name or copied comment cannot establish all three contracts.

The [reproducible fixture](evidence/federation-identity-probe.py) compiles the actual VisionClaw URI module in a temporary offline Rust crate and calls the actual agentbox JavaScript bridge. [Results](evidence/federation-identity-probe.json) include source hashes and invented inputs. No server, relay, persistent mapping or deployed ingest path ran.

## Hash parity is over the same bytes

VisionClaw content_address hashes supplied bytes; the agentbox BC20 sha12 helper hashes a string as UTF-8. Both prepend sha256-12- to the first twelve lowercase digest hex characters. Empty text, ASCII, two Unicode encodings and a JSON text fixture agree in this probe.

The composed and decomposed Unicode strings have different byte sequences and therefore different addresses in both implementations. Neither normalises them in this boundary. Likewise, agentbox uris.js stable-serialises structured payloads before hashing, whereas the BC20 bridge hashes the incoming URN string. These are different input contracts sharing a digest format. The fixture does not prove arbitrary-object canonicalisation parity or collision handling.

## Translation coverage differs

| Invented agentbox kind | JavaScript bridge | Rust crossing |
|---|---|---|
| agent with valid owner scope | DID | Same DID |
| activity with valid scope/local | Execution address | Same address |
| thing with valid scope/local | KG address | Same address |
| bead with valid content address | Bead address | None |
| memory without elevation options | None | None |

Agentbox additionally supports memory elevation when explicit domain/slug options are supplied; Rust's ordinary crossing deliberately returns None for memory. That source distinction was not exercised as an elevation journey in this fixture. Rust also has a structurally validated already-converged DID path. None should remain a visible unmapped result, not a fabricated identity.

The bead difference is an interoperability decision to resolve, not evidence that the Rust closed map itself violates its stated list. Any supported-kind extension needs matching consumer semantics and migration evidence. In particular, preserving a mapping record must be tested through real persistence and round-trip use; these pure helper calls establish neither.

## Precomputed addresses are checked only by prefix

The actual Rust kg_with_address constructor and KG parser accept the valid fixture plus an empty suffix, a non-hex suffix and an overlong uppercase suffix, provided they start with sha256-12-. They validate the owner scope separately. This is weaker than validating the advertised twelve-lowercase-hex grammar.

The hash-producing function emits the intended shape; the weakness is accepting precomputed input. The helper fixture does not establish that an exposed route accepts those strings or that malformed identifiers have entered deployed storage. Trace validation at each caller before making that claim.

## Closeout requirements

CP-01/02/04/05 require one versioned two-language fixture set defining input byte encoding, serialisation, exact grammar, supported kinds, elevation and unmapped behaviour. Test valid and malformed precomputed addresses, owner scope, duplicates and mapping recovery. Run the fixture in both repositories' CI before changing a shared primitive. The local review fixture is available, but is not wired as that CI gate.

Agentbox ADR-2025 remains proposed; its implementation is partial because helper primitives exist and now have paired local evidence, while the cross-repository acceptance gate is absent. VisionClaw ADR-2023/2025 retain scoped hash/map decisions with explicit limits. The new agentbox protocol governing document fills a missing reference without retroactively ratifying the proposed contract.

## Mint-site coverage and proof of identity

The earlier paired helper evidence still matches all three inspected source hashes. That confirms its continued relevance to helper grammar and crossing behaviour, not universal use of the helper. A current [mint-site review](evidence/identifier-mint-sites.json) finds ontology_mutation_service constructing the persisted provenance activity string with format! directly, alongside a formatted did:nostr agent identifier. The graph repository still formats urn:ngm node and edge identifiers. Rejecting legacy input in parse() does not prevent those other functions minting legacy identifiers.

ADR-2021 is therefore partial against its universal typed-only/no-new-legacy claim. This does not authorise rewriting named graphs or persisted identifiers. The repository needs a classified inventory of new minting, legacy-compatible writes, parsing, lookup and display sites, with explicit exceptions and a separately accepted migration contract. The source finding identifies bypass of constructor validation; it does not assert that every generated value is malformed.

The DID constructor validates canonical string shape. The Nostr identity verifier separately checks a Schnorr signature using the raw claimed public key, and verify_did_matches_challenge parses the payload DID then compares its public key with the verified result. This is the right distinction between naming and proof of control; it does not grant an application role or prove all routes use the same challenge policy. Freshness, challenge reuse, audience/session binding and downstream authorisation need their own end-to-end receipts.

CP-01/04/05/08 requires constructor adoption/approved exceptions at every durable mint site, grammar and old-ID lookup fixtures, idempotent identity joins and rollback before any legacy retirement. Keep names, signed proof, admission and persisted ownership separate. ADR-2022 retains its scoped canonical identity decision with full persistence/caller coverage still open. No graph rewrite, signature challenge or live federation request ran.

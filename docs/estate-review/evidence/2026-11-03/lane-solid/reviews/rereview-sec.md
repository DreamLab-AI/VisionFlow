I reviewed `653b0fd..HEAD` by reading the code only; nothing was built or run. Two of the fixes fall short: the N3 `where` bound still leaves a denial-of-service path, and the new Read gate covers N3 but not JSON Patch. One new defect comes with them: the binding cap refuses some legitimate patches. The other fixes look correct, and I found nothing that weakens main's 3 Nov repairs: malformed-ACL denial, condition preservation, payment read-failure denial, did:nostr provenance and the ACL lockout guard.

## Findings

**1. HIGH — CONFIRMED (code); size of the effect estimated. `where` evaluation still allows a CPU and memory DoS while holding the write lock.**
- **Where:** `crates/solid-pod-rs/src/n3_patch.rs:328`.
- **Cause:** `let mut nb = b.clone()` runs for every pattern×triple step, before the match is tested. `Bindings` is `Vec<(String, Term)>`, so each step deep-copies every bound value. `MAX_MATCH_STEPS` counts steps but not their cost, and `apply` runs inline on a tokio worker (there is no `spawn_blocking` in `lib.rs`).
- **Scenario:**
  - Any account holder with Read+Write on a resource in their own pod stores a document with one large literal and about 20k small triples.
  - The default body cap is 50 MiB (`lib.rs:483`), and see #4: the 4 MiB Turtle cap does not apply on the fallback path.
  - They PATCH with `where { <s0> <p0> ?big . <s1> <p1> ?w1 . … <s31> <p31> ?w31 }`. Each pattern matches exactly one triple, so the partial set stays at 1 and the 1,000-binding cap never fires.
  - That is about 640k steps, under the 4M budget, but each step copies the multi-MiB literal: terabytes of memcpy.
  - A handful of concurrent requests pins every worker. Each one also holds a stripe lock, so unrelated writers that share the stripe block as well.
- **Why the tests miss it:** `an_explosive_where_clause_is_refused_promptly` only uses small literals and all-variable patterns.
- **Fix:**
  - Match against the borrowed binding and clone only on success. Better, bind by reference into the graph (`Vec<(&str, &Term)>`), which makes a step O(1).
  - Also run `apply` (and the seed parse) under `spawn_blocking`.

**2. MEDIUM — CONFIRMED. `MAX_PARTIAL_BINDINGS` refuses legitimate unique matches.**
- **Where:** `n3_patch.rs:32` and `:337`.
- **Cause:** the comment says a clause exceeding the cap "cannot match exactly once". That is false, because the cap counts intermediate bindings before later patterns narrow them.
- **Scenario:**
  - `where { ?t a ex:Task . ?t ex:id "42" }` on a document with 1,001 tasks returns 409, even though the match is unique.
  - Swapping the two patterns makes it succeed, so the outcome depends on the order the client wrote the patterns in.
  - `MAX_MATCH_STEPS` has the same problem: about 400 partials over 10k triples hits 4M.
- **Fix:** reorder patterns most-bound-first (constants and already-bound variables) before joining, or index the graph by subject/predicate. Correct the comment and the CHANGELOG wording either way.

**3. MEDIUM — CONFIRMED. JSON Patch is still a content oracle for a write-only agent.**
- **Where:** `lib.rs:2041` (the Read gate applies to N3 only) and `ldp.rs:1970–2051`.
- **Cause:** `test`, `remove`, and `copy`/`move` with a missing `from` all return `PreconditionFailed` → 412.
- **Scenario:** an agent with Write but no Read sends `[{"op":"test","path":"/secret","value":"guess"}]` to a JSON resource. A 412 means the guess is wrong; a 204 means it is right. Repeating that enumerates the content.
- **Fix:** require Read (with `charge=false`) when any op is `test`, `remove`, `copy`, `move` or `replace`, or gate every JSON Patch on Read.
- **SPARQL is fine to defer:** `apply_sparql_patch` silently ignores absent `DELETE DATA` triples and does not evaluate `WHERE` (`ldp.rs:1369–1400`), so every SPARQL patch answers 204 and there is no observable oracle.

**4. MEDIUM — CONFIRMED. The seed fallback bypasses the size cap and the IRI-space fix.**
- **Where:** `lib.rs:2357`.
- **Cause:** `parse_turtle(..).or_else(|_| parse_ntriples(..))` also falls back when the error is `PayloadTooLarge`. `MAX_RDF_INPUT_BYTES` is 4 MiB, but stored bodies can reach 50 MiB.
- **Effects:**
  - Bodies over 4 MiB go to the uncapped lenient reader, which makes #1 worse.
  - Those bodies are also read in the old IRI space: relative IRIs stay unresolved and deletes 409. That fails closed, but it silently reverts the IRI fix above 4 MiB.
- **Fix:** fall back only on a syntax error. On `PayloadTooLarge`, refuse with 413, or raise the cap consistently.

**5. MEDIUM — CONFIRMED. A refused PATCH has already been charged for its Write.**
- **Where:** `lib.rs:2003` (`enforce_write_ctx` debits any `PaymentCondition`) runs before `lib.rs:2042` (the Read gate).
- **Scenario:** a resource has a paid Write rule and a paid or absent Read. A `deletes`/`where` PATCH debits the ledger, then returns 403. The CHANGELOG's line "a paid Read refuses the PATCH instead" is true, but the caller has paid by then.
- **Fix:** parse the patch and run the Read check before `enforce_write_ctx`, or make the Write debit the last step before `put`.

**6. LOW — SUSPECTED. The Host header can become persisted IRIs.**
- **Where:** `lib.rs:2030` and `:2357`.
- **Cause:** with `public_origin` unset, `request_origin` comes from Host. The Turtle seed now resolves every relative IRI in a non-ACL document against it and writes them back as absolute N-Triples.
- **Scenario:** a writer who PATCHes via an alternate hostname (localhost, a LAN IP, or a forged Host) permanently rewrites relative IRIs in, for example, a group document's `vcard:hasMember </bob/profile/card#me>` to the wrong host. Bob then loses group-based access, which fails closed.
- **Fix:** require `public_origin` when serving, or keep relative IRIs relative on write-back for non-ACL documents, as `acl_targets_as_paths` already does for ACL targets.

**7. LOW — CONFIRMED. The mint amplification fix is partial.**
- **Where:** `lib.rs:265–279` and `:1872`.
- **Effect:** the repeat-candidate break reduces the worst case from 64×10k to 2×10k `exists()` probes per POST, once 10k siblings exist. Each POST still costs 10k sequential storage probes.
- **Fix:** cap the linear probe (say at 100) and then use a random suffix.

**8. INFO — CONFIRMED. N-Triples leniency.**
- **Where:** `ldp.rs:842` combined with the new `strip_prefix('.')`.
- **Effect:** the reader now also accepts `<a> <b> <c> . .` and lines with no terminating dot. No triple is dropped, so there is no security impact.

## Checked and found correct
- **Dropped dedup:** removing the deduplication in `solutions` is sound, because `Graph` is a `BTreeSet`.
- **Early exit:** stopping at two solutions on the last pattern is correct.
- **`write_acl` lock:** the key matches the one LDP uses for the `.acl`. The lockout guard reads only the proposed document, so there is no window between check and write.
- **MCP container mint:** keys with the trailing slash, matching LDP.
- **`charge=false` path:** a sidecar Control elevation is never charged.
- **Provisioning order:**
  - The root ACL's public rule has no `acl:default`, and the owner rule does.
  - So if provisioning fails part-way, the children inherit owner-only access, never public.
  - The ACL lockout guard and malformed-ACL denial paths are untouched.

## The "deliberately not done" items
- **Releasing the lock before `git_mark_write`:** agree it should stay held, because of the provenance-to-authority binding.
- **Host-derived WebID on `/.pods`:** defer only if `public_origin` is mandatory in deployment. Otherwise it should be refused at startup; the same root cause drives #6.
- **422 vs 400/409:** agree; cosmetic.
- **`#[non_exhaustive]`:** agree; an API-policy choice.
- **Preconditions on container PUT/COPY/MCP:** agree; low risk.
- **Predictable stripe hasher: disagree.** `DefaultHasher::new()` uses fixed keys, so an attacker can find a path they own that shares a victim's stripe. Together with #1, that blocks writes to the victim's resource. A `RandomState` built once at startup is a one-line fix.
- **Same-origin ACL targets rewritten to path form:** agree; only a Control holder can trigger it, and it has no authorisation effect.

`★ Insight ─────────────────────────────────────`
- An operation budget only bounds cost if each operation costs the same. Here a "step" includes cloning bindings whose size the attacker controls, so the step cap limits the count but not the work.
- A cap on intermediate join results is not a cap on answers. Whether a nested-loop join hits the limit depends on pattern order, which the client chooses; reordering patterns most-bound-first is what makes such a limit both safe and correct.
- §5.3.1's "inspect needs Read" rule applies to any dialect whose response differs with the stored content. SPARQL is harmless here only because this engine ignores `WHERE` and absent deletes, while JSON Patch `test` exists precisely to be an oracle.
`─────────────────────────────────────────────────`
EXIT 0

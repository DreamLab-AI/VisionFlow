Merge verdict: **don't merge yet**. Three high-severity defects come in with this branch's new N3 Patch path and Turtle seeding, and none of the current tests catch them. I found no deadlocks: every handler holds at most one stripe, and no PAYMENT_STATE_LOCK or trail-lock holder ever takes a stripe.

## High

**1. CONFIRMED — deletes and `where` stop matching stored relative or path-form IRIs.**
- Where: `crates/solid-pod-rs-server/src/lib.rs:2328` (seed) and `crates/solid-pod-rs-server/src/lib.rs:2013` (`N3Patch::parse` with base).
- Cause: the seed reads stored bodies with the lenient N-Triples reader first. That reader keeps `<#a>` and `</alice/public/>` exactly as written. The patch, however, is resolved against `base_iri`, so its IRIs become absolute (`http://host/doc.ttl#a`, `http://host/alice/public/`).
- Scenario A: a client PUTs `<#a> <#b> "v1" .` as `text/turtle`; this is the shape used by `stale_if_match_and_if_none_match_star_are_refused`. A later PATCH with `solid:deletes { <#a> <#b> "v1" }` gets 409 "triple to delete not found". An insert puts an absolute duplicate next to the relative one.
- Scenario B: every body written by main's old N3 PATCH stored `<#x>` relative, so those bodies hit the same failure. Main kept both sides relative and matched them.
- Scenario C (ACLs): the first PATCH of `/alice/public.acl` stores `acl:accessTo </alice/public/>`, made path-form by `acl_targets_as_paths`. On a second PATCH, a `deletes` or `where` on `acl:accessTo </alice/public/>` resolves to `http://host/alice/public/` and gets 409. Only inserts still work.
- Fix: put stored bodies and the patch in one IRI space before `apply`.
  - Either resolve relative IRIs in the leniently parsed seed against `base_iri`, or always parse the seed as Turtle with that base.
  - For ACLs, also run the patch's `deletes` and `where` terms through the same origin-to-path rewrite before `apply`. Then run `acl_targets_as_paths` after it, as now.
- Tests to add: deleting a triple that was PUT with relative IRIs, and two sequential ACL PATCHes that delete an `accessTo`.

**2. CONFIRMED — after any RDF PATCH of an ACL, conditional grants silently stop granting.**
- Where: `crates/solid-pod-rs-server/src/lib.rs:2095-2131`, `crates/solid-pod-rs/src/wac/parser.rs:433-441`, `crates/solid-pod-rs/src/wac/conditions.rs:346`.
- Cause: before this branch, an ACL PATCH could never succeed (409). Now a Turtle ACL is seeded through the Turtle fallback and written back as N-Triples. That turns `acl:condition [ a acl:PaymentCondition ; acl:costSats 30 ]` into `acl:condition _:bN` plus separate `_:bN …` statements.
- The WAC Turtle parser has no lookup for blank-node conditions. `_:bN` falls into the IRI-reference branch and becomes `Condition::Unknown { type_iri: "_:bN" }`, which evaluates to `NotApplicable`, i.e. deny.
- Scenario: an owner PATCHes a paid-read ACL to add one friend. The PATCH answers 204, the owner's unconditional Control rule passes the lockout guard, and the payment rule never admits a payer again. No 422 is raised, because PATCH never calls `validate_for_write`.
- Fix (either):
  - Teach `parse_turtle_condition_objects` to resolve `_:label` against that blank node's statements, which means passing the merged subject map down.
  - Or refuse with 409/422 any ACL PATCH whose pre-patch document has conditions that the post-patch document no longer parses to the same `Condition` set.
- Test to add: PATCH an ACL carrying a `ClientCondition` or `PaymentCondition`, then check that the condition still evaluates.

**3. CONFIRMED — an authorised writer can exhaust memory/CPU, and the write lock is held throughout.**
- Where: `crates/solid-pod-rs/src/n3_patch.rs:284-310`.
- Cause: `solutions()` is an unbounded nested-loop join. Each pattern multiplies the partial bindings by the graph size (each binding cloned), so the cost is n^k. It then sorts all bindings by `format!("{x:?}")`.
- Scenario (open registrations): register a pod, PUT 2,000 triples, then PATCH with `solid:where { ?a ?b ?c . ?d ?e ?f . ?g ?h ?i }`.
  - That is 8×10⁹ bindings, so the process runs out of memory.
  - Before that, the actix worker spins synchronously while holding the stripe lock.
  - `DefaultHasher::new()` uses fixed keys, so an attacker can also choose a path that shares a victim's stripe and stall the victim's writes.
- Fix:
  - Cap the number of `where` patterns and the number of intermediate bindings. Fail with 409/422 once `partial.len()` exceeds a bound, or once there are 2 or more solutions, since only exactly one solution is valid anyway.
  - Index the graph by bound terms instead of scanning every triple.
  - Drop the `Debug`-string sort.
  - Optionally use a `RandomState` hasher for the stripe index.

## Medium

**4. CONFIRMED — MCP `write_acl` writes `.acl` without the stripe lock.**
- Where: `crates/solid-pod-rs-server/src/mcp/tools.rs:908-916`.
- Scenario: an owner revokes a grant through MCP `write_acl` while an LDP PATCH of the same `.acl` is between its read and its write. The PATCH writes back the old graph and the revocation is lost.
- This contradicts the "Every write path takes it" rustdoc at `crates/solid-pod-rs-server/src/lib.rs:228` and the CHANGELOG entry.
- Fix: take `crate::lock_resource(&acl_path)` before `reserve_quota_for_size`. Also make the rustdoc name exactly which writers are covered: provisioning writes and git-mark sidecars are not under their own keys.

**5. CONFIRMED (lock is held) / SUSPECTED (impact) — the write stripe stays held across `git_mark_write`.**
- Where: `crates/solid-pod-rs-server/src/lib.rs:1828, 1951, 2089, 2134, 2187`.
- On git-backed pods, `git_mark_write` does a git commit, takes the `EPOCH_PATH` trail lock and may call `anchorer.anchor` over the network (`crates/solid-pod-rs-server/src/handlers/prov.rs:264, 312`).
- Every write to any path hashing to the same 1-in-256 stripe waits for that on-chain anchor.
- No deadlock: the lock order is stripe → trail and PAYMENT → trail, and no trail or PAYMENT holder ever takes a stripe.
- Fix: drop `_write_guard` explicitly before `git_mark_write`. It runs after the write and is best-effort.

**6. CONFIRMED — `patching_away_the_owners_control_is_refused` does not prove the lockout guard fires.**
- Where: `crates/solid-pod-rs-server/tests/named_pod_owner_access.rs:359`.
- The test asserts only "409 and unchanged". But 409 is also what you get from `PodError::Conflict` "triple to delete not found" (for example from the IRI mismatch in finding 1), and on main every Turtle-seeded ACL PATCH already got 409. So the test passes with or without the guard.
- Fix: also assert the body contains "would not grant Control". Add a positive control: first apply a PATCH that removes a different mode and returns 204, which proves the delete actually matches.

**7. CONFIRMED — public API behaviour changes beyond what the CHANGELOG records.**
- Where: `crates/solid-pod-rs/src/ldp.rs:1195, 2185`, `crates/solid-pod-rs/src/error.rs:9`, `crates/solid-pod-rs/src/n3_patch.rs:67`.
- `apply_n3_patch` and `apply_patch_to_absent(PatchDialect::N3, …)` now refuse bodies without `@prefix solid:`, bare `insert {}` keywords, and any relative IRI. `apply_patch_to_absent` has no base parameter, so a library caller has no migration path. Two existing tests had to be edited to keep passing.
- `PodError` gains `Conflict` without `#[non_exhaustive]` (this is documented). `N3Patch` has public fields and derives `Default`, so adding a field later is also breaking.
- Fix:
  - Mark `PodError` and `N3Patch` `#[non_exhaustive]` now, while the crate is at 0.5.0-alpha.
  - Add `apply_patch_to_absent_with_base`, or route the N3 case through `N3Patch`.
  - List the dropped legacy syntaxes and the relative-IRI refusal in CHANGELOG.

## Low

8. **CONFIRMED — pod provisioning order and atomicity are overstated.** In `crates/solid-pod-rs-server/src/lib.rs:3049-3080`, the `.meta` files are written before the ACL tree, which contradicts the comment at line 3071. A failure mid-way leaves a partial pod, and a retry then gets 409 "already exists". Commit 8c607bc says "atomically". Fix: write the ACLs first, and either roll back on error or reword the claim.
9. **CONFIRMED — stale comments.** The comment at `crates/solid-pod-rs-server/src/lib.rs:2033-2041` and the `seed_graph_from_patch_target` rustdoc at `crates/solid-pod-rs-server/src/lib.rs:2307` still say non-N-Triples bodies are refused. They now fall back to Turtle.
10. **CONFIRMED — MCP container create uses the wrong key.** `create_resource` with `isContainer` (`crates/solid-pod-rs-server/src/mcp/tools.rs:470-487`) mints and locks `base` but creates `{base}/`. LDP container PUT locks with the trailing slash, so the two never serialise, and the mint probes the wrong key. Fix: mint, lock and probe on `{base}/` for containers.
11. **SUSPECTED — oversized N3 Patch answers 500, not 413.** `N3Patch::parse` returns `PayloadTooLarge`, which goes through `patch_parse_err` → `to_actix`. `to_actix` has no arm for it, so it answers 500. This is only reachable if the actix payload limit exceeds 4 MiB. Fix: add a `PayloadTooLarge` arm to `to_actix`.
12. **CONFIRMED — the N-Triples reader now rejects valid N-Triples.** A trailing comment such as `<a> <b> <c> . # note` fails `parse_nt_line`, because rest is `. # note`.
    - Conneg GET (`crates/solid-pod-rs-server/src/lib.rs:2283`) falls back to serving the body verbatim, possibly under a type the client did not ask for.
    - PATCH falls back to Turtle.
    - External callers of the public `Graph::parse_ntriples` get an error.
    - Fix: strip a `#` comment outside IRIs and literals after the object.
13. **CONFIRMED — preconditions are not checked everywhere.** Container PUT (lib.rs:1767), COPY to an existing destination, and the MCP writes skip `check_write_preconditions`. `If-Match` also uses weak comparison, which predates this branch.
14. **CONFIRMED — N3 Patch deviates from the Solid Protocol in several ways.**
    - Syntax errors give 400 and unbound variables give 409, where the spec says 422.
    - Several `solid:inserts` formulae are merged instead of rejected.
    - Blank nodes in `where` or `deletes` can never match, so they always give 409.
    - The `N3Patch` rustdoc (n3_patch.rs:48-50) says an insert-only patch needs only `acl:Append`, but `handle_patch` always requires Write. That is a parity gap with JSS.
15. **SUSPECTED — `POST /.pods` takes the owner WebID from the Host header.** `handle_create_pod` builds the WebID from `request_origin` (lib.rs:3127). Unless a public origin is configured, that comes from the Host header, and it is pasted unescaped into the ACL Turtle (`named_pod_acls`). `/account` uses `nodeinfo.base_url` instead. Fix: use one configured origin for both, and reject IRIs containing `>` or whitespace.
16. **CONFIRMED — new 409 under contention.** More than 64 concurrent POSTs with the same Slug can now get 409 from `mint_and_lock_target`. Once all 10,000 suffixes are taken, the deterministic hash fallback leads to 64 × 10,000 probes before the 409.
17. **CONFIRMED — test gaps.**
    - Concurrency is tested only for PATCH and POST, not for PUT, DELETE, COPY or MCP.
    - The If-Match and If-None-Match tests run sequentially, so they cannot show the check happens under the lock.
    - Nothing tests two sequential ACL PATCHes (finding 1), a conditional ACL PATCH (finding 2), or `If-None-Match: *` on a PATCH that creates the resource.

No critical-severity findings.

`★ Insight ─────────────────────────────────────`
- The central bug pattern is a mismatch in how IRIs are represented. The seed reader keeps stored text as-is (`</alice/>`, `<#a>`), while the patch reader resolves against a base. A set-difference over two IRI spaces silently stops matching. Fix it by choosing one IRI form at every point where triples are compared, not by adjusting individual predicates.
- Writing back as N-Triples is lossless as RDF but not for this codebase's hand-written WAC reader, which only understands inline `[ … ]` blank nodes for conditions. A format that is a "subset of Turtle" is only safe if every consumer can read the whole subset.
- The striped lock holds up structurally: one stripe per handler and no stripe acquired while a payment or trail lock is held. The remaining risk is how long it is held (anchoring, the n^k join), not lock ordering.
`─────────────────────────────────────────────────`
EXIT 0

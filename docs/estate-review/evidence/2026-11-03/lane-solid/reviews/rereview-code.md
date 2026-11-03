Two problems need fixing before merge: a bound on `where` evaluation that refuses valid patches (1), and a JSON Patch way for a write-only agent to learn stored values (2). Neither weakens main's 3 Nov repairs. I reviewed the code only and ran no tests, as instructed.

## Findings

**1. Medium — CONFIRMED. The `where` bounds refuse patches that have exactly one solution, and the rustdoc justifying them is wrong.**
- **Where:** `crates/solid-pod-rs/src/n3_patch.rs:28-35` (constants and their doc) and `:325-337` (the checks).
- **Scenario:** a document holds 2,000 `?s a ex:Person` triples and one `?s ex:id "42"`. `solid:where { ?s a ex:Person . ?s ex:id "42" }` has exactly one solution. The first pattern alone produces 2,000 partial bindings, which exceeds the 1,000 cap, so the patch gets 409 "too broad".
- **Same problem with the step budget:** the join compares every partial binding against every triple. At the 250k-triple input cap, a first pattern with about 17 or more matches already exceeds 4M steps.
- **Wrong claim:** the doc says "a clause broad enough to exceed it cannot match exactly once". That is false, because partial bindings are not final solutions. The CHANGELOG repeats it.
- **Fix:** before joining, sort the patterns so those with the most concrete terms come first, and those sharing variables already bound. Apply the cap to final solutions, or raise it well above the step budget. Correct the doc and the CHANGELOG. Add a test where a selective two-pattern `where` succeeds over a large document.

**2. Medium — CONFIRMED. A write-only agent can still probe stored values through JSON Patch.**
- **Where:** `crates/solid-pod-rs-server/src/lib.rs:2041` gates only the N3 dialect. The JSON Patch branch at `:2088` calls `apply_json_patch` without a Read check.
- **Scenario:** a JSON Patch `{"op":"test","path":"/secret","value":"guess"}` answers 412 on a mismatch and 204 on a match. That works as a value oracle, the same leak §5.3.1 is cited against.
- **SPARQL:** the deferral is harmless today. Deleting an absent triple is a silent no-op and `WHERE` is ignored, so no status reveals anything (`ldp.rs:1369-1385`). The CHANGELOG line "SPARQL Update `DELETE`/`WHERE` is not yet covered" implies an open leak where there is none. Reword it, and name JSON Patch instead.
- **Fix:** require Read (with `charge=false`) for any JSON Patch containing `test`, `remove`, `move` or `copy`, or simply for every JSON Patch.

**3. Low–Medium — SUSPECTED. Reading the stored body as Turtle first makes every PATCH permanently absolutise relative IRIs.**
- **Where:** `lib.rs:2357`.
- **What happens:** every PATCH, including insert-only ones, resolves every relative IRI against `request_origin`. Without a configured `public_origin`, that is the client's `Host` header. The result is written back as absolute N-Triples.
- **Scenario:** a client PATCHes through `localhost:3000`. From then on the document contains `http://localhost:3000/doc.ttl#a`, and a later `<#a>` sent through the public hostname no longer matches.
- **ACLs:** `acl_targets_as_paths` restores only `accessTo`/`default`. A relative `acl:agentGroup`, `acl:origin`, or the subject IRIs get frozen to whichever host the PATCH arrived on.
- **Fix:** when writing back, turn same-origin IRIs back into document-relative or path form, or use only the configured public origin as the base.

**4. Low — CONFIRMED. The fallback reader silently brings back the IRI mismatch on large bodies.**
- **Where:** `lib.rs:2357-2358`.
- **What happens:** `parse_turtle` refuses bodies over 4 MiB or 250k triples with `PayloadTooLarge`. `or_else` then hands the body to the lenient N-Triples reader, which has no triple bound and leaves relative IRIs unresolved.
- **Scenario:** above the cap, `<#a>` in an insert adds an absolute twin of a stored relative triple, and deletes fail to match.
- **Fix:** fall back only on a syntax error (`BadRequest`) and pass `PayloadTooLarge` through as 413.

**5. Low — CONFIRMED. Patching a conditional ACL fails closed, but tells the caller the wrong reason, and the test doesn't check the reason.**
- **Fail-closed holds:** the WAC parser reads `acl:condition _:bN` as `Condition::Unknown` (`wac/parser.rs:432-441`), and `validate_acl_document` rejects that inside the lockout guard (`lib.rs:994`).
- **Wrong message:** the reply is "the patched ACL would not grant Control … use an absolute WebID", which sends the user after the wrong cause.
- **Test gap:** `tests/patch_iri_space.rs:176` asserts only the status.
- **Fix:** have the guard return why it refused, and give conditional ACLs their own message ("ACL carries `acl:condition`; PUT the whole document"). Pin that message in the test.

**6. Low — CONFIRMED. The join copies the binding set on every comparison.**
- **Where:** `n3_patch.rs:328`. The partial binding (up to 96 entries) is cloned before each match is attempted.
- **Cost:** up to 4M clones, all inside the resource's write lock and on a Tokio worker thread, so one request can stall for seconds.
- **Fix:** test the match against the existing binding without changing it, and clone only when it matches. Optionally run `apply` on a blocking thread.

**7. Low — test gaps (CONFIRMED).**
- **Step budget never tested:** both "bounded" tests (`n3_patch_quads.rs` `where_evaluation_is_bounded` and `an_explosive_where_clause_is_refused_promptly`) hit the partial-binding cap on the second pattern. `MAX_MATCH_STEPS` is never exercised.
- **`charge=false` never tested:** there is no test with a paid Read rule plus Write that sends a `where`-patch and asserts 403 with the balance unchanged.

**8. Low — SUSPECTED. A paying agent can be refused even though a free rule would admit them.**
- **Where:** `wac/evaluator.rs:438-444`. The first matching rule in document order sets the cost.
- **Scenario:** an agent has enough balance, and the paid Read rule comes before a free Read rule. Their `where`-patch is refused, although the free rule alone would admit them.
- **Context:** this is the same rule order the charged GET path uses, so it is consistent, but it is not what the CHANGELOG describes.

**9. Nit — CONFIRMED.**
- **Mint error message:** after the early break on a repeated candidate, it still says "after 64 attempts" (`lib.rs:282`).
- **SUSPECTED, pre-existing:** the file and container keys `/c/box` and `/c/box/` are never checked against each other or locked together (`mcp/tools.rs:478`, `lib.rs:1950`). Minting a container can sit beside a file of the same name.

**Checked and not regressed:**
- Malformed-ACL denial still applies on the new Read path.
- Condition preservation is fail-closed (finding 5).
- Payment balance read is unchanged.
- did:nostr-bound export and provenance are untouched.
- The lockout guard is intact, and its test is stronger.
- Holding one stripe at a time still holds: COPY drops the destination guard before taking the sidecar's.
- `PayloadTooLarge` reaches 413 through `patch_parse_err`.
- Writing ACLs first and the `write_acl` lock are correct.

## The "deliberately not done" items

- **Releasing the lock before `git_mark_write`:** agree it stays held. Releasing it would let another request's bytes be committed under this agent's name.
- **Host-derived WebID on `/.pods`:** disagree unless a public origin or allowed-hosts list is enforced. Otherwise a caller-chosen `Host` header picks the owner's WebID.
- **422 vs 400/409:** agree to defer; it is cosmetic.
- **`#[non_exhaustive]`:** disagree. This release already breaks exhaustive matches on `PodError`, so adding it now costs nothing, while adding it later is a second break.
- **Preconditions on container PUT, COPY and MCP:** agree. They are serialised under the lock already; this is only a gap in 412 support.
- **Predictable stripe hasher:** agree. A random path already shares a victim's stripe 1 time in 256, and `where` is now bounded.
- **Untouched ACL targets rewritten to path form:** agree, since only a Control holder can trigger it. But finding 3 is the wider version of the same problem and deserves to be judged separately.

`★ Insight ─────────────────────────────────────`
- An oracle can hide in any patch dialect, not only N3. What matters is whether the response status depends on stored state: JSON Patch `test` does, while this crate's SPARQL `DELETE DATA` does not.
- A cap on intermediate join results is only a valid "no single solution" proof if pattern order is fixed by selectivity. Otherwise it measures how the patterns were ordered, not what the data contains.
- A fallback chain (`a.or_else(b)`) can swallow a resource-limit error and quietly swap in different semantics. Fall back only on the error class you meant to handle.
`─────────────────────────────────────────────────`
EXIT 0

I found no authorisation bypass, and the payment debit still lands on the same agent as before. There are two Medium defects, both confirmed by tracing the code. I didn't build or run anything, as instructed.

## Authorisation reorder: no bypass found

- **Whether Read is checked depends only on the request.** It is decided by the dialect and the parse result (`lib.rs:2040-2046`), never by stored state, and the parse result is reused rather than redone after authorisation. So the request that is authorised is the request that runs.
- **Skipping Read leaks nothing about content.** Read is skipped only when the dialect is unsupported, the body isn't UTF-8, or the N3 parse fails. All three end in 415/400/413 before storage is read.
- **Charging is unchanged.** `enforce_read_inner(.., false)` never debits and refuses a paid Read. `enforce_write_ctx` is still the only debit, for the same agent and resource. A sidecar is raised to Control on both checks and is never metered.
- **The 3 Nov repairs from main hold.** Owners can still PATCH ACLs (Read on `.acl` also needs Control). The write lock and precondition check still sit after authorisation. Provisioning isn't touched.

## Findings

**1. Medium: the join order can still refuse a unique `where` match.** `n3_patch.rs:390-399`
- **Cause:** patterns are ranked by `fixed_positions` first, and the real match count only breaks ties.
- **Scenario:** on the 20,000-person graph from the test, the clause `?s a <Person> . ?s ?p "4242"` picks `?s a <Person>` first (2 constants, 20,000 matches) over the pattern that matches once (1 constant). That creates 20,000 partial bindings, over the 10,000 cap, so the patch gets 409 "too broad" even though the answer is unique.
- **Gap in tests:** the test only uses patterns with the same number of constants, so the tie-break hides this.
- **Fix:** rank by the counted matches (`remaining[i].1`) first. Use "shares a bound variable" to avoid cartesian products, and treat `fixed_positions` as secondary.
- CONFIRMED (code trace).

**2. Medium: a stored resource over the RDF bound can no longer be PATCHed. This is a regression against main.** `lib.rs:2394-2400` together with `graph_to_turtle` at `lib.rs:2272`
- **Cause:** main seeded the graph with the unbounded `parse_ntriples`, so stored N-Triples of any size could be patched. Now anything over 4 MiB or 250,000 triples gets 413.
- **How a resource gets there:**
  - PUT accepts bodies up to `DEFAULT_BODY_CAP` (50 MiB).
  - PATCH writes N-Triples back with no size check. Expanding Turtle prefixes into full N-Triples IRIs, or a large inserted literal, can push a resource over the bound, after which every later PATCH fails.
- **Status code:** 413 is also wrong here, because the request body is small; the stored resource is what's too big.
- **Fix:**
  - check the serialised output against the bound before `put`, and refuse with 409 or 507;
  - answer 409 (or 507), not 413, when the stored body is over the bound;
  - or keep a bounded N-Triples fallback for oversized bodies the server wrote itself.
- CONFIRMED (code trace).

**3. Low: the commit message and CHANGELOG say a refused PATCH is never billed, which is untrue.** CHANGELOG entry and commit body
- **Scenario:** the Write debit happens before these refusals, so all of them are still billed:
  - 415 (unsupported dialect), 400 (bad UTF-8, N3 or JSON parse error);
  - 412 (precondition);
  - 409 (`where` miss, delete of an absent triple, lockout guard, conditional ACL);
  - 507 (quota).
- **What is actually true:** only a refusal by the Read check is unbilled.
- **Fix:** say exactly that. Or, if the intent is that no refused PATCH is billed, move the debit to after a successful `put`.
- CONFIRMED.

**4. Low: "SPARQL Update … reveals nothing" overstates it.** `lib.rs:2037-2039`
- **Scenario:** a caller with Write but no Read can still learn something about a stored resource through SPARQL Update or an insert-only N3 patch:
  - 409 if it isn't parseable RDF;
  - 413 if it's over the bound (new, from #2);
  - 507 from quota.
- This exposes format and size, not triple values, so it's minor.
- **Fix:** reword the comment and CHANGELOG to say these patches reveal no triple values.
- CONFIRMED.

**5. Low: unauthenticated callers can now make the server do parse work.** `lib.rs:2019-2033`
- **Scenario:** the N3 parse (up to 4 MiB) and a copy of the body via `to_string()` now run before authorisation. Before, they ran only after Write was granted.
- The cost is bounded and grows linearly, so it gives only a small amplification for a 401/403 caller.
- **Fix:** acceptable as it stands. Or classify cheaply first (dialect, then UTF-8, then parse only when the dialect is N3), and skip the `to_string` copy until after authorisation.
- CONFIRMED.

**6. Low: the selectivity pre-scan uses up most of the step budget on large documents.** `n3_patch.rs:375-388`
- **Scenario:** with 32 patterns over 250,000 triples, the pre-scan takes 8M of the 10M steps. After that, any join stage with more than about 8 partial bindings fails with 409, even for a selective clause.
- **Fix:** scan the graph once and count matches for every pattern in that pass, or give the pre-scan its own budget.
- CONFIRMED (arithmetic).

**7. Low: a writer without Read can no longer create a resource with JSON Patch.** `lib.rs:2044`
- **Scenario:** an agent with Write but no Read sends a JSON Patch `add` to an absent path. Main returned 201; now it gets 403.
- Main's 201 revealed nothing about content, since the resource didn't exist.
- **Fix:** I'd accept this, because exempting absent targets would need an existence check before the lock, and that is racy. Record it as an intended change in the CHANGELOG.
- CONFIRMED.

- **Critical / High:** none.
- **No defects found in:** the RandomState stripe key (`LazyLock` and `hash_one` are fine under MSRV 1.89), the shorter `mint_and_lock_target` message (no test matches the old text), and the conditional-ACL refusal (it runs before the lockout guard on both the existing and create paths).

`★ Insight ─────────────────────────────────────`
- In a greedy join, counting "how many positions are fixed" is only a stand-in for how selective a pattern is. Here the code already counts the real matches for each pattern, so sorting by that count first costs nothing extra and avoids the case in #1.
- Moving the Read check ahead of Write is safe here because Read never charges and its decision depends only on the request. Things to check in any reorder like this: does the earlier check have side effects, and can its decision change between the check and execution?
- #2 comes from an asymmetry: the server reads stored bodies with a size limit but writes them back without one. Any limit applied when reading needs the same limit applied when writing, or the server can lock a resource against its own later reads.
`─────────────────────────────────────────────────`
EXIT 0

I found one Medium issue, four Low issues and nothing at High or above. Moving the Read check ahead of the Write check and holding parse errors until after authorisation lets no request skip or weaken either check, and the same party is charged as before. HEAD does not undo any of main's 3 Nov repairs.

Everything below comes from reading the code. Nothing was built or run, because the worktree is read-only.

## Direct answers

**Can the new order skip or weaken a check?** No.
- **Write cannot be skipped.** `enforce_write_ctx` (`lib.rs:2055`) runs on every path before any storage read or write. A Read refusal returns early, so Write is still never skipped on a path that goes on to mutate.
- **Read is skipped only when the body is unusable.** That happens for an N3 body that fails to parse, a non-UTF-8 JSON Patch body, or an unsupported content type. All three end in 400 or 415 before stored content is touched. The parse result depends only on the body and the base IRI, so it reveals nothing about the resource.
- **SPARQL is not a way round the Read gate.** I checked the claim in the code: `DeleteData` and `DeleteInsert` call `graph.remove` without erroring on absent triples, and the `WHERE` clause is never evaluated (`ldp.rs:~1370-1400`). A caller with Write only learns nothing about triple presence.
- **Read and Write are still evaluated independently.** Both resolve their own policy and balance. A sidecar path is still raised to Control for both checks, so a JSON Patch on a `.acl` needs Control twice, not anything weaker.
- **Billing.** The Read check runs with `charge=false` and refuses a paid Read instead of debiting it. Only Write debits, and only when the Write grant was not raised to Control for a sidecar, exactly as before. What this commit fixes is billing a Read refusal; other refusals are still billed (finding 2).

**Did HEAD undo main's 3 Nov repairs?** No. `main` is the merge base (`7718095`), so all of main is already in this branch. HEAD does not touch the code from `c27cdfe` (shared payment store, ACL parser), `72d3f45` (provenance) or `7718095` (a test). The new refusal of conditional ACL patches matches the parser: `wac/parser.rs:269-276` only reads inline `[ … ]` conditions, so the N-Triples write-back really would break them. The new `RandomState` key leaves the "never hold two stripes" rule intact (`lib.rs:3304`, `3343`).

## Findings

**1. Medium: the step budget does not limit the work done in a `where` evaluation. CONFIRMED by code trace, not executed.**
- **Where:** `crates/solid-pod-rs/src/n3_patch.rs:318`, `:323` and `:35-38`.
- **Cause:** `ldp::Term` derives `PartialEq` over `String`s (`ldp.rs:568`), so comparing two terms of equal length that differ only near the end costs time proportional to their length. The variable-name lookup in `extend` (`*k == name.as_str()`, `:323`) has the same cost, and variable names have no length cap. The new comment's claim that "each comparison is constant-time" is false. The commit also raised both caps (1k→10k partial bindings, 4M→10M comparisons) on that basis.
- **Scenario:** a caller with Read and Write on any one resource (for example, their own pod) does the following:
  - stores two triples whose subject IRIs are about 1.9 MiB each, equal in length and differing only in the last byte;
  - sends a `where` clause of 13 all-variable patterns, which builds 2^13 = 8,192 partial bindings;
  - follows it with 19 patterns of the form `?x13 ?pk ?ok`.
- **What happens:** the planner breaks ties by index, so the 13 fresh patterns are chosen first. The ones reusing `?x13` only gain a fixed position after that. This gives about 290k steps, well under 10M, but each step compares about 2 MiB, roughly 600 GiB of byte comparison in total.
- **Impact:** this runs synchronously on a tokio worker while holding the resource's write-lock stripe (there is no `spawn_blocking`). A few such requests on different resources tie up every worker. The pre-HEAD code had the name-comparison cost too, but the commit raised the caps on a false premise.
- **Fix:**
  - Before the join, give every document term and every variable a small integer id in one pass, and compare `u32`s.
  - Cap variable-name length when parsing.
  - Alternatively, charge the budget by bytes compared rather than by comparisons, and run `apply` inside `spawn_blocking`.

**2. Low: most refused PATCHes are still billed, and the commit message and CHANGELOG say they are not. CONFIRMED.**
- **Where:** `lib.rs:2055-2073`, `:2082`, and the seed step at `:2398`.
- **Cause:** the Write debit happens before the 415 response, the non-UTF-8 400, the N3 parse 400, the 412 precondition check, the seed's 409 or 413, the `where`/delete 409, and the lockout and quota refusals.
- **Scenario:** a caller paying per Write sends an N3 Patch whose `where` matches nothing. They are charged and get 409.
- **Regression:** most of these were billed before as well. The new 413 is a new refusal that is billed.
- **Fix:** grant Write without charging, then parse and check Read, then debit immediately before `put`. This also fixes finding 4.

**3. Low: the new 413 is a size oracle and the wrong status, and the "fallback only on syntax errors" claim is inaccurate. CONFIRMED.**
- **Where:** `lib.rs:2398`, and `rdf_input.rs:236`, `:248`.
- **Size oracle:** an insert-only N3 Patch or a SPARQL Update needs Write only. It now gets 413 when the stored body is over 4 MiB or 250k triples, so a caller with Write but no Read learns the resource's size class.
- **Wrong status:** 413 tells the client its own request is too large, which is not true here.
- **Inaccurate claim:** errors other than syntax errors still fall back to the lenient reader. These include `Unsupported` (N3 formulas, `log:implies`) and the `BadRequest` raised for a literal subject or an invalid base IRI.
- **Fix:** answer 409 or 507 with a body about the stored resource. Either reword the claim or match on the syntax-error kind only.

**4. Low: N3 is now parsed before authentication. CONFIRMED.**
- **Where:** `lib.rs:2029-2031`.
- **Scenario:** an anonymous caller can make the server parse up to 4 MiB of N3, and copy the body into a `String` (`:2024`), on every request. This is the first time the parser sees unauthenticated input.
- **Fix:** the reordering in finding 2. Run the Write check without charging first, then parse.

**5. Low: functional regressions. CONFIRMED.**
- **Paid Read blocks JSON Patch (`lib.rs:2044`):** wherever Read is sat-priced, every JSON Patch now gets 403, because the Read check refuses rather than charges, so a client has no way to pay.
- **JSON Patch create needs Read:** creating a resource with JSON Patch on an absent path now also requires Read.
- **Pre-scan uses up the budget (`n3_patch.rs:377-388`):** the pre-scan uses up to 32 × 250k = 8M of the 10M comparisons. On a document at the triple limit, any `where` clause with more than about 20 patterns is refused as "too broad", however selective it is.
- **Fix:**
  - Allow a paid Read to be charged once for the combined PATCH, or document the restriction.
  - Exempt JSON Patch creation of an absent resource from the Read requirement, or document that too.
  - Give the pre-scan a separate budget, or build a predicate index once.

**None at Critical or High.**

`★ Insight ─────────────────────────────────────`
- Borrowing bindings removed the cost of copying values, but not of comparing them. The derived `Term` equality still walks every byte of two equal-length strings. A step budget only bounds work when every step costs the same, which is why giving terms small integer ids is the usual fix in RDF engines.
- The cleanest order for a metered gate is: grant without charging, then parse, then check secondary permissions, then debit right before the side effect. Any refusal after the debit is billed, so the debit belongs at the last point the request can still fail.
- A response about the stored resource (409, 413) given to a caller who lacks Read leaks a little about that resource. The Read requirement in Solid Protocol §5.3.1 exists to close exactly this kind of oracle.
`─────────────────────────────────────────────────`
EXIT 0

Two Medium findings. Both are documentation that no longer matches the code. No correctness defect in the code itself reached Medium.

**1. ADR-2005 still describes the Turtle behaviour this merge removed, and names the rewritten test as evidence for it.** CONFIRMED
- **Where:** `crates/solid-pod-rs/docs/adr/ADR-2005-fail-closed-untrusted-parsing.md:171-181`. This file is not in the diff.
- **Scenario:** The "Residual behaviour" paragraph says `parse_turtle_acl` is lenient. It says a broken body comes back as `Found` with `inherited == false`. It also says `broken_turtle_policy_denies_and_never_inherits` asserts a deny for a named and an arbitrary agent.
  - After this merge, that test asserts `PolicyOutcome::Invalid { reason: Malformed(_) }` and `document().is_none()` (`crates/solid-pod-rs/tests/wac_policy_outcomes.rs:270-290`). It no longer calls `evaluate_access` at all.
  - The test comment at `:255-258` says the ADR-2005 residual gap "is now closed". The ledger record still says it is open and "deferred rather than guessed at".
  - Anyone following the repo's lookup order (baseline, then ledger) reads a security-relevant claim that is now false, backed by a test citation that no longer checks it. I grepped the baseline and found no stale copy, so the drift is in the ledger only.
- **Fix:** Add a progress note to ADR-2005 recording that c27cdfe makes malformed Turtle statements `AclParse` errors. The resolver then settles them as `Invalid`, and the test now pins that. Mark the residual paragraph as superseded rather than silently rewriting it, then regenerate the ADR index.

**2. The `If-Match` contract changed in a public API, but only part of the change is documented.** CONFIRMED
- **Where:** the rustdoc at `crates/solid-pod-rs/src/ldp.rs:1469-1480` (re-exported at `crates/solid-pod-rs/src/lib.rs:227`) and the entry at `CHANGELOG.md:332-335`.
- **Scenario:** Commit 50bf3ce also changed how `*` works. Before, `parse_etag_list` stripped `W/` and the quotes before the `w == "*"` check. So on an existing resource, `If-Match: "*"`, `W/"*"` and `"x", *` all acted as wildcards and returned `Proceed`. Now all three return 412.
  - The CHANGELOG entry mentions only weak tags. It says nothing about the `*` change.
  - The public rustdoc still says only "`If-Match: "etag1", "etag2"` — pass if any matches". It doesn't mention strong comparison, that weak tags are dropped, or that `*` counts only as the whole field value.
  - A downstream caller of `evaluate_preconditions` would see new 412s that neither the doc nor the CHANGELOG explains.
- **Fix:**
  - In the rustdoc's `If-Match` bullets, add that comparison is strong (§13.1.1), so `W/` tags never match, and that `*` is a wildcard only as the entire trimmed value. Note that `If-None-Match` stays weak.
  - Add a CHANGELOG line for the `*` change.

I also checked whether the new ACL-link assertion in `crates/solid-pod-rs/tests/interop_jss.rs:353-365` can fail. Both sides come from the same helper, `acl_sidecar_key`: the expected value, the Link header (`ldp.rs:153`) and the resolver's probe (`wac/resolver.rs:379`). So it guards that the advertised and enforced locations agree, but not the `/container.acl` form itself. Asserting the literal `</container.acl>; rel="acl"` would pin that too. I rated this Low.

`★ Insight ─────────────────────────────────────`
- The change in finding 2 comes from the order of operations. The old parser normalised each entry (stripped `W/` and quotes) *before* the wildcard check, so a quoted `"*"` became `*` and matched. Moving the `*` check onto the raw field value is what makes that 412.
- The `"\"*\""` case in `if_match_uses_strong_comparison` does fail if anyone reintroduces an in-list `w == "*"` check. That's the case that actually protects the fix.
- The payment tests are real concurrency guards only because `YieldingStorage` yields on every `get`. On the plain in-memory backend the two handlers would never interleave, so a missing `PAYMENT_STATE_LOCK` would go unnoticed.
`─────────────────────────────────────────────────`

I found no exploitable defect. None of the test changes stop guarding a security property, and f137d1c opens no bypass. There are four low-severity points.

**What each commit still guards**
- **b57bea4 (broken Turtle ACL).** The test is stricter than before. It now requires `Invalid { policy_path: "/secret.acl", Malformed }`, `!may_inherit()`, no document handed on, and `into_result().is_err()`. The old `evaluate_access` deny loop was dropped, which is fine: with no document there is nothing to evaluate, and the server denies on `is_failure()` (`crates/solid-pod-rs-server/src/lib.rs:1081` and `:1299`).
- **8f69aa1 (advertised ACL = enforced ACL).** The advertised link and the resolver's lookup both come from the same function, `acl_sidecar_key` (`ldp.rs:153`, `ldp.rs:1801`, `wac/resolver.rs:379`). The test asserts the `rel="acl"` link is exactly that value and appears once. Every other JSS link must still match exactly.
- **f137d1c (strong If-Match).** I checked each edge case in `ldp.rs:1490-1551`:
  - A list of only weak tags becomes empty after dropping them, so the request fails with 412 (fail-closed).
  - Mixed lists match only on the strong tag.
  - A bare `*` takes its own branch, unchanged.
  - `W/*` used to be stripped to `*` and matched as a wildcard; it is now dropped, so this is tighter than before.
  - Lowercase `w/"x"` is not treated as weak, and with its quotes trimmed it never equals a real ETag.
  - If-None-Match still uses weak comparison, as before, and is still evaluated after If-Match passes.
  - The server evaluates both headers under the write lock (`lib.rs:289-314`).
- **055a47a (owner-only export).** The owner check runs before `GitHttpService::new` reads the env var for each request (`lib.rs:5150-5160`). The non-owner test (`public_root_read_does_not_authorise_whole_history_export`) still asserts 403.

**Findings**

1. **Low: the payment test can't catch a concurrent double-spend.** `crates/solid-pod-rs-server/tests/payment_atomicity.rs:91`. CONFIRMED.
   - **Scenario:** the seller is funded with 1,000 tbtc3 for two concurrent 10-unit sells, and the test only checks that both offers persist (`:135`). If the "only list what you hold unlisted" check (c27cdfe) were read-then-write rather than atomic, two concurrent sells against a balance of 10 would both succeed, and no test would notice. My grep of the pay test suites found only single-request balance checks (`pay_phase4_routes.rs:302`, `:441`).
   - **Note:** this is not a regression, since the test failed before this branch and guarded nothing. But "payment lock atomicity" is currently guarded only against lost offer updates, not against overselling.
   - **Fix:** add a companion test that funds exactly 10 and sends two concurrent 10-unit sells. Assert exactly one 200 and one rejection, and that the remaining unlisted balance is 0.

2. **Low: `set_var` in a multi-threaded test binary.** `crates/solid-pod-rs-server/tests/git_wac_gating.rs:48`. SUSPECTED; it only matters in theory.
   - **Scenario:** the test harness runs tests in parallel. `set_var` can race with `getenv` in other threads, such as `GitHttpService::new` at `service.rs:233` or `Command::spawn`'s environment copy. That is formally unsound on glibc; the crate is on edition 2021, where `set_var` is still safe to call.
   - **Masking:** I don't think it can hide a failure. Every other test in the binary asserts 401 or 403 before the CGI runs, so a broken gate fails whether or not the backend exists. The leak only changes how such a failure shows up (200 or 404 instead of 500).
   - **Fix:** set the variable once through a `std::sync::Once` / `OnceLock`. Better, if the API allows it, give `AppState` an explicit backend path so no process-global state is touched.

3. **Info: rustdoc now attaches to the wrong function.** `crates/solid-pod-rs-server/tests/git_wac_gating.rs:27-36`. CONFIRMED.
   - **Scenario:** the new function was inserted between `owner_only_acl`'s `///` block and its `fn`. The "Owner-only ACL…" text now becomes the head of `use_installed_git_http_backend`'s doc, and `owner_only_acl` has none.
   - **Fix:** move the new function and its comment above line 27.
   - **Related:** the module header (`:10-15`) still says the suite is "CGI-independent". The owner export test now needs a real backend, so that line should be qualified.

4. **Info, already on main and not introduced by f137d1c: lenient ETag-list parsing.** `crates/solid-pod-rs/src/ldp.rs:1502` and `:1549`. CONFIRMED.
   - **Scenario:** `trim_matches('"')` turns a quoted `"*"` into `*`, and the list path accepts `w == "*"`. So `If-Match: "*"` (an opaque tag that is literally `*`) and `If-Match: "a", *` both act as a wildcard on an existing resource. Unquoted tags are also accepted.
   - **Impact:** this is not a bypass, because a plain `If-Match: *` gives the same result legitimately and absent resources still fail. It is only a departure from RFC 9110.
   - **Fix:** require `DQUOTE etagc* DQUOTE` and drop the in-list `"*"` match.

`★ Insight ─────────────────────────────────────`
- Dropping weak tags is fail-closed here only because of the `.any()` check. An empty list makes `any` false, so a header made entirely of weak tags becomes 412. A design that skipped the check whenever the list came out empty would have been a bypass.
- The ACL interop test protects the "advertised = enforced" property because the Link header and the resolver both call `acl_sidecar_key`. If either one stopped using it, the assertion would break.
- An atomicity test that funds far more than the orders need can only catch lost updates. To catch a race in the funds check, the funding has to be exactly enough for one order.
`─────────────────────────────────────────────────`
EXIT 0

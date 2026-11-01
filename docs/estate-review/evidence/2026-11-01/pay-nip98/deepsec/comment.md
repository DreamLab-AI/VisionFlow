## 🔎 deepsec found 8 findings

🟡 MEDIUM ×3 · 🟣 HIGH BUG ×2 · 🟣 BUG ×3

<sub>scope: `git-diff:working` · run `20261101185817-9debd89a60e10fc7`</sub>

### 🟡 MEDIUM · `src/handlers/pay_handler.rs:L646-L896`

**"Authenticated" == any freshly minted keypair; unbounded orders/pools re-serialised per request**
<sub>slug: `expensive-api-abuse` · confidence: high</sub>

There is no account registration: a valid NIP-98 token can be produced by any self-generated keypair at zero cost, so every 'authentication required' check on these handlers establishes identity but confers no scarcity or confidentiality. Consequences: (1) the NIP-98 per-pubkey replay cap (1000) is trivially bypassed by rotating keys; (2) `/pay/.sell` and `/pay/.pool/liquidity` let an attacker create unbounded orders and pools, and because every request reads and pretty-print-writes the ENTIRE `exchange.json` under a global lock (L383-389), file size grows without bound and each subsequent re…

**Recommendation:** Rate-limit and size-cap per authenticated pubkey (and overall): bound orders/pools per identity, cap exchange.json size, and require a funded balance (deposit) before an identity may create orders/liquidity. Correct the comments so they do not claim confidentiality the design does not provide.

---

### 🟡 MEDIUM · `src/handlers/pay_handler.rs:L404-L418`

**Host-header-derived URL binding + nginx default_server enables cross-service NIP-98 replay**
<sub>slug: `auth-bypass` · confidence: low</sub>

`extract_caller_pubkey` binds the token to `nip98_request_url(req)` (utils/auth.rs L170-189), which takes the host from the attacker-controllable `X-Forwarded-Host`/`Host` header; prod nginx forwards `$http_host`, and `nginx.production.conf` L85-86 is `listen 3001 default_server; server_name _;` — i.e. the backend answers for ANY Host. The `/pay/.swap`, `/pay/.pool/swap`, `/pay/{resource}` paths intentionally mirror the JavaScript Solid Server path scheme. A token a victim signed for a same-path resource on another did:nostr service reachable at the same path (or signed with a relative `u` ta…

**Recommendation:** Pin the expected host to a server-side allowlist (configured canonical origin) instead of trusting forwarded headers, set nginx `server_name` to the real hostnames rather than `_`, and reject relative `u` tags on these money-moving routes (require an absolute URL whose host matches the pinned origin).

---

### 🟡 MEDIUM · `src/handlers/pay_handler.rs:L303-L557`

**Process-local NIP-98 replay cache does not span the multi-process ledger it guards**
<sub>slug: `rate-limit-bypass` · confidence: medium</sub>

`FsPaymentStore` is explicitly designed for multiple processes sharing one ledger dir via `flock` (L193-201, 'blue-green deployments, multiple workers'), but the NIP-98 single-use replay cache (nip98.rs L200-221) is process-local by design (documented as SECURITY-profiles invariant 4). If an operator runs the documented multi-process topology WITHOUT sticky routing, a single signed GET to a gated resource (`pay_resource_handler`) or a signed mutating POST can be replayed against a different worker process within the 60s freshness window, each incurring a separate debit/access. The two compone…

**Recommendation:** Either drop the cross-process flock design (commit to single-process + in-memory replay cache) or move the replay cache to shared storage (e.g. Redis) so single-use enforcement spans every process that shares the ledger. Make the deployment requirement fail-closed, not documentation-only.

---

### 🟣 HIGH BUG · `src/handlers/pay_handler.rs:L260-L628`

**Fail-open ledger/exchange read overwrites all balances on any I/O error**
<sub>slug: `other-fail-open-dataloss` · confidence: high</sub>

`read_ledger` (L260-271) and `read_exchange` (L344-352) return a brand-new empty `WebLedger`/`Exchange` on ANY `Err(_)` from the read — not only `NotFound`, but also `EIO`, `EACCES`, EMFILE (too many open files), etc. `with_exchange_and_ledger` (L383-389) and `debit` then write the resulting state back unconditionally. In `pay_sell_handler` (L615-632) the mutation closure never touches `_ledger` and always returns `Ok`, so a single `/pay/.sell` request issued while the ledger file is transiently unreadable causes the empty ledger to be persisted via tmp+rename — destroying every identity's ba…

**Recommendation:** Distinguish `ErrorKind::NotFound` (legitimately empty → new ledger) from all other I/O/parse errors (propagate as `PaymentError::Store` and abort the mutation without writing). Never write back a state derived from a failed read.

---

### 🟣 HIGH BUG · `src/handlers/pay_handler.rs:L388-L389`

**Non-atomic two-file commit can desync pool reserves from ledger balances**
<sub>slug: `other-nonatomic-commit` · confidence: high</sub>

`with_exchange_and_ledger` (L388-389) persists the exchange file first and the ledger second, as two independent write+rename operations. If the ledger write fails after the exchange write succeeds, the handler returns 400 but the pool reserve changes / order fills are already durably on disk with no corresponding balance movement. A crash between the two renames produces the same divergence. For `.pool/swap`, `.swap` and `.pool/liquidity` this means value can be created or destroyed across the two files (e.g. pool reserves credited while the payer was never debited, or vice versa).

**Recommendation:** Serialise both documents into a single file (or a single journaled record) so the commit is one atomic rename; or write a combined temp, fsync, then rename once. At minimum fsync each file before the paired rename and recover/roll back on partial failure.

---

### 🟣 BUG · `src/handlers/pay_handler.rs:L198-L378`

**Blocking flock held across await points on the tokio worker**
<sub>slug: `other-blocking-in-async` · confidence: medium</sub>

`debit` and `with_exchange_and_ledger` acquire the synchronous `libc::flock` (via `acquire_file_lock`, L255-258/L338-342) and then perform `.await` on `read_ledger`/`write_ledger`/`read_exchange`/`write_exchange` while the `FileLockGuard` is still live (L303-311, L378-389). This blocks a tokio worker thread on a blocking syscall and holds the OS advisory lock across awaits — directly contradicting the L198-201 doc comment that claims 'The in-process mutex prevents the tokio task from holding the file lock across an await point.' Under contention this serialises and can stall the runtime.

**Recommendation:** Perform the locked read-modify-write inside `tokio::task::spawn_blocking` (or use an async advisory-lock strategy), and fix the comment to match actual behaviour.

---

### 🟣 BUG · `src/handlers/pay_handler.rs:L617-L821`

**Trade settlement correctness depends on unreviewable upstream (escrow / currency confusion)**
<sub>slug: `other-upstream-dependency` · confidence: low</sub>

`pay_sell_handler` calls `order_book.create_order` without any local check that the seller holds the sell_currency, and `pay_pool_swap_handler`/`pay_swap_handler` delegate settlement to `solid_pod_rs::trading` (create_order/execute_swap/get_or_create_pool/swap/add_liquidity). That crate (crates.io `solid-pod-rs 0.5.0-alpha.17`) is not vendored in this environment, so two correctness-critical questions cannot be statically confirmed here: (1) whether sell orders / swaps escrow or verify seller funds, and (2) how balances are keyed by currency given `WebLedger::get_balance(did)` takes no curren…

**Recommendation:** Review the pinned solid-pod-rs trading implementation for fund escrow on order creation and for currency-scoped balances; add an integration test that attempts to sell/swap a currency the account does not hold and asserts rejection.

---

### 🟣 BUG · `src/handlers/pay_handler.rs:L721-L712`

**Unvalidated numeric/currency extractor content (fee_bps, zero amounts, empty query currencies)**
<sub>slug: `other-input-validation` · confidence: high</sub>

Deserialised request types validate structure, not content. `AddLiquidityRequest.fee_bps: u64` (L721) is unbounded — the first caller to create a pool fixes its fee permanently and may set 10000 bps (100%) or more. `sell_amount`/`price` (SellRequest) and `amount` (PoolSwapRequest) accept 0, allowing zero-value / degenerate orders and swaps. In `pay_pool_swap_handler` the pool currencies come from the query string with `unwrap_or_default()` (L777-778), so a request missing `?a=&b=` silently creates/operates a pool whose currency names are empty strings, corrupting the pool keyspace.

**Recommendation:** Validate before mutating: bound `fee_bps` (e.g. <= 1000), reject zero `amount`/`sell_amount`/`price`, and reject missing/empty `a`/`b` query currencies with 400 rather than defaulting to empty strings.

---

<sub>🤖 generated by [deepsec](https://github.com/vercel/deepsec)</sub>
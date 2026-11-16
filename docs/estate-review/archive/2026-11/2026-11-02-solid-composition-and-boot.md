---
title: Solid Pod workspace composition and server boot
date: 2026-11-02
status: assessed
type: explanation
---

# Solid Pod workspace composition and server boot

I assessed all 23 panels in SP-01 (workspace composition, 12 panels) and SP-02 (server boot and routes, 11 panels) by tracing source at solid-pod-rs `9b6143c`. Results: 3 accurate, 17 corrected, 3 accurate with limits, 0 inaccurate-unfixed. Both topics are now restamped to that revision and pass the strict generator gate.

Most corrections are line re-resolutions: server `lib.rs` moved by 13 to 20 lines between `09ec90f` and the pin. Five corrections change what a panel claims:

- SP-01.13 said CI never compiles the examples. In fact `cargo check --all-targets` runs across the whole build matrix.
- SP-01.8 carried a stale ADR-2010 status.
- SP-02.8 said the error logger observes guard short-circuits. Actix runs middleware in reverse registration order, so that logger is the innermost layer and cannot see them.
- SP-02.5 overstated how the config loader tolerates bad values.
- SP-02.13 left `install` unconnected and missed that the shipped `account delete` can never delete a real user.

## Findings

**SC-01: `account delete` operates on an empty store (Debt).** `dispatch` builds `solid_pod_rs_idp::InMemoryUserStore::new()` for every `account delete` run (`crates/solid-pod-rs-server/src/cli/mod.rs:380`), then calls `store.delete` (`cli/mod.rs:311`).
- Consequence: the shipped binary always prints "no such user". Its help text promises that pods and the WebID profile are removed too (`cli/mod.rs:105`), but nothing on that path touches them.
- To close: dispatch against the server's persistent user store, or remove the subcommand. Add a test that deletes a real seeded user through `dispatch`.

**SC-02: SP-02.8's middleware note and the product comment invert actix order (Drift, corrected in the topic).** `build_app` wraps the layers in this order: `ErrorLoggingMiddleware`, `CorsHeaders`, `NormalizePath`, `PathTraversalGuard`, `DotfileGuard` (`crates/solid-pod-rs-server/src/lib.rs:5211-5218`). Its comment claims the error logger is outermost (`lib.rs:5207-5210`).
- Framework evidence: actix-web 4.14.1 (the version in `Cargo.lock`) documents that middleware is "executed in opposite order as registration" (`actix-web-4.14.1/src/middleware/mod.rs:18`, cargo registry). That makes the error logger innermost.
- First consequence: it logs handler 5xx only (`lib.rs:3708`).
- Second consequence: 403 and 421 rejections from the guards bypass `CorsHeaders`, which adds its headers on the inner service's response (`lib.rs:3568`).
- Third consequence: both guards inspect the path before `NormalizePath` merges `//`.
- Limit: this comes from reading source against the framework docs. No request was run.
- To close: correct the comment. Decide whether guard rejections should carry CORS headers. Add an HTTP test for a 421 response with an `Origin` header.

**SC-03: `--ssl-key` and `--ssl-cert` are silently ignored without `tls` (Tension).** The flags are parsed in every build, but only the `cfg(feature = "tls")` arm reads them (`crates/solid-pod-rs-server/src/main.rs:465-478`). Otherwise the server listens in plaintext (`main.rs:480-483`).
- Consequence: an operator who passes a certificate to a default build gets plain HTTP with no warning.
- To close: warn or exit when either flag is set without `tls`.

**SC-04: the config loader does not collect bad values as warnings (Drift, corrected in SP-02.5).**
- The only warning `load` emits is for memory storage with a root set (`crates/solid-pod-rs/src/config/loader.rs:181`).
- An unparseable boolean env value is dropped with no warning (`config/sources.rs:276-279`).
- An unknown storage type is kept as given (`sources.rs:267-271`), then fails deserialisation as a fatal error (`loader.rs:196`). It is not caught by `validate()`.
- The binary never calls `with_cli_overlay`. It sets host and port by hand after load (`main.rs:290-296`).
- To close: none needed beyond the topic correction. Optionally, warn on unparseable values.

**SC-05: examples are compiled in CI, and the documented oidc stub never builds (SP-01.13, corrected).**
- The `build-test` job runs `cargo check --all-targets` in `crates/solid-pod-rs` (`.github/workflows/ci.yml:62`, `:84`) for the `default`, `oidc` and `all-features` sets (`ci.yml:53-59`). That includes the examples, so the panel's claim that the gate is hand-run only was false.
- `oidc_client` declares `required-features = ["oidc"]` (`crates/solid-pod-rs/Cargo.toml:360`), so cargo skips it without the feature. The stub `main` (`examples/oidc_client.rs:224`) is never built, which contradicts `docs/examples-index.md:39-41`.
- Cross-topic: SP-09.10, owned by w-solid-d, carries the same "never builds examples" claim and should be re-checked against `ci.yml:84`.
- To close: drop `required-features` so the stub compiles, or rewrite the index paragraph.

**SC-06: ADR-2010 status was stale in SP-01.8 (corrected).** The ADR front matter now reads `decision_status: accepted`, `implementation_status: complete`, `activation_status: staged` (ADR-2010 lines 5-7). The panel said "implementation partial, activation inactive".

**SC-07: the server's `install` feature is vestigial (Drift).** The server's core dependency always enables `nip98-schnorr` (`crates/solid-pod-rs-server/Cargo.toml:44`). No source tests `feature = "install"`, and `build_auth_header` calls `nip98::mint` unconditionally (`cli/install.rs:139`).
- Consequence: the manifest comment (`Cargo.toml:174-177`) and the "rebuild with `--features solid-pod-rs-server/install`" hint (`install.rs:141-142`) describe a build that cannot occur.
- To close: remove the feature or make it meaningful.

**SC-08: `acl-origin` and the empty server default (Drift, corrected in SP-01.6 and SP-01.7).**
- SP-01.6 said "a default build ignores the Origin header". That is true only for a default *library* build. The server's core dependency always carries ten features, `acl-origin` among them (`crates/solid-pod-rs-server/Cargo.toml:44`).
- The core manifest comment says the server does not enable `acl-origin` (`crates/solid-pod-rs/Cargo.toml:142-143`), and the server manifest contradicts it.
- SP-01.7 now shows the ten always-on features beside the empty `default = []`.

**SC-09: smaller corrections.**
- SP-01.1: the one-way dependency shape holds for `[dependencies]` only. The core dev-depends on the server (`crates/solid-pod-rs/Cargo.toml:307`), the server depends on the IdP (`server Cargo.toml:48`), and the forge depends on the git crate (`forge Cargo.toml:22`).
- SP-01.3: `bitcoin_tx` sits in the always-compiled list but is internally gated on `mrc20` and not-wasm32 (`src/lib.rs:135-139`).
- SP-01.5: the stub's doc comment says "No-op" while the body returns `Err` (`auth/nip98.rs:379-384`).
- SP-01.11: the licence allow list starts at `deny.toml:74` (not 73), and `allow-git = []` is at `:151`.
- SP-02.7: `base_url` reflects a shifted port only when no base URL is configured (`main.rs:312-316`).
- SP-02.6: rejection of an unknown backend happens at deserialisation (`loader.rs:196`). The S3 removal is confirmed in the 0.5.0-alpha.8 changelog section (`CHANGELOG.md:931-946`).
- SP-02.13: `install` is an `OperatorCommand` variant (`cli/mod.rs:59`), dispatched at `cli/mod.rs:391`. `--yes` bypasses the Prompt (`cli/mod.rs:295`). `quota reconcile` errors in a default build (`cli/mod.rs:190`, `:249-251`).

**SC-10: neither topic has "What it shows / Why it is this way" paragraphs.** Both topics carry their mechanism claims only inside Mermaid note nodes. I assessed those nodes. I did not backfill 23 narratives. I added register markers as standalone lines under the panels where I found something, in the format SP-04, SP-05 and SP-14 use. The missing narrative prose is still an open gap for the corpus owner.

## Panel dispositions

| Panel | Disposition | Evidence |
| --- | --- | --- |
| SP-01.1 | corrected | Eight members and AGPL-3.0-only verified (`Cargo.toml:2-10`, `:17`); every sibling depends on core. Added **Tension** for the core→server dev-dependency (`crates/solid-pod-rs/Cargo.toml:307`) and the sibling edges. |
| SP-01.3 | corrected | All 26 module lines verified (`src/lib.rs:134-209`); `blocktrail` walk gated at `blocktrail.rs:896`. Added **Drift** for `bitcoin_tx`'s inner gate (`lib.rs:135`). |
| SP-01.4 | accurate-with-limits | `core`, `std`, `tokio-runtime` and backend features (`crates/solid-pod-rs/Cargo.toml:121-135`), k256 0.14 (`:73`), wasm getrandom (`:290`), `classify_policy_read` is synchronous (`wac/resolver.rs:265`). Limit: the nostr-rust-forum consumer claim (E1) was not checked in that repository. |
| SP-01.5 | corrected | Stub moved from 376 to 381 (`auth/nip98.rs:381`) and returns `Err`. All feature lines verified. Added **Drift** for the "No-op" comment. |
| SP-01.6 | corrected | Every flag line verified (`Cargo.toml:160-230`). O1 now qualifies that the server always enables `acl-origin` (`server Cargo.toml:44`). Added **Drift**. |
| SP-01.7 | corrected | `git_mark_write` shim moved from 4110 to 4123 (`server lib.rs:4123`) and returns `ProvenanceSkip::NotConfigured`. Added the ten always-on core features. |
| SP-01.8 | corrected | All feature lines verified (`server Cargo.toml:130-178`). The mirror boot condition is confirmed through `forge_service` (`lib.rs:5034-5039`), which calls `mirror_config` (`lib.rs:4978`). ADR-2010 status updated. Added `dev-bearer` (`:133`) and an `install` **Drift**. |
| SP-01.9 | accurate | Forge, git and IdP features verified. activitypub, nostr and didkey have no `[features]` table, so "every sibling default = []" holds. |
| SP-01.10 | accurate-with-limits | Trait methods and types verified (`storage/mod.rs:27-130`). Limit: the VC-26 and ES-08 consumer claims belong to other repositories and were not traced. |
| SP-01.11 | corrected | Citations re-pointed to `deny.toml:74` and `:151`. The DEBT entry matches `deny.toml:57-63`. Added **Open**: whether the cargo-deny gate actually guards release builds is not visible in source. |
| SP-01.12 | accurate | Seven example headers and the index map verified (`examples/*.rs:1`, `docs/examples-index.md:1-30`). |
| SP-01.13 | corrected | Seven `[[example]]` entries verified (`Cargo.toml:333-360`). N rewritten for `required-features`. N3 reversed: CI runs `--all-targets` (`ci.yml:84`). Added **Drift** for `examples-index.md:39`. |
| SP-02.1 | corrected | Boot order verified (`main.rs:261-518`). `build_app` re-resolved to `server lib.rs:5188`. Added **Tension** for the ssl flags being ignored without `tls`. |
| SP-02.2 | accurate | Assembly order and defaults verified (`main.rs:318-451`, `lib.rs:401-424`). |
| SP-02.3 | accurate-with-limits | Every CLI flag is off or None by default (`main.rs:39-173`). Admin returns 403 without a key (`lib.rs:4456-4462`). The stand-in flag is gone (only CHANGELOG and ADR-2008 mention it). Added **Open**: configuration can open registration (`main.rs:350`), and the trust_proxy effect was not exercised. |
| SP-02.5 | corrected | Schema fields and `validate` verified (`schema.rs:27-360`). Note rewritten per SC-04. Added **Drift** for silently dropped booleans. |
| SP-02.6 | corrected | The enum holds only Fs and Memory (`schema.rs:186-195`). The rejection edge now cites `sources.rs:267` and `loader.rs:196`. |
| SP-02.7 | corrected | `bind_available` verified (`main.rs:196-218`). Note split across `main.rs:202`, `:301` and `:312` and qualified for a configured base URL. |
| SP-02.8 | corrected | All wrap lines re-resolved (`lib.rs:5208-5218`, `:3494`, `:3843`). N2 rewritten per SC-02. N4 adds the empty-list rule (`scheme.rs:256`). Added **Drift** and **Tension**. |
| SP-02.9 | corrected | Every route re-resolved by content at the pin (`lib.rs:5229-5315`). Meaning unchanged. |
| SP-02.10 | corrected | Routes re-resolved (`lib.rs:5291-5424`). The `.git` block is outside every feature gate, and push is routed only to be refused (`lib.rs:5361-5370`). |
| SP-02.11 | corrected | Catch-all re-resolved (`lib.rs:5429-5444`). HEAD maps to `handle_get` and COPY uses raw method bytes. Meaning unchanged. |
| SP-02.13 | corrected | `install` connected as an `OperatorCommand` variant. `--yes` noted. Added two **Debt** markers (SC-01 and the quota fallback). |

## Verification

- **Evidence class:** source tracing at the pinned revision only. I ran no product tests, no cargo and no HTTP requests. Nothing here is runtime or deployment evidence.
- **Pinned revision:** solid-pod-rs `9b6143ce55b242c322c92d8c6bcc0c1a5371b451`, read with `git show` and `git grep` against that SHA. Both topics previously declared `09ec90f26aa573731460292d0e80343914cb43d9`.
- **Changed sources:** between `09ec90f` and the pin, these cited files changed: root `Cargo.toml` (MSRV only), server `src/lib.rs`, `src/cross_identity.rs` (not cited) and `auth/nip98.rs`.
- **Moved citations:** I re-resolved every moved citation by content at the pin. I printed each cited line at `9b6143c` and matched it to the label, not to an offset; the full list is in the session log.
- **New citations:** I read each added citation at the pin before using it.
- **Framework evidence:** the actix ordering claim comes from the locked actix-web 4.14.1 source in the local cargo registry. It is cited only in this report, not in the topic.

Gate commands, run from the worktree root, with the generator's exit status captured directly rather than through a pipe:

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 01-workspace-composition.md --check --cite-check --strict-citations --render   # exit 0
node scripts/diagram-index-gen.cjs docs/diagrams --only 02-server-boot-and-routes.md --check --cite-check --strict-citations --render   # exit 0
```

- **SP-01 result:** 143 citations verified at the declared revision, 0 warnings, 0 unresolvable, 12/12 diagrams rendered with Pele, no JEV advisories.
- **SP-02 result:** 149 citations verified, 0 warnings, 0 unresolvable, 11/11 rendered, no JEV advisories.

**Not verified:**
- consumer claims in nostr-rust-forum, VisionClaw and agentbox (SP-01.4 E1, SP-01.10)
- whether cargo-deny gates release builds
- runtime behaviour of the middleware stack, proxy-header handling and port shifting
- CI run results, as opposed to the workflow definitions

**For the lead:** `docs/diagrams/VERIFICATION.md` is now stale for SP-01 and SP-02 after the restamp. It is not my file, so I left it unchanged.

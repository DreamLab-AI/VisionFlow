# Passkey contract evidence

Observed 1 November 2026 against forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. Twelve exact exports and old/consumed revision comparisons are in `source-manifest.json` and `source-differences.json`. Production code and accounts were not changed.

## Native reproduction

Create a disposable Git archive of the pinned forum commit. Before adding tests, the existing suite is:

```sh
cargo test --manifest-path /path/to/archive/Cargo.toml -p nostr-bbs-auth-worker --lib webauthn::tests
```

From this evidence directory, append the audit module only to that disposable archive:

```sh
python3 prepare_tests.py --kit /path/to/archive --out /path/to/fixture-output
cargo test --manifest-path /path/to/archive/Cargo.toml -p nostr-bbs-auth-worker --lib webauthn::
```

The preparer refuses a Git checkout and requires byte-identical original worker source. It inserts `audit-tests.rs`, replacing its labelled placeholder with the exact hash statements exported from `register_verify`. `appended-tests.rs` is the complete executed module. `extraction.json` records the original source/slice hashes and adaptation: a synthetic explicit server-secret parameter replaces environment lookup. No production configuration secret is read.

`native-existing.log`: 65 existing cases pass. `native-audit.log`: the same 65 plus ten audit cases pass, 75 unique cases total. Tests call actual worker parsing helpers and core key derivation in a native unit-test build. No Worker request, browser API, D1 access or hardware PRF runs. The salt case deliberately holds PRF output constant while changing the actual HKDF salt; it isolates that defect and does not emulate an authenticator.

`harness-api-error.log` records an initial audit-module compilation error: the pinned p256 API names the point conversion `to_sec1_point`, not `to_encoded_point`. The fixture was corrected and rebuilt. This is a harness error, not a product failure. The observed runs reused the existing forum `target` directory through CARGO_TARGET_DIR; only disposable source was changed.

The ten audit cases cover changed salt/identity, publicly reproducible empty-secret fallback, a nonempty-secret control, missing/contradictory COSE metadata, duplicate coordinates, invalid curve-point rejection, empty packed attestation parsing, truncated unknown-field acceptance and truncated known-field rejection. A packed parser result is not a full registration acceptance result; the handler has additional checks.

## SQL reproduction

```sh
python3 sql_probes.py --out /path/to/sql-results.json
```

Four scenarios use SQLite and the exact SQL strings extracted from retained source. The output contains those strings and their hashes. Cases cover duplicate owner rows, known/missing credential lookup, two prechecked requests finishing unconditional batches, and serial reuse correctly finding no challenge.

These use the worker bootstrap schema only. They do not establish effective deployed constraints or actual D1 scheduling. The concurrency case chooses two reads before either write, assumes both requests have independently valid NIP-98 tokens and WebAuthn assertions, and executes only the database portion. It does not bypass or test the separate NIP-98 replay store. Synthetic owner/credential strings stand in for previously validated request fields.

## Source-only findings

Missing registration headers, optional creation-time PRF results, assertion transport API mismatch, saved-public-key discovery, public lookup routing, cleanup scope and packed-statement verification were traced in source and compared with primary documentation. They are not full browser/HTTP reproductions. The report distinguishes these from native parser/derivation evidence and SQL schedule evidence.

No browser, real account, credential, message, provisioning operation or deployment ran. Native signature controls include both the existing wrong-key rejection and high-S acceptance cases. Passing those tests does not establish a working client/worker registration or recovery journey.

## Documentation validation

`validation.json` records the final checks. All 32 panels in the three touched topics rendered. Nine changed panels were visually inspected; their PNGs are retained in `previews/`. Whole-corpus citation resolution covers 193 topics and 1,673 panels: 25,037 citations resolve at declared revisions, with zero warnings, zero working-tree fallbacks, zero unresolvable citations and nine pre-existing allowlisted exceptions. Index/revision freshness passes. These structural checks do not establish whole-estate semantic completion.

The report has zero suspicious Unicode findings. The prose scan reports three flags: two ordinary definite-article openings and the concrete term `test-harness`; these were manually reviewed and retained. Fifteen local link targets exist. All twelve source exports were byte-compared with their declared Git revision. `evidence-sha256.json` covers every evidence file except itself.

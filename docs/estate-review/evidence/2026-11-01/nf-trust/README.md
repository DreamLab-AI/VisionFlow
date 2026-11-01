# Trust-transition evidence

Eleven exact source/document exports pin forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. Every export matches both old diagram revision `1b27f79` and website-consumed revision `9c36b6032640c89879ae52fe25da99493979a0b6`. `source-manifest.json` records hashes and comparisons. Production source and member state were not changed.

## Native tests

Create a disposable Git archive of the pinned forum commit, then run:

```sh
cargo test --manifest-path /path/to/archive/Cargo.toml -p nostr-bbs-relay-worker --lib trust
python3 prepare_tests.py --kit /path/to/archive
cargo test --manifest-path /path/to/archive/Cargo.toml -p nostr-bbs-relay-worker --lib trust
```

The preparer refuses a Git checkout and requires original `trust_sweep.rs` bytes to match the retained export. It inserts `audit-tests.rs` inside the existing test module to use the production engine and the upstream in-memory store. No policy or engine body is replaced. `extraction.json` records the original and added-test hashes. The observed runs reused the forum's existing target directory via CARGO_TARGET_DIR.

`native-existing.log` has 28 passing existing cases. `native-audit.log` has those same 28 plus four new cases: 32 unique tests, zero failures. The added cases cover a held prefix starving a later candidate for three capped ticks; a larger-budget positive control; a persistently failing prefix with the same starvation; and unchanged qualifying counters retaining TL2 after one and ten additional years. The fairness fixtures use four rows and cap three, exercising the actual configurable engine. They do not simulate 50,000 deployed users or execute D1.

## SQL tests

```sh
python3 source/scripts/tests/test_trust_transaction.py -v
python3 promotion_sql_probes.py --out /path/to/promotion-sql-results.json
```

The retained upstream script locates the retained `source/crates/.../trust_sweep.rs` through its own relative path. Five existing tests execute exact demotion SQL, including eight concurrent policy-input subcases within one test, no-op/replay, missing rows, null-safe conflict and audit-failure rollback. Count this as five tests, not five plus its subcases. `demotion-sql.log` records the observed run from the disposable archive with identical files.

Six promotion scenarios extract the UPDATE and INSERT strings from retained `trust.rs`, retain their hashes and execute them in separate SQLite autocommit operations. The probe mirrors the source's ignored statement-error sequencing after a qualifying member snapshot. Cases cover serial success, a concurrent TL3 grant, concurrent moderation history, update failure, audit failure and a deleted member. The latter five expose the recorded gaps. Synthetic table definitions contain the relevant columns; effective deployed schema and scheduler interleavings are not established.

No live Worker, D1 binding, cron tick, administrator endpoint, account or browser was exercised. The full async promotion function is traced in source rather than executed. The September workerd/miniflare receipts are retained as historical documents and were not rerun or counted among today's checks.

## Documentation and comparison

NF-03.8/9 and NF-11.11/12 receive scoped corrections; four new NF-24 panels separate engine fairness, guarded demotion, unchecked promotion and policy semantics. Other panels rendered as part of those topics are not newly assessed. `web-sources.json` records the two primary documentation sources used for policy/transaction comparison. Neither market adoption nor community-health benefit was measured.

## Final validation

All 34 panels in the three touched topics render; eight changed panels were visually inspected, with PNGs retained in `previews/`. The old trust-ladder layout was replaced after inspection exposed clipped labels; sweep sequence order now advances the cursor before policy/commit. Initial render logs retain corrected authoring mistakes: two punctuation-only line citations and semicolons rejected by the sequence renderer. They are documentation-tool failures, not product findings.

The corpus gate covers 194 topics and 1,677 panels: 25,048 citations resolve at declared revisions, zero warnings, zero working-tree fallbacks, zero unresolvable citations and nine pre-existing allowlisted exceptions. Freshness checks pass across 47 revision labels. The report has zero suspicious Unicode findings and zero mechanical prose flags; nine local link targets exist. These checks do not certify estate-wide semantic review. `validation.json` records the scope and `evidence-sha256.json` hashes every evidence file except itself.

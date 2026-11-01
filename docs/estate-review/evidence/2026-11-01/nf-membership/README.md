# Membership lifecycle evidence

Twenty-five exact exports pin forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. `source-manifest.json` records SHA256 values and parity with old diagram revision `1b27f79` and website-consumed revision `9c36b6032640c89879ae52fe25da99493979a0b6`. All exported bytes match both revisions. No production source, account or member state was changed.

## Native execution

Create a disposable Git archive of the pinned forum commit. From that archive run:

```sh
cargo test -p nostr-bbs-core --lib whitelist_sql
cargo test -p nostr-bbs-forum-client admin::membership
cargo test -p nostr-bbs-relay-worker --lib whitelist::tests
cargo test -p nostr-bbs-relay-worker --test whitelist_cohort_writers
python3 /path/to/this/evidence/prepare_tests.py --kit /path/to/archive
cargo test -p nostr-bbs-core --lib whitelist_sql
cargo test -p nostr-bbs-forum-client admin::membership
```

The preparer refuses a Git checkout, verifies the two original files against retained exports and inserts the added cases inside their existing test modules. It changes no production function body. Observed runs reused the forum's target directory through CARGO_TARGET_DIR. `extraction.json` records original-file and test-addition hashes.

| Selection | Existing | Added | Unique passing |
| --- | ---: | ---: | ---: |
| Shared whitelist SQL | 12 | 3 | 15 |
| Client membership | 9 | 3 | 12 |
| Relay cohort deltas | 6 | 0 | 6 |
| Source/migration guards | 4 | 0 | 4 |
| Total | 31 | 6 | 37 |

`core-existing.log` and `client-existing.log` retain baseline results. `core-audit.log` and `client-audit.log` contain the final 15/12-case runs. `relay-delta.log` and `source-guards.log` retain the other selections. The initial `client-target-error.log` records a harness mistake: this client has a binary target, so the attempted `--lib` selection failed. The corrected run omitted that flag; the error is not a product defect or a passing test.

Added core cases execute the actual shared SQL through upstream rusqlite helpers: duplicate grants on new/existing rows and suppression by a legacy JSON null. Normal typed writer inputs do not create nulls. Added client cases call the actual pager and Pending helper: the real 500-page/50,000-row cap; a deletion shifting a 201-row fake-relay OFFSET walk; and case-folded deduplication retaining only the first row's cohorts. The fixture is neither a browser nor a D1 integration test.

## Extracted SQL and source inventory

```sh
python3 sql-probes.py > /path/to/results.json
```

The runner verifies source hashes, extracts Rust string literals and records the executed statements. Python supplies SQLite fixtures and selected statement schedules, not a reimplementation of the Rust handlers. Its seven cases cover preserved expiry with a non-expired control, lost interleaved invite grant, serial invite control, residual auth admin flag, case-sensitive identity lookup, invite code/ID mismatch and grant after hard deletion. Synthetic seed/update statements arrange the fixture; they are not claimed to be production mutations.

`sql-probes.json` records SQLite version and assertions. `phase-inventory.json` records commands, output and exit codes for bounded migration, module, route and shared-writer searches in the disposable source archive. It finds no proposed relay membership module/table/endpoint, while retaining unrelated reconciliation hits. Searches alone do not establish runtime correctness or whole-tree absence of every possible implementation.

No live D1, async Worker handler, administrator request, browser session, pod removal or end-to-end signup-to-post journey ran. Passing audit tests reproduce the specified counterexamples; they do not mean those defects were fixed.

## Documentation validation

All 15 panels in the two touched topics render. Seven changed/new panels were visually inspected and their previews retained. The initial NF-25 render found three blank-line citations; these were corrected before the final run. The old NF-08 bootstrap/template notes retain their prior review scope.

The corpus gate covers 195 topics and 1,683 panels: 25,078 verified citations, zero warnings, zero working-tree fallbacks, zero unresolvable citations and nine pre-existing allowlisted exceptions. Freshness checks pass across 47 revision labels. These are documentation checks, not an estate completion percentage.

The report has no suspicious Unicode findings. After editorial review, the prose scanner reports one sense-dependent spelling flag around the migration-acceptance sentence; the UK-English wording is retained. Initial and final logs remain available. `validation.json` records check scope; `evidence-sha256.json` hashes all evidence files except itself.

`web-sources.json` identifies RFC 7644 and the versioned Keycloak API used for protocol and propagation comparisons. No adoption, performance or usability study was performed.

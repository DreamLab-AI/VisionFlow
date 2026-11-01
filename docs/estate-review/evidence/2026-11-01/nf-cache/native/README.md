# Native cache audit fixture

From this directory, run:

```sh
rustc --edition=2021 --test harness.rs -o /path/on/executable/filesystem/cache-tests
/path/on/executable/filesystem/cache-tests --test-threads=1
```

`read_cache.rs` contains the unchanged complete source module followed by five audit tests. `harness.rs` contains four unchanged extracted methods with fake dependencies and seven additional tests. Eight original module tests also run. `extracted-methods.json` identifies source paths, start lines and method hashes. The retained build script regenerates from `/tmp/vf-nf-cache-artifacts/source`; compiling these committed fixture files does not require that export directory.

No network calls, actual database, Workers runtime or socket delivery is involved. The authoriser fixture always withholds, allowing the kind-41 dispatcher omission to be compared against kind-40/42 controls. Native outcomes demonstrate routing, cache and ledger boundaries only. The unused LOOKUP_TTL_SECS warning is from the isolated fixture's literal TTL.

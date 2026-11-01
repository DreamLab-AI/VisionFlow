# Session lifecycle evidence

Source pin: agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf`. `source-manifest.json` records 32 byte-exact files; `source-slices.json` records bounded wiring excerpts and their original whole-file hashes. `revision-comparison.json` compares retained files with AB-12's existing `6500582dfae23bf8a38a6942916d208154e43625` stamp: 31 are unchanged, while ADR-2153 has later notes. Neither stamp is a newly established running-image identity.

## Reproduce the native probes

Export the pinned repository into a disposable directory, then apply `probes/session-probes.patch` with `git apply`. It adds tests to `services/agentbox-sessions/tests/plane.rs`, without changing production code. Run from that crate:

```sh
cargo test --locked --offline -- --test-threads=1
```

This execution used `/tmp/vf-ab-session-artifacts/agentbox` and the source checkout's ignored Cargo target directory, because `/tmp` is not executable. Fixtures create temporary Git repositories, dummy shell processes and uniquely named private tmux servers; their drop paths remove those servers and temporary repositories. Real provider/management variables and `CODEX_HOME` were removed from the test environment, and fixture planes disable boundary HTTP. Tests were serial because an existing test changes process environment. No operator pane was sent input, attached, switched or terminated.

| Receipt | Result |
| --- | --- |
| `logs/native-tests.log` | 44 unit and 22 integration tests pass: 59 existing tests and seven audit additions. Zero skipped native tests; zero doc tests receive no extra credit. |
| `logs/initial-tests.log` | First run fails a broader directory-move hypothesis: Git's already-checked-out branch guard prevents replacement. The final suite keeps that case as a passing counterexample and adds the narrower worktree-to-shared case. |
| `logs/node-tests.log` | Nine existing tests pass; one live-provider-launch tier is explicitly skipped. Static launch-contract parity, mocked session-control forwarding and bridge readiness loss/recovery are exercised. |
| `logs/initial-node-tests.log` | Archive run fails two environment-dependent cases: executable access under the noexec temporary mount and missing `ws` dependency. Re-running existing tests read-only from the pinned checkout resolves both; these were fixture limitations, not product findings. |
| `node-environment.json` | Live launch/round-trip flags were disabled; records the primary checkout used and equal hashes for the installed/source session client. |
| `runtime-observation.json` | Read-only supervisor and socket list at the recorded time. No provider prompt, pane transcript or binary/source identity check. |
| `retired-repository-removal.json` | Separately authorised removal of the clean retired AoE source checkout after fresh remote identity and local-use checks. Native service worktrees and recovery mounts were preserved. |

Existing Node checks ran with:

```sh
node --test tests/cli/session-launch-contract.test.mjs \
  tests/cli/session-control.test.cjs \
  tests/cli/tab0-coordinator-readiness.test.mjs
```

`AGENTBOX_SESSIONS_BIN` pointed to the built audit binary for static resolution. The readiness test starts a local bridge against a fake socket daemon and an absent tmux target. It checks readiness clears and returns; it does not submit an agent prompt. The source and installed clients match byte-for-byte.

## Interpreting the probes

New tests assert current behaviour: directory-change and Git-error restarts despite work; an API tool blocked in status but accepting input; submission preceding a hook-write failure; removed seeds surviving reload and refusing API close; and API worktrees misreported and retained after close. The seventh test is the Git branch-guard counterexample. Their success is evidence of those behaviours, not remediation.

Source-only findings cover bridge duplicate-delivery risk, independent CLI controllers, invoking-client ambiguity, idle/readiness boundaries and non-durable boundary failure handling. No live duplicate instruction, client switch, management outage, provider round trip or production worktree cleanup was attempted. Full role isolation, provider provisioning and submodule deletion safety remain separate acceptance work.

## Document verification

Render, citation, source-hash and document-check receipts are retained under `logs/` and in `validation.json`. Only AB-12.15 and five new AB-39 panels receive review credit. Rendering other AB-12 panels does not reassess them. Whole-corpus citation/index checks validate references and catalogue structure, not the semantics of unreviewed topics.

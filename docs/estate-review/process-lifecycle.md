---
title: Runtime process identity and shutdown
status: in-progress
date: 2026-09-04
type: explanation
---

# Runtime process identity and shutdown

A persistent agent workspace needs to recover abandoned work without stopping unrelated sessions. Agentbox's reaper improves process discovery by preserving argv boundaries, but recognising a command shape is only one part of safe lifecycle control. [Evidence](evidence/process-lifecycle-snapshot.json) records four passing native tests and current source hashes. No daemon was started, stopped or inspected through a live discovery command in this pass.

## What the reaper establishes

[The shared helper](../../../project/agentbox/services/agentbox-ops/src/procs.rs) accepts separate `daemon`, `start` arguments following recognised ruflo/claude-flow launchers. Node scripts must have a recognised basename or a `cli.js` path containing a recognised package component. Shell command strings, searches and embedded prompts do not qualify. This is a lexical allowlist, not executable authenticity or a binding to the session that started the process. Unknown Node flags and wrappers are refused until deliberately supported.

Three existing helper tests cover launcher examples, workspace argument boundaries and elapsed-time formatting. The [reaper's](../../../project/agentbox/services/agentbox-ops/src/bin/ruflo-daemon-gc.rs) additional test rejects zero, oversized and truncating registry PIDs. These tests establish useful local safeguards; none exercises signalling or a process-replacement race.

## Identity, policy and outcome are separate

Registry discovery runs first. The process sweep uses `entry(...).or_insert(...)`, so it does not replace a registry workspace with the live argv workspace for the same PID. Staleness uses that retained workspace's existence or process age above the TTL. A stale registry workspace can therefore influence eligibility even when current argv contains another workspace. This is a source-derived case requiring a fixture, not an observed mistaken termination.

Confirmation takes a fresh process snapshot and checks launcher shape only. Age is read separately. Before signalling, the reaper checks shape again but does not bind the PID to captured start time or re-evaluate workspace/age. PID reuse by another recognised daemon can satisfy the same predicate. A process handle tied to identity, plus a defined policy for rechecking eligibility, remains necessary for the stronger guarantee. ADR-2032 already acknowledges the narrower confirm-to-signal race.

The `killed` output list records successful SIGTERM delivery, not observed exit. Closeout needs distinct requested, signalled, exited and failed outcomes with timeout handling. A daemon may handle or ignore termination; the source does not wait for completion.

## Other signalling paths

The [Hermes scheduler](../../../project/agentbox/services/agentbox-ops/src/hermes/mod.rs) reads its PID file and checks `/proc/<pid>` existence. Its CLI Stop command invokes `daemon_stop`, which sends SIGTERM without argv/start-time confirmation and then removes the PID file even if signalling fails. Successful signal delivery is printed as stopped. A stale PID file naming an unrelated live process would pass the existence check. This is source evidence only; no stop command or forged PID file was used.

The same existence-only test can report a scheduler as running and suppress a start. This makes stale-state recovery part of lifecycle acceptance, beyond the reaper's launcher matcher. The token-audit paths inspected here use the shared sweep for reporting; that does not make them signalling paths.

## Closeout

ADR-2032 is partial against its stated rule for any signalling tool; its narrower ruflo matcher remains implemented and staged activation is preserved. CP-01/04/08 should inventory signal callers and define identity and authority per process role. Runtime/operations maintainers should cover stale registry versus live workspace, unrelated PID reuse, reuse by a recognised daemon, unknown wrappers, inaccessible proc data, failed signals and delayed/ignored termination. Use isolated owned subprocesses and explicit expected outcomes. Capture release identity and acceptance receipts before claiming deployed recovery. Do not turn this assessment into a live reaping exercise.

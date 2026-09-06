---
title: Adapter lifecycle and dispatch contracts
status: source-and-isolated-probe-verified
date: 2026-09-04
type: explanation
---

# Adapter lifecycle and dispatch contracts

The five adapter slots provide a useful substitution boundary for beads, pods, memory, events and orchestration. The current lifecycle and middleware implementation is more conditional than the operative ADRs' universal guarantees. [Evidence](evidence/dispatch-privacy-probe.json) pins the source and records actual privacy-wrapper calls with an injected local fetch response; no real adapter, sidecar or persistent store was exercised.

## Connection rejection and timeout differ

The [resolver](../../../project/agentbox/management-api/adapters/index.js) constructs all five slots, defaults unspecified selections to off, and wraps prototype methods except lifecycle and underscore-prefixed helpers. The [server connection phase](../../../project/agentbox/management-api/server.js) handles an explicit orchestrator connect rejection by exiting. Other explicit failures mark the slot degraded and attempt replacement with its off implementation.

However, the aggregate connection phase races against a ten-second timer. On timeout, startup logs a warning and continues with partially connected adapters. The pending promises are not cancelled by that race; late completion or rejection can still affect state. If constructing the off replacement fails, a caught error leaves the degraded original adapter in place. Therefore neither universal fatal orchestrator non-readiness nor universal replacement is established. These are source-control-flow findings, not a live startup fault injection.

Acceptance must distinguish connecting, ready, disabled, degraded, timed out and late-failed states. Define readiness per operation, decide what happens when a load-bearing connect never settles, and prove replacement/recovery behaviour before relying on the slot's absence as a cleanly disabled condition.

## Middleware traversal is not universal redaction

[wrapDispatch](../../../project/agentbox/management-api/observability/metrics.js) applies observability around the privacy wrapper. JSON-LD encoding remains a separate caller responsibility, as the source comment states. The wrapper itself does not structurally compose every call with an encoder.

The [privacy filter](../../../project/agentbox/management-api/middleware/privacy-filter.js) recognises only `store`, `write`, `create`, `publish`, `append` and `emit` as write operations. It extracts an object's `value` field or the second positional argument. Other methods pass through. OPF mode defaults off; configured policies can also be soft or off. The marker records traversal, including intentional pass-through, rather than proof that all content was sanitised.

The [reproducible probe](evidence/dispatch-privacy-probe.cjs) uses strict policy and a deterministic mock redaction response:

| Synthetic call | Filter calls | Adapter-visible result |
|---|---:|---|
| store with sentinel in key, value and metadata | 1 | Value replaced; key and metadata unchanged |
| createEpic with sentinel title | 0 | Title unchanged |
| store with object-valued value | 1 | Object value replaced by returned string |

`createEpic` is a real method name in the beads external adapter, but the probe exercises the wrapper with a fake function, not actual persistence. These results establish dispatch coverage and argument-shape limits, not real data disclosure. Value stringification also requires a schema-preservation decision when adapters expect structured input.

The [encoder](../../../project/agentbox/management-api/middleware/linked-data/encoder.js) checks privacy traversal before surface selection. An unmarked payload throws for pods/memory but is logged for other slots when OPF is active; with OPF off the check is disabled. This enforces a configured order at an explicit encoder entry point, not universal redaction of adapter results or arbitrary direct calls. Observability logs error messages/stacks directly, so its diagnostic content needs a separate policy; no error-content leak was reproduced here.

## Closeout requirements

CP-03/04/08 require a per-adapter method contract identifying mutation, sensitive fields, argument/result schema and encoding responsibility. Replace assumptions based solely on method names with verified coverage. Exercise strict/soft/off, invalid policy, redactor failure, structured values, metadata and direct encoder calls. Prove that payload types and allowed content survive the full route, adapter and federation sequence.

Add lifecycle fault injection for never-settling connect, late rejection, failed off construction and restart. Capture readiness and mutation outcomes without equating process liveness to correct storage. ADR-2004/2005 become partial for their broad guarantees while preserving the accepted slot and ordering design. This assessment does not implement those repairs or claim an operational incident.

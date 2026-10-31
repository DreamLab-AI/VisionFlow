---
title: Obsidian plugin exclusions and write recovery
status: in-progress
date: 2026-10-31
type: explanation
---

# Obsidian plugin exclusions and write recovery

Scope: `ontomark-obsidian` at `272259edcda1fbc2fc5856ffa633f1a9a01a670b`, specifically ADR-OBS-005, exclusion rules, host indexing/event adapters and the patch application boundary. The checkout has pre-existing changes in other files, including ADR-OBS-003 and README; those are not assessed here. The inspected API, main, exclusions and ADR-OBS-005 files have no diff from HEAD. [Hashes and probe receipt](evidence/2026-10-31/obsidian-boundary-probe.json) bind the observations to source bytes.

## ADR-OBS-005 section dispositions

| Section | Evidence and assessment | Remaining acceptance |
|---|---|---|
| Context | The declared template-placeholder problem motivates filtering before indexing. `src/excludes.ts:8` defaults to templates; `src/main.ts:267` filters the initial list. This supports the chosen mechanism, not a reproduction of the historical bug. | Historical diagnosis need not be repeated to verify current filtering. |
| Decision 1: top-level folder rules | `src/excludes.ts` compares the first path segment without case; `folderListProblem` rejects separators and parent-like names. `src/settings.ts:121` wires validation into the control. Fourteen exclusion tests passed, including nested-folder and similarly named file counterexamples. | Obsidian control interaction was not exercised. |
| Decision 2: native ignored files | `parseIgnoreFilters` accepts prefixes and slash-wrapped regexes, returning no filters on malformed JSON. `src/main.ts:334` retains the previous filters on read exceptions; `startApi:301` reloads before indexing and `saveSettings:167` reindexes on changed rules. Source agrees with the specified delayed refresh. | Malformed configuration intentionally removes restrictions: this is a convenience exclusion mechanism, not a fail-closed confidentiality boundary. Independent parity with Obsidian's implementation is unverified. |
| Decision 3: initial index and events | `src/main.ts:267` filters listing; handlers at `main.ts:368` wait for readiness and filter create/modify. Rename into an excluded folder calls deletion; rename out follows the target. `reindex:180` replaces the API and reloads filters through `startApi`. | This is a source trace. Real event timing, worker replacement and the cited offline end-to-end suite were not executed. The phrase “every point” should be read in its host indexing/event scope, not as authorisation enforced inside the public API. |
| Decision 4: pure rules | `src/excludes.ts` has no Obsidian import and its unit suite ran directly. | Host wiring still needs independent runtime verification. |
| Why host-side | The code places folder policy in the plugin adapter while the API accepts files from its vault port. This agrees with separation of policy and engine semantics. | No conclusion about CLI parity or other engine hosts. |
| Not adopted | Content heuristics are explicitly rejected; inspected exclusion rules use paths/configuration. | No whole-repository absence claim about unrelated features. |

## Confirmed failure boundary: write succeeds before index update rejects

`src/api.ts:237` confirms the patch, then calls the vault's atomic `process` callback, rejecting stale base text. This is a useful concurrency guard. After the file mutation, `api.ts:254` awaits `client.upsert` outside the write-error catch. If that call rejects, the API rejects after the write has already persisted and produces no `applied` outcome. There is no rollback or “written, indexing failed” outcome in this method.

An isolated probe bundled the actual API with esbuild and injected a failing index client:

```javascript
const api = new OntomarkApi(
  { upsert: async () => { throw Error('injected index failure'); } },
  { process: async (path, transform) => text = transform(text) }
);
// text initially 'old'; proposal base_text 'old', new_text 'new'; confirm returns true
```

Observed result: `{"file_text":"new","rejected":"injected index failure"}`. The fake vault applies exactly the documented callback contract. No actual vault was modified. This proves the API boundary, not the frequency of worker failures or what a real Obsidian user sees after recovery. The host's event handler may independently reindex the file; that recovery was not demonstrated.

**Impact:** callers cannot infer “nothing changed” from rejection. A blind retry may be refused as stale even though the original change succeeded. **Proposed owner:** plugin API/host maintainers. **Acceptance:** inject index failure after an approved write, expose the persisted-write state, demonstrate idempotent recovery/reindex, and verify the visible outcome through the real host. A repair should preserve the atomic stale-text guard rather than replaying the mutation without checking it.

## Validation limits

The command `vitest run test/excludes.test.ts test/api.test.ts` exited 0 with one file and fourteen tests passed. There is no `test/api.test.ts` in this checkout: Vitest accepted the selection while running only the exclusion file. Therefore this is not an API-suite pass. The separately executed injected-failure probe supplies the bounded API evidence. No browser, model endpoint, real Obsidian instance or full plugin build ran.

The other four plugin ADRs now have [source-qualified section assessments](2026-10-31-obsidian-decisions.md). This pass does not certify plugin security, all ingest/network paths, or engine correctness.

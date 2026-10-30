# Jev option

An opt-in, **advisory** step that asks a typed-decision model (TypeSafe System One: Choice / Score / Noul over `POST /v1/systemone`) a bounded question beside four existing deterministic checks. **The library defaults off; the Agentbox estate deployment enables all four through `[system_one.consumers].visionflow_advisories`.** With the flag off, or with Jev unavailable for any reason, every script prints exactly what it printed before and exits with the same code (asserted against the pre-change scripts in `tests/gates/jev-option.test.cjs`).

Status of every item: *proposed research candidate; no measured uplift.* The tests prove wiring and fallback, not value.

## Items

| Item | Seam | Flag | Data class | Jev adds |
|---|---|---|---|---|
| VF-06-S1-02 citation vs source | `scripts/diagram-index-gen.cjs` | `--jev[=diagram]` with `--cite-check` (+ `--jev-report PATH`, `--jev-max N` requests of 24 citations, default 5) | public | One Noul per citation the gate resolved: do the cited lines hold what the diagram label says? A citation whose defect probability (`1 - noul`) is at least 0.9 (`VISIONFLOW_JEV_FLAG_AT`) prints to **stderr** as `JEV-ADVISORY topic:diagram file:line p_defect=…`; the JSON report lists every answered citation. Never changes an exit code, an index or a stamp. |
| VF-01-S1-01 claim vs evidence | `scripts/check-augmentation-citations.cjs` | `--jev[=claim]` | local-only | One Choice per non-`absent` matrix cell (`supported / overstated / contradicted / insufficient`); non-supported verdicts print as `JEV claim ...` **after** the gate's summary line. No status is changed. |
| VF-03-S1-01 health priority | `scripts/estate-health.mjs check` | `--jev[=health]` | public | One Score per failed/unreadable observation; `JEV priority N ...` lines before the verdict sentinel. |
| ES-09-S1-01 failure triage | `scripts/estate-health.mjs check` | `--jev[=triage]` | public | One Choice per failed CI run (`source-build / stale-bind-path / pin-or-schema / dependency / gpu-xr / runtime-evidence-gap / insufficient`); `JEV triage ...` lines before the sentinel. |

`VISIONFLOW_JEV` enables the same items from the environment: `1`/`all`, or a comma list (`diagram,claim,health,triage`). `--jev` on the command line wins. Agentbox projects `VISIONFLOW_JEV=all` and an explicit sovereign endpoint into service and shell environments at boot. This covers local diagram/claim checks and health/triage CLIs without changing their library defaults. Hosted CI and the isolated dream annexe do not inherit Agentbox environment: an unset flag there still makes no call. The annexe remains token-free.

**Authority stays in code.** The CI-colour rule, HEAD verdict, `unknown`/`red`/`green`/exempt rules, citation resolution, syntax/render gates, ADR checks and the sentinel line are untouched and run first. Jev output cannot change a pass/fail result, write an index, restamp a topic, or edit the snapshot. `collect` never calls Jev.

## What leaves the process, and where to

Nothing, unless the flag is on **and** an endpoint resolves. Then one HTTPS `POST` per request carries the model name, a structured `state` and the questions:

- VF-06: for up to 120 citations (5 requests of 24), the diagram line that carries the citation and the cited lines themselves (each line cut to 220 characters).
- VF-01: the matrix cell text and bounded excerpts of the cited sibling-repo files.
- VF-03/ES-09: repository names, workflow names and outcomes (up to 12 items). These are already published in `website/static/data/estate-health.json`; **no URLs, logs or failed-step text are sent** (the snapshot carries none, so triage evidence is thin by construction).

VF-01 reads source from sibling checkouts, so it is **local-only**: it refuses the agentbox proxy and the TypeSafe cloud endpoint unless the endpoint is declared local (`AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL=1`), or you name an explicit endpoint yourself with `VISIONFLOW_JEV_ENDPOINT`. VF-06 is **public** since S1-02 (operator decision, 2026-10-29: the estate code it quotes is not private), because its 0.9 flag was calibrated on the TypeSafe cloud model and would not carry over to a local one.

Endpoint resolution (first match wins; `scripts/lib/jev.cjs`):

1. `VISIONFLOW_JEV_ENDPOINT` (+ `VISIONFLOW_JEV_MODEL`, default `jev-latest`; `VISIONFLOW_JEV_KEY_ENV` names the env var holding a Bearer key).
2. `AGENTBOX_SYSTEM_ONE_ENDPOINT` (or `AGENTBOX_SKILL_ROUTE_API`) + `AGENTBOX_SYSTEM_ONE_MODEL` (or `AGENTBOX_SKILL_ROUTE_MODEL`) when `AGENTBOX_SYSTEM_ONE_PROXY=1` (loopback proxy; no `Authorization` header sent).
3. `https://api.typesafe.ai/v1/systemone`, model `jev-latest`, Bearer `$TYPESAFE_API_KEY`, only if that key is set.
4. Otherwise Jev is unavailable and the script falls back.

## Fallback

Any of: flag off, no endpoint, local-only refused, transport error, timeout (hard deadline 15 s, `VISIONFLOW_JEV_TIMEOUT_MS`), 429/529 after one bounded retry, any non-2xx, unparseable body, a missing answer or wrong answer type, a Choice key outside the offered set, a Score outside the rubric, a Noul outside [0,1], or confidence below `VISIONFLOW_JEV_MIN_CONFIDENCE` (default 0.5, **unfitted**; Noul confidence is taken as `|p−0.5|×2`) → the script behaves exactly as before. One stderr line is logged (`jev: fallback reason=<code> set=<question-set version>`); no request content, response body or key is ever logged. A request fails over as a whole: one bad answer discards that request's answers. VF-06 sets no confidence floor (`minConfidence: 0`): its 0.9 flag on `1 - noul` is the threshold, so a hesitant answer is kept, not discarded.

Question-set versions: `vf-01-s1-01/claim-check/v1`, `vf-06-s1-02/citation-check/v1`, `vf-03-s1-01/health-priority/v1`, `es-09-s1-01/failure-triage/v1`.

## VF-06 calibration

S1-01 asked one question per whole diagram (Noul + Choice over at most eight short excerpts). Measured on 28 diagrams that each held a confirmed citation error, reviewed with the error and again with it fixed, it could not tell the two apart: 82% of 784 answers were `uncheckable` and the support score sat near 0.05 either way.

S1-02 asks one question per citation. On 1,048 labelled citations (49 confirmed wrong by a Haiku-then-Sonnet review, 48 planted wrong lines, 951 correct), split in half by id with the wording chosen on one half and measured once on the other, against `jev-1.14.0`:

| | held-out result |
|---|---|
| AUC (wrong vs correct) | 0.86 |
| flag at `p_defect` ≥ 0.9 | 16 of 50 wrong citations, 7 false flags in 465 correct (precision 0.70) |
| flag at ≥ 0.7 | 35 of 50, 46 false flags (precision 0.43) |

A multiple-choice variant (cited lines / elsewhere in the window / not supported) scored 0.84; averaging both, 0.87 at twice the cost; narrowing prose claims to their own clause did not help. Batch size (1, 8 or 24) made no difference, and run-to-run noise is about ±0.01. The labels are noisy (two of the held-out "false flags" are real errors), so these are floors. The wording is pinned by `tests/gates/jev-option.test.cjs`: changing it means a new version and a new measurement.

## Host-owned: VF-04-S1-01 (dream hypotheses vs repo disciplines)

VisionFlow owns the disciplines (`dream.config.json` `extraDisciplines`) but nothing in this repository receives a hypothesis: hypotheses are generated and judged inside the dream engine, outside this tree, and the annexe that runs a night holds no tokens. A screening script here would have no caller, which this repo already records as a defect class (VF-06.13). That host hook now exists: Agentbox `services/dream-engine/src/jev_triage.rs` screens hypotheses and the repository’s declared `extraDisciplines` before candidate evaluation, enabled by `[dream_machine].jev_triage`. It runs in the host process, not by injecting tokens into the annexe.

## Tests

`node tests/gates/jev-option.test.cjs` (also in `tests/gates/run-all.sh`). Mocked transport only (`tests/gates/jev-mock-fetch.cjs`); no live calls.

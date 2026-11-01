# Research and claim boundaries

Prepared 1 November 2026 against VisionFlow `4d150fe64aab9811d51d00a77ac03028bea06c95`. This is source-grounded pre-production research. No new application session, sensor feed, voice capture, maintenance-system connector or industrial end-to-end run was exercised. [Source pins](source-pins.json) retain revisions, line ranges and hashes; they establish what was inspected, not deployed acceptance.

## Load-bearing claims

| ID | What the script may say | Evidence inspected | Boundary for the edit |
| --- | --- | --- | --- |
| C01 | VisionFlow brings together tools for authored knowledge, grounding, agent work and graph views. | VisionFlow `README.md`, “How the pieces fit” and “One workflow”; its explicit integration qualifications | Describe a composition of components. No turnkey industrial deployment claim. |
| C02 | Ontomark provides commands to validate, build, find, retrieve and apply changes to a Markdown corpus. | Ontomark `crates/ontomark-cli/src/main.rs:83–145`, revision in source pins | A custom asset vocabulary and fixture adapter are needed. Validation checks represented structure; it does not establish the truth of a maintenance diagnosis. |
| C03 | Loom exposes retrieval and chat routes, with grounding metadata that includes evidence status and serving identity. | Loom `crates/loom-facade/src/routes/mod.rs:50–67`; `routes/grounding.rs:85–109` | The industrial answer card and refusal policy are proposed consumer behaviour. Neither a score nor `corpus_backed` proves a causal claim. The pinned source has evolved beyond the earlier September audit; use these inspected bytes, then reverify the build used for capture. |
| C04 | VisionClaw has graph-rendering and ontology-browser surfaces. | VisionClaw `client/src/features/graph/components/GraphCanvas.tsx:10–44,50–79`; `features/ontology/components/OntologyBrowser.tsx:43–87` | The pump graph is a prepared application dataset, not an existing plant twin. An animated graph is not a live physical simulation. |
| C05 | VisionClaw includes proposal states and separate governance indicators. | VisionClaw `client/src/features/ontology/components/OntologyProposalList.tsx:25–75` | A displayed “Approved” state is not proof that the reviewed bytes were applied or reached a serving consumer. Do not use it as an industrial work permit. |
| C06 | The amendment writer checks a proposal digest, requires a single-page scope, checks the current subject and can refuse stale patch context. | Ontomark `crates/ontomark-compile/src/write/apply.rs:208–248,305–330` | These are specific guards. They do not certify the whole distributed approval path. The demo changes one knowledge note and preserves its identity. |
| C07 | Agentbox has a native session service and an adapter for applying supported ontology amendments. | Agentbox `services/agentbox-sessions/src/api.rs:89–98`; `management-api/lib/ontology-apply.js:692–725` | Session operations are not machinery controls. Industrial connector, scheduling and complete approval-to-receipt acceptance remain unproved. Agent of Empires is retired and has no role in this story. |
| C08 | Existing audit findings prevent an unconditional claim that every approval and receipt refer to the same applied content. | [Proposal application audit](../../docs/estate-review/2026-10-31-proposal-application.md), “Observed boundaries”; [proposal construction audit](../../docs/estate-review/2026-10-31-proposal-construction.md), identity and grouped-application findings | Capture the exact supported path and receipt before presenting a connected success. Otherwise keep review/application shots staged. Do not hide a refusal with a cut. |
| H01 | P-204, the plant, people, records and timeline exist within this fictional scenario. | [Scenario fixture](fixtures/scenario.json) | Synthetic story facts. No customer endorsement or field trial implied. |
| H02 | An industrial application links asset-specific records, distinguishes superseded procedures and presents evidence beside an answer. | Proposed application behaviour and fixture relations | Requires domain modelling, document-control policy, UI and connector work. The general ontology tools do not supply these domain rules automatically. |
| H03 | A reviewed investigation plan can be handed to a planner and the next shift. | Scripted human workflow; fixture decision | Manual rehearsal first. No live CMMS, ERP, PLC, SCADA or machine-command integration is established. |
| H04 | A pilot might reduce time spent finding sources and reconstructing decisions. | Proposed evaluation in the production guide | A question to measure, not a realised saving. Do not invent percentages, avoided downtime, diagnosis accuracy or a payback period. |

## What each transition must prove later

| Transition | Evidence needed to call it a connected demonstration | Fallback |
| --- | --- | --- |
| Technician observation → incident | Actual input, transcript correction and stored incident tied to P-204 | Actor line plus labelled staged incident card |
| Source records → asset graph | Import mapping, stable identifiers, source references and visible same-asset links | Deterministic editorial graph from the fixture |
| Current corpus → answer | Loaded demo corpus identity, complete request/response, referenced source content and tested no-evidence case | Scripted answer card labelled as proposed application behaviour |
| Proposed note → reviewed decision | Exact content, reviewer action and scoped decision identity | Presenter-led comparison of before/after text, labelled rehearsal |
| Decision → applied note | Agreement of reviewed subject/digest, output bytes and application receipt; refusal case retained | Manual fixture update, explicitly described as staged |
| Applied note → next-shift retrieval | Consumer refresh, new served identity and a fresh query returning the updated content | Editorial handover card; no automatic-refresh animation |
| Plan → maintenance system | Implemented adapter, permission decision, external receipt and duplicate/retry handling | Manual export of a proposed plan; no external success badge |

For each eventual capture, record the repo revision, deployed build, fixture hash, configuration relevant to the result, command or route, output, and media filename. Keep credentials and private records out of production files. No request or response should be fabricated to fill a missing receipt.

## Editorial judgement

Continuity of evidence across people and shifts gives this story its purpose. A chat answer alone would not demonstrate that. The revision conflict and unresolved cause create a meaningful choice, while the review and handover make the outcome visible. The estate's current integration findings are why the initial film is framed as a hypothetical application with a small number of independently verifiable software inserts.

Three adjacent applications can reuse this structure: manufacturing quality investigations, commissioning handover and supplier-change review. Each would need its own domain rules and acceptance evidence. They are closing-discussion options, not extra claims or branches in this film.

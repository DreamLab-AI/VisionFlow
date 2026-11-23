---
id: LM-04
title: Chat completions — passthrough, verbatim and delegated serving, the Jev option and the published client
area: loom
governing:
  - ../loom/docs/design/ADR-138-confidence-surfacing-contract.md
  - ../loom/docs/design/ADR-139-per-request-scaffold-opt-out.md
  - ../loom/docs/jev-option.md
adrs: [loom:ADR-135, loom:ADR-138, loom:ADR-139, agentbox:ADR-2084]
sources:
  - ../loom/crates/loom-facade/src/routes/mod.rs
  - ../loom/crates/loom-facade/src/routes/grounding.rs
  - ../loom/crates/loom-facade/src/serving.rs
  - ../loom/crates/loom-facade/src/config.rs
  - ../loom/crates/loom-facade/src/jev.rs
  - ../loom/crates/loom-backend-openai/src/lib.rs
  - ../loom/crates/loom-client/src/client.rs
  - ../loom/crates/loom-client/src/error.rs
  - ../loom/crates/loom-client/src/tokens.rs
  - ../loom/deploy/compose.profile-a.yml
  - ../loom/deploy/compose.profile-b.yml
  - ../loom/README.md
verified_commit: {loom: 355946dd2deacfdef21171d53eafd025887e447e}
---

## LM-04.1 One handler, four exits: streaming passthrough, verbatim, delegate, backend failure

```mermaid
flowchart TB
    IN["chat_completions<br/>loom-facade/src/routes/mod.rs:344"]
    FL["request_flags read from the ORIGINAL body, loom_options stripped<br/>loom-facade/src/routes/mod.rs:357<br/>loom-facade/src/serving.rs:89,95"]
    SP{"scaffold false AND stream true?<br/>loom-facade/src/routes/mod.rs:361"}
    SSE["opaque SSE proxy, served-mode passthrough header<br/>loom-facade/src/routes/mod.rs:362,522,529"]
    SC["scaffold the LAST user message unless passthrough<br/>loom-facade/src/routes/mod.rs:369,384,385"]
    FAILS["scaffold error: warn and delegate the raw prompt<br/>loom-facade/src/routes/mod.rs:395,398"]
    VB{"verbatim mode, engaged, not streaming, single-turn lookup,<br/>not opted out, top score at or above threshold?<br/>loom-facade/src/routes/mod.rs:415,416,418,577"}
    VER["serve the block as the answer, no backend call<br/>loom-facade/src/routes/mod.rs:580,584,585"]
    MERGE["merge preamble and block into the first string system message or insert one<br/>loom-facade/src/routes/mod.rs:441,442,603,618,621"]
    THINK["engaged only: no-think kwargs or think-token floor<br/>loom-facade/src/routes/mod.rs:452,453"]
    BK["backend.chat<br/>loom-facade/src/routes/mod.rs:461"]
    ANN["200: add the loom block with grounding, exposure and generation<br/>loom-facade/src/routes/mod.rs:466,467"]
    BF["error: backend_failure_response keeps the grounding contract<br/>loom-facade/src/routes/mod.rs:492,493"]
    IN --> FL --> SP
    SP -->|yes| SSE
    SP -->|no| SC
    SC --> FAILS
    SC --> VB
    FAILS --> VB
    VB -->|yes| VER
    VB -->|no| MERGE --> THINK --> BK
    BK -->|ok| ANN
    BK -->|err| BF
    TENS["TENSION: stream true with the scaffold on stays on the JSON path,<br/>the adapter pops stream, so the client gets one body, not SSE<br/>loom-facade/src/routes/mod.rs:358,359,360<br/>loom-backend-openai/src/lib.rs:113,114"]
    THINK --- TENS
    FLOOR["adapter floor: sub-floor integer max_tokens raised, inserted only when<br/>both token keys are absent, LOOM_MIN_MAX_TOKENS default 1536<br/>loom-backend-openai/src/lib.rs:116,122,126,127<br/>loom-facade/src/config.rs:171"]
    BK --- FLOOR
```

**What it shows.** Passthrough is decided before retrieval, and verbatim before any model call. The F1, F2 and F3 switches are all off by default (`loom-facade/src/config.rs:184,186,187,188`); Profile A turns F1 and F3 on (LM-06.3).

## LM-04.2 What the grounding status says, and the per-request switches behind it

```mermaid
flowchart TB
    CS["chat_status<br/>loom-facade/src/routes/grounding.rs:180"]
    P["Passthrough: loom_options.scaffold false<br/>loom-facade/src/routes/grounding.rs:186,187<br/>loom-facade/src/serving.rs:67,71"]
    N["NoMatch: nothing engaged<br/>loom-facade/src/routes/grounding.rs:189,190"]
    O["OptOut: verbatim was possible, loom_options.verbatim false declined it<br/>loom-facade/src/routes/grounding.rs:192,193<br/>loom-facade/src/serving.rs:52,56"]
    S["SemanticFallback: engaged through the HNSW branch<br/>loom-facade/src/routes/grounding.rs:195,196"]
    D["Delegated: engaged lexical scaffold sent to the model<br/>loom-facade/src/routes/grounding.rs:198"]
    V["Verbatim: set on the verbatim exit, not by chat_status<br/>loom-facade/src/routes/mod.rs:583"]
    CS --> P
    CS --> N
    CS --> O
    CS --> S
    CS --> D
    CS -.-> V
    CNT["confidence window and mark_served skip passthrough requests<br/>loom-facade/src/routes/mod.rs:428,429,430,431"]
    P --- CNT
    SHAPE["single-turn lookup = last message from the user, no assistant turn<br/>loom-facade/src/serving.rs:35,44"]
    O --- SHAPE
    INV["INVARIANT: only an explicit false opts out, any other value leaves the mode on<br/>loom-facade/src/serving.rs:47,48,49,50"]
    P --- INV
    SEE["see AB-24.11 for the consumers that send scaffold false"]
    P --- SEE
```

**What it shows.** The status separates "the corpus had nothing", "the caller declined" and "the model answered over a scaffold", which a bare 200 cannot. The README states passthrough reports `corpus_backed: false` (`../loom/README.md:65`).

## LM-04.3 The Jev option: three stages that can only withhold, reorder or annotate

```mermaid
flowchart TB
    ON["Jev stages run only when LOOM_JEV is on, after retrieval and the opt-out<br/>loom-facade/src/routes/mod.rs:387,388,389,390"]
    GATE["gate: may withhold an engaged scaffold, inert until<br/>LOOM_JEV_GATE_MIN_CONFIDENCE is set<br/>loom-facade/src/jev.rs:15,16,19,20"]
    RANK["rank: reorder and re-assemble through the same gate, kept only if budget,<br/>decision and generation are unchanged<br/>loom-facade/src/jev.rs:21,22,23,24"]
    SUP["support: annotation only, adds jev.support, never edits the answer<br/>loom-facade/src/jev.rs:25,26<br/>loom-facade/src/routes/mod.rs:479,482"]
    ON --> GATE --> RANK --> SUP
    subgraph RES["endpoint resolution, first match wins, loom-facade/src/jev.rs:212"]
        R1["LOOM_JEV_ENDPOINT, accepted when LOOM_JEV_LOCAL, a local URL or allow_cloud<br/>loom-facade/src/jev.rs:220,221,222"]
        R2["agentbox loopback proxy, needs BACKEND_LOCAL or allow_cloud<br/>loom-facade/src/jev.rs:232,233,234"]
        R3["TypeSafe cloud, needs a key AND allow_cloud<br/>loom-facade/src/jev.rs:247,248"]
        R4["nothing resolves: NoEndpoint fallback<br/>loom-facade/src/jev.rs:257"]
        R1 --> R2 --> R3 --> R4
    end
    ON --> RES
    TENS["TENSION: LOOM_JEV_LOCAL=1 is a declaration that skips the URL check,<br/>and both profiles set it with an operator-supplied endpoint<br/>loom-facade/src/jev.rs:221,222<br/>../loom/deploy/compose.profile-a.yml:35,37<br/>../loom/deploy/compose.profile-b.yml:33,35"]
    R1 --- TENS
    DEF["config default is off, both compose profiles default LOOM_JEV to 1<br/>../loom/deploy/compose.profile-a.yml:31<br/>../loom/deploy/compose.profile-b.yml:29"]
    ON --- DEF
    SEE["see ES-06 for the estate's privacy line around the Loom"]
    RES --- SEE
```

**What it shows.** Every Jev failure collapses to a named fallback and the request continues unchanged (`loom-facade/src/jev.rs:6,7,8`). The data boundary is local-only by default (`loom-facade/src/jev.rs:30,31,32`), but the local check can be declared rather than proven.

## LM-04.4 The published client: retries worth making, and failures that look like success

```mermaid
flowchart TB
    CH["LoomClient chat<br/>loom-client/src/client.rs:291"]
    BUD["budget = max of requested and the 1536 floor<br/>loom-client/src/client.rs:292,295<br/>loom-client/src/tokens.rs:17"]
    ATT["one attempt: POST base chat/completions<br/>loom-client/src/client.rs:322,333"]
    TR["finish_reason length: double the budget, try again<br/>loom-client/src/client.rs:301,302,381,382"]
    TX["transient = transport, empty, or HTTP 5xx: back off, retry<br/>loom-client/src/client.rs:304,306<br/>loom-client/src/error.rs:121,123,124"]
    SO["ScaffoldOnly: content carries the verbatim marker<br/>loom-client/src/client.rs:393,394"]
    PR["PassthroughRefused: asked for passthrough, a mode was reported and it was not<br/>loom-client/src/client.rs:402,403,404,406"]
    CH --> BUD --> ATT
    ATT --> TR
    ATT --> TX
    ATT --> SO
    ATT --> PR
    TR --> ATT
    TX --> ATT
    INV["INVARIANT: the client detects a verbatim serve by a substring the facade writes,<br/>the two strings must stay identical<br/>loom-client/src/client.rs:25<br/>loom-facade/src/serving.rs:132,133"]
    SO --- INV
    SEE["see ES-06.11 and AB-24.10 for the five callers that use this client"]
    CH --- SEE
```

**What it shows.** A verbatim serve, a truncated reasoning answer and a silently scaffolded passthrough each return HTTP 200; the client turns each into a typed error or a retry instead (`loom-client/src/client.rs:300,308,314`).

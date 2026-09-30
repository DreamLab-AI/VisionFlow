---
id: AB-25
title: Ontology tools and governed writes
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2022, ADR-2023, ADR-2028, ADR-2054]
sources:
  - ../project/agentbox/mcp/servers/ontology-workingset.cjs
  - ../project/agentbox/scripts/ontology-condense-scheduler.mjs
  - ../project/agentbox/scripts/ontology-condense-refresh.sh
  - ../project/agentbox/config/hooks/ontology-monitor.cjs
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/flake.nix
  - ../project/agentbox/ontology/decision-layer.ttl
  - ../project/agentbox/ontology/decision-layer.ofn
  - ../project/agentbox/ontology/test_decision_layer.py
  - ../project/agentbox/services/agentbox-ops/src/bin/voyager-gate.rs
  - ../project/agentbox/services/agentbox-ops/src/voyager/mod.rs
  - ../project/agentbox/management-api/lib/uris.js
  - ../project/agentbox/management-api/lib/ontology-apply.js
verified_commit: 6a4ad132f2dc5ddaedd05c679fdd10066bf30a0f
---

Note: `ontology-bridge.js`, `ontology-propose.js`, `ontology-local.cjs`, `ontology-authoring-authority.js`
and `vault-frontmatter.js` — the whole MCP-tool authoring-authority surface this topic previously covered
(AB-25.1–.6, .10) — are deleted. The governed write door is now `vault edit --expect`, reached from a signed
kind-31403 forum decision through `applyOntologyDecision` (`management-api/lib/ontology-apply.js`); see
**AB-04.13** for the full sequence and **AB-14** for the 31402/31403 forum round-trip that precedes it.

## AB-25.7 Condensation index — the staleness scheduler

```mermaid
sequenceDiagram
    autonumber
    participant SCH as ontology-condense-scheduler.mjs<br/>agentbox/scripts/ontology-condense-scheduler.mjs:1
    participant GATE as manifest gates
    participant CORP as authored corpus<br/>VAULT_PAGES
    participant OUT as condense output
    participant REF as ontology-condense-refresh.sh<br/>agentbox/scripts/ontology-condense-refresh.sh:105
    participant IDX as ontology-index-build.js
    participant LLM as Loom facade<br/>agentbox/agentbox.toml:837 — see AB-24
    participant RV as RuVector ns ontology-classes<br/>see AB-20

    Note over SCH: supervised as [program:ontology-condense-scheduler] (agentbox/flake.nix:2032) — launched<br/>unconditionally, exits fast when its gate is off
    loop tick — schedule_interval_mins 60, jittered plus or minus 20 percent (agentbox.toml:850)
        SCH->>GATE: require BOTH ONTOLOGY_CONDENSE_ENABLED and ONTOLOGY_CONDENSE_SCHEDULE
        alt either off
            SCH-->>SCH: no-op — byte-identical-when-off until an operator opts in and the container reboots
        else both on
            SCH->>CORP: newest page mtime
            SCH->>OUT: last condense output mtime
            alt corpus newer OR output missing OR older than schedule_max_age_hours 24 (agentbox.toml:851)
                SCH->>REF: exec the refresh
                Note over REF: flock-serialised (ontology-condense-refresh.sh:57-59) — SKIPS if a refresh already holds<br/>the lock. Stages overwrite/resume deterministically, so the scheduler is idempotent
                REF->>IDX: parse the corpus into classes (ontology-condense-refresh.sh:105,111)
                loop each KG class, max_concurrency 2 (agentbox.toml:840)
                    REF->>LLM: POST /v1/chat/completions — one retrieval sentence + a synonym list<br/>(ontology-condense-refresh.sh:108)
                    Note over LLM: model qwen3.8-27B style openai (agentbox.toml:838-839). The model runs BEHIND the Loom<br/>facade so it is swappable with zero change here
                end
                REF->>OUT: PUSH Class-Summary cache
                REF->>RV: condensed store ns ontology-classes
            else fresh
                SCH-->>SCH: writes NOTHING — no LLM load, no cache churn
            end
        end
    end
    Note over SCH: the refresh had NO natural trigger — nothing re-ran it when GitHubSync/elevation rewrote<br/>the corpus, so the per-turn [ONTOLOGY] breadcrumb cache silently went stale. The<br/>"triggered incrementally on sync" claim was UNWIRED (:7-13)
    Note over SCH: fail-open, and a THIN wrapper — parse, cheap-LLM pass and cache fold all stay in the<br/>refresh script (:15-16). Mirrors the ruvector-aggregate-sweep house pattern, see AB-21
```

## AB-25.8 ontology-monitor — SessionEnd governed elevation

```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code SessionEnd
    participant MON as ontology-monitor.cjs<br/>agentbox/config/hooks/ontology-monitor.cjs:1
    participant CHILD as detached child<br/>ontology-monitor.cjs:244
    participant LOC as local ontology route<br/>VisionClaw-free
    participant ZAI as Z.AI GLM worker
    participant JSONL as ontology-proposals.jsonl<br/>AGENTBOX_STATE
    participant RELAY as forum broker gate<br/>Nostr relay
    participant H as Human approver

    CC->>MON: SessionEnd (stdin payload)
    MON->>MON: off = gatedOff() (:77-81)
    alt AGENTBOX_ONTOLOGY_MONITOR not 1, then no Z.AI key, then publish mode missing<br/>MANAGEMENT_API_KEY/NOSTR_RELAYS
        MON-->>CC: silent no-op — fail-open on any miss (:78-80)
    else all gates pass
        alt AGENTBOX_ONTOLOGY_MONITOR_FOREGROUND not set (:240-242)
            MON->>CHILD: spawn detached, own session, unref'd,<br/>stdin payload carried via env (:244-259)
            MON-->>CC: exit at once — SessionEnd never waits on the review (:269)
            CHILD->>CHILD: gather the session's concept-bearing work
            CHILD->>LOC: match ontology concepts via the local route
            CHILD->>ZAI: ONE review call
            ZAI-->>CHILD: proposals where the corpus looks stale, wrong or missing
            CHILD->>CHILD: drop proposals already in the seen-ledger (fingerprint dedup, :271-282)
            loop each fresh proposal
                CHILD->>CHILD: build an ACSP ActionRequest kind 31402
                alt AGENTBOX_ONTOLOGY_MONITOR_MODE = dryrun (the DEFAULT)
                    CHILD->>JSONL: append — NO relay egress (:198-206)
                else mode = publish
                    CHILD->>RELAY: publish the signed 31402 live (:208-233)
                    RELAY->>H: forum approval
                    H-->>RELAY: kind 31403 approval — see AB-14
                end
            end
        else foreground (tests, manual runs)
            MON->>MON: run the review inline, same steps as CHILD above
        end
    end
    Note over MON,CHILD: INVARIANT: nothing here signs the DECISION — a human approves via the forum.<br/>Read-pervasive, write-governed (:9-10)
    Note over MON: DETACHED, see comment :25-30. SessionEnd fires on exit and on /clear, and Claude Code<br/>waits for the hook - the review can take up to BUDGET_MS default 180000ms, see :38. So the hook<br/>process only checks the gates and hands off before returning, never blocking the session
```

## AB-25.9 Session working set and the drift guard

```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Held : note iri-or-slug
    note right of Empty
        ontology-workingset CLI — a session-scoped, IRI-keyed working set of
        compacted ontology-class digests carried ACROSS TURNS.
        Session defaults to AGENTBOX_SESSION_ID, bound to a beads epic or a
        session URN. agentbox/mcp/servers/ontology-workingset.cjs:1-18
    end note
    Held --> Held : get iri-or-slug
    Held --> Held : list
    Held --> Held : note another iri
    Held --> Revalidating : revalidate
    Revalidating --> Held : digests still match the live corpus
    Revalidating --> Drifted : corpus moved beneath the digest
    note right of Drifted
        The DRIFT GUARD. A digest compacted in an earlier turn can silently
        disagree with the corpus a later turn reasons over, so revalidate
        diffs the held digests against the live corpus rather than trusting
        carried state.
    end note
    Drifted --> Held : re-note the changed IRIs
    Held --> Empty : drop iri-or-slug
    Held --> Empty : clear
    Empty --> [*]
    note right of Held
        Resolution order is the repo lib first, then the baked /opt copy —
        the same pattern as continual-harness.cjs. ontology-local.js is now
        READ-ONLY (its write path was deleted with ontology-bridge.js; the
        governed write door is `vault edit`, see AB-04.13).
    end note
```

## AB-25.11 The Decision Layer OWL vocabulary — a formal schema with no runtime consumer

```mermaid
classDiagram
    class DecisionRecord {
        <<owl:Class, subClassOf prov:Activity>>
        decision-layer.ttl:43-44
    }
    class RelationalProperties {
        <<owl:ObjectProperty, domain+range DecisionRecord>>
        +caused ttl:60-63
        +precedentFor ttl:71-74
        +influenced ttl:85-88
    }
    class ExternalLinks {
        <<owl:ObjectProperty, range prov:Entity>>
        +consideredInput ttl:96-99
        +governedBy ttl:108-111
    }
    class TemporalWindow {
        <<owl:DatatypeProperty, xsd:dateTime>>
        +validFrom ttl:124-127
        +validTo ttl:136-139
    }
    class ProvEntity {
        <<prov:Entity>>
    }
    DecisionRecord --> DecisionRecord : RelationalProperties
    DecisionRecord --> ProvEntity : ExternalLinks
    DecisionRecord --> TemporalWindow : has
    note for DecisionRecord "INVARIANT (ADR-047 EL-safety, ttl:5-12): the vocabulary MUST stay in the OWL 2 EL<br/>profile so Whelk can classify it in polynomial time — no owl:inverseOf,<br/>owl:FunctionalProperty or owl:InverseFunctionalProperty anywhere in the file"
    note for RelationalProperties "caused and precedentFor are deliberately NOT transitive: reachability must stay<br/>query-derived, distinguishable from directly-asserted evidence"
    note for ExternalLinks "decision-layer.ofn:8-19 is a hand-kept functional-syntax PROJECTION of the same<br/>TTL for the horned-owl/Whelk loader — ofn:2 says 'keep in sync with the TTL'.<br/>No generator ties the two files together"
```

## AB-25.12 Where this vocabulary is actually exercised — and where it is not

```mermaid
flowchart TB
    subgraph schema["ontology/decision-layer.{ttl,ofn}"]
        TTL["decision-layer.ttl<br/>149 lines, OWL 2 EL, PROV-O aligned"]
        OFN["decision-layer.ofn<br/>39 lines, functional-syntax projection"]
    end
    subgraph tested["The only automated consumer found"]
        PYTEST["test_decision_layer.py<br/>rdflib.Graph#40;#41;.parse#40;TTL#41;<br/>ontology/test_decision_layer.py:23"]
        ELGUARD["EL-SAFETY GUARD: asserts NO triple in the graph uses<br/>owl:inverseOf / owl:FunctionalProperty / owl:InverseFunctionalProperty<br/>ontology/test_decision_layer.py:9-11"]
    end
    subgraph runtime["Runtime governance-write path (ADR-2109) — see AB-04.13"]
        APPLY["applyOntologyDecision — signed 31403 → vault edit --expect<br/>management-api/lib/ontology-apply.js:350"]
        URIS["management-api/lib/uris.js — the ONE URN minter for<br/>all 19 kinds, incl. decision (agentbox/CLAUDE.md URI/URN scheme)"]
    end
    TTL --> PYTEST
    PYTEST --> ELGUARD
    TTL -.->|"documents the SAME 'decision' concept CLAUDE.md/ADR-048 name<br/>that uris.js mints and ontology-apply.js applies"| APPLY
    OFN -.->|"no loader found in mcp/servers, services/ontology-tools,<br/>or services/agentbox-ops — grep-verified"| APPLY
    APPLY --> URIS
    subgraph notes["Invariants and drift"]
        direction TB
        N1["DIVERGENCE: decision-layer.ttl/.ofn are a formal, EL-profile-safe OWL vocabulary for<br/>agent decisions (ADR-047/048/049) with ONE automated consumer — a standalone Python<br/>structural test. No Rust or JS code under services/ or mcp/ parses, loads or reasons<br/>against either file (grep-verified across the whole repo). The 'governed ontology writes<br/>checked against formal OWL semantics' framing describes an aspiration this repo documents<br/>but does not itself execute — reasoning happens estate-side (VisionClaw Oxigraph/Whelk,<br/>see VC-20), not inside agentbox at write time"]
        N2["EXTERNAL: code-harness.ttl #40;the sibling OWL file in the same directory#41; IS referenced<br/>from Rust doc-comments at services/agentbox-ops/src/voyager/mod.rs:3 and<br/>services/agentbox-ops/src/bin/voyager-gate.rs:3 as the<br/>CONCEPTUAL SPEC the ported voyager-gate logic implements — see AB-28.9 (M9) for the<br/>voyager skill-library gate module. That reference is documentation, not a runtime TTL load"]
        N1 ~~~ N2
    end
```

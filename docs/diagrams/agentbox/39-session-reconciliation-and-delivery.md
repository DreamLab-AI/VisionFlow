---
id: AB-39
title: Session reconciliation and delivery boundaries
area: agentbox
governing:
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
adrs: [ADR-2153]
sources:
  - ../project/agentbox/config/session-plane/launch.json
  - ../project/agentbox/services/agentbox-sessions/src/plane.rs
  - ../project/agentbox/services/agentbox-sessions/src/reconcile.rs
  - ../project/agentbox/services/agentbox-sessions/src/status.rs
  - ../project/agentbox/services/agentbox-sessions/src/worktree.rs
  - ../project/agentbox/services/agentbox-sessions/src/main.rs
  - ../project/agentbox/services/agentbox-sessions/src/pick.rs
  - ../project/agentbox/services/agentbox-sessions/src/api.rs
  - ../project/agentbox/config/tab0-bridge/server.mjs
  - ../project/agentbox/config/session-plane/client.cjs
verified_commit: {agentbox: 59b1aca1cc3541b767872ecbc161c89e65d6ebca}
---

## For developers

The [session lifecycle audit](../../estate-review/2026-11-01-agentbox-session-lifecycle.md) retains seven private-tmux probes, a counterexample and exact source. These diagrams describe the native replacement service; Agent of Empires is retired. Passing defect probes do not establish fixes or provider acceptance. Reviewed at e458b308: the new legacy-worktree adoption path changes the registered `aoe-` to `abs-` migration case, drawn in AB-39.6. The old probes remain dated; no new provider or operator-client acceptance is claimed.

## For the business

A running pane, a submitted command and a completed agent task are different states. Users need confidence that configuration changes preserve work and that a retry will not submit the same instruction twice.

## AB-39.1 Restart checks use the desired worktree

```mermaid
flowchart TD
    PLAN["Desired plan and observed owned pane<br/>plane.rs:353,357"] --> REASON["Drift, death or fixable blocker<br/>reconcile.rs:70,72,74,75"]
    REASON --> WT{"Desired worktree enabled?<br/>reconcile.rs:82"}
    WT -->|yes| DIRTY["Inspect desired plan.dir<br/>Git error becomes zero changes<br/>plane.rs:361"]
    DIRTY -->|changes found| HOLD["Hold dirty worktree<br/>reconcile.rs:85"]
    DIRTY -->|zero| BACK["Check process-local restart allowance<br/>reconcile.rs:88,97"]
    WT -->|no| BACK
    BACK -->|allowed| START["Adopt a legacy worktree if eligible, then ensure destination,<br/>seed trust, kill any remaining old pane and start replacement<br/>plane.rs:303,304,306,309,313"]
```

**What it shows.** A switch to the shared directory skips the worktree check, while a Git inspection failure looks clean. **Why it matters.** Both probes replace a process whose old worktree contains work. Files remain. The dated probe found a prefix-only move refused by Git before killing the old pane. The current contract now explicitly adopts the registered legacy `aoe-` directory before ensuring `abs-` (AB-39.6); that earlier probe does not establish the behaviour of this added migration path. Foreign ownership and restart backoff still have separate holds.

## AB-39.2 Listing and sending classify API sessions differently

```mermaid
flowchart LR
    API["Owned API-created session<br/>plane.rs:498"] --> LIST["Status reconstructs tool launch rules<br/>plane.rs:410,414"]
    LIST --> BLOCK["Tool blocker detected<br/>status.rs:57,58"]
    API --> SEND["Send looks up manifest seed only<br/>plane.rs:441"]
    SEND --> COMMON["No launch: common blockers only<br/>plane.rs:378,380"]
    COMMON --> CHECK["Refuse only classified dead or blocked<br/>plane.rs:442,443,444"]
    CHECK -->|otherwise| INPUT["Submit text<br/>plane.rs:447"]
```

**What it shows.** An API tool's login pattern can appear in its blocked status row yet be absent from its send check. **Why it matters.** The native probe observes the dummy process consuming input. This is an authorised-caller guard inconsistency, not an ACL bypass.

## AB-39.3 Submission can precede failure and fallback

```mermaid
flowchart TD
    REQUEST["Socket send dispatch<br/>api.rs:91"] --> SUBMIT["tmux submits text<br/>plane.rs:447"]
    SUBMIT --> HOOK["Claude or Codex busy-hook write<br/>plane.rs:450,451"]
    HOOK -->|write fails| FAIL["API returns ok false<br/>api.rs:106"]
    FAIL --> CLIENT["Client promise rejects:<br/>error reply or timeout<br/>client.cjs:9,18"]
    TIME["Client timeout: completion unknown<br/>client.cjs:4,9"] --> CLIENT
    CLIENT --> FALLBACK["Bridge catches send error<br/>then uses tmux fallback<br/>server.mjs:168,173,179"]
    FALLBACK -.->|may submit again| SUBMIT
```

**What it shows.** The error response does not always mean input was withheld. **Why it matters.** The probe proves delivery followed by failure; duplicate bridge delivery is a source-supported consequence, not a live transmission performed by the audit. A delivery ID and staged receipt are needed before retry can be made reliable.

## AB-39.4 Controller and API worktree lifetimes diverge

```mermaid
flowchart TD
    LOAD["Manifest and contract loaded once<br/>plane.rs:184,185"] --> LOOP["Reconcile stored seeds only<br/>plane.rs:345"]
    CLI["Direct CLI builds another Plane<br/>main.rs:96,100,120,121,122"] --> SEPARATE["Independent lock and restart map<br/>plane.rs:187"]
    LOOP --> REMOVE["Removed seed after reload:<br/>owned survivor still listed<br/>plane.rs:409,419"]
    REMOVE --> REFUSE["Close requires origin api<br/>plane.rs:510"]
    CREATE["API request includes worktree intent<br/>plane.rs:493"] --> STATUS["Non-seed status says worktree false<br/>plane.rs:419"]
    CREATE --> CLOSE["Close rebuilds shared-directory plan,<br/>emits close and kills pane<br/>plane.rs:511,513,515"]
    CLOSE --> LEFT["Current API worktree remains;<br/>legacy reaper targets old root<br/>plane.rs:563,564,566"]
```

**What it shows.** Restarting a controller reloads desired state without defining every surviving-session transition. API close also lacks a current-worktree cleanup contract. **Why it matters.** Orphaned work needs a visible owner and deliberate archive/cleanup policy. Managed configuration uses rebuild; this is not a claim that hot reload was promised.

## AB-39.5 Idle and explicit client targeting have narrower guarantees

```mermaid
flowchart TD
    OBS["Observed session and caller context"] --> STATE["Dead, blocker, then hook checks<br/>status.rs:54,57,60"]
    STATE --> READY["Ready pattern, otherwise age<br/>status.rs:65,68"]
    READY --> IDLE["After settle time: idle<br/>even without a response<br/>status.rs:71"]
    OBS --> PANE["Inherited TMUX_PANE selects session<br/>pick.rs:24,25"]
    PANE --> CLIENT["Most active attached client<br/>within that session<br/>pick.rs:17"]
    CLIENT --> SWITCH["Switch chosen client explicitly<br/>pick.rs:71"]
```

**What it shows.** Idle can be a time-based fallback. Explicitly naming a selected client does not prove that it is the invoking client. **Why it matters.** Provider round trips and a two-client picker test must establish the stronger claims. The audit did not attach or switch an operator client.


## AB-39.6 Retired worktree prefixes are adopted before a fresh worktree is ensured

```mermaid
flowchart TD
    CONTRACT["Current abs- prefix, legacy aoe- prefix<br/>config/session-plane/launch.json:3-5"] --> START
    START["Start a worktree seed: adopt before ensure<br/>plane.rs:302-304"] --> CANDIDATE{"Destination absent and legacy directory exists?<br/>plane.rs:273-275"}
    CANDIDATE -->|yes| OTHER{"Another session inside canonical legacy directory?<br/>plane.rs:277-281"}
    OTHER -->|yes| HOLD["Refuse this start, retry next pass<br/>plane.rs:281"]
    OTHER -->|no| OLD["Close the seed's own old pane if it is there<br/>plane.rs:283-286"]
    OLD --> REGISTERED{"Source registered to this repository<br/>and target still absent?<br/>worktree.rs:117-125"}
    REGISTERED -->|yes| MOVE["Rename, git worktree repair, relink submodules<br/>worktree.rs:130-134"]
    MOVE --> SUB["Rewrite core.worktree under the new directory,<br/>including nested and slash-named modules<br/>worktree.rs:165-196"]
    REGISTERED -->|no| ENSURE
    CANDIDATE -->|no| ENSURE
    SUB --> ENSURE["Ensure destination, seed trust and start session<br/>plane.rs:304,306,313"]
    MOVE -.-> ERROR["Adoption error is logged as a note, then ensure runs<br/>plane.rs:288-291"]
    ERROR --> ENSURE
```

**What it shows.** The migration retains the registered worktree and its files, repairs Git's location pointers and defers when another session occupies the old directory. **Why it is this way.** Git's worktree move refuses initialised submodules; the helper renames the directory and repairs both the worktree and submodule pointers. This is the implemented migration path, not a new live migration performed by this review. The existing API delivery and lifetime findings above remain separate.

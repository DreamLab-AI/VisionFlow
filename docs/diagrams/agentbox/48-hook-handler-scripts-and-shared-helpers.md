---
id: AB-48
title: Hook handler scripts — adapter, trust-seed, brain-ground, publishers, trajectory recorder and shared helpers
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2015, ADR-2026, ADR-2068]
sources:
  - ../project/agentbox/config/hooks/claude-flow-hook-adapter.cjs
  - ../project/agentbox/config/hooks/trust-seed.cjs
  - ../project/agentbox/config/hooks/nostr-live-mirror.cjs
  - ../project/agentbox/config/hooks/project-tracking-publish.cjs
  - ../project/agentbox/config/hooks/ontology-monitor.cjs
  - ../project/agentbox/config/hooks/ruvnet-brain-ground.cjs
  - ../project/agentbox/config/hooks/trajectory-recorder.cjs
  - ../project/agentbox/config/hooks/dream-inbox-surface.cjs
  - ../project/agentbox/config/hooks/lib/trajectory-util.cjs
  - ../project/agentbox/config/hooks/lib/egress-policy.cjs
  - ../project/agentbox/config/hooks/lib/failure-jev.cjs
  - ../project/agentbox/config/hooks/lib/ontology-proposal-jev.cjs
  - ../project/agentbox/config/hooks/activity-sink.cjs
  - ../project/agentbox/config/hooks/README.md
  - ../project/agentbox/services/agentbox-manifest/src/stacks.rs
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/mcp/servers/lib/ontology-local.js
  - ../project/agentbox/flake.nix
  - ../project/agentbox/mcp/servers/lib/ontology-push.js
  - ../project/agentbox/config/hooks/skill-route.cjs
  - ../project/agentbox/config/hooks/lib/hook-output.cjs
verified_commit: 59b1aca1cc3541b767872ecbc161c89e65d6ebca
aliases:
  AB-08.8: AB-48.1
  AB-08.10: AB-48.2
  AB-08.11: AB-48.3
  AB-08.12: AB-48.4
  AB-08.13: AB-48.5
  AB-08.14: AB-48.6
---
Split from AB-08 on 2026-11-23: the individual handler scripts and their shared helpers (formerly AB-08.8 and AB-08.10 to AB-08.14). Registration, the per-event pipelines and the registration mechanisms stay in AB-08. Old panel IDs resolve through the `aliases:` map above; the stamp is AB-08's, and the panels were not re-read.

## AB-48.1 claude-flow-hook-adapter.cjs — stdin-to-CLI translation (per-profile stacks only)
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core<br/>(profile stack, e.g. claude-core)
    participant AD as claude-flow-hook-adapter.cjs<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:106
    participant CLI as claude-flow hooks CLI<br/>AGENTBOX_FLOW_BIN env, default claude-flow (:32)
    participant OP as ontology-push.js<br/>mcp/servers/lib (optional)
    participant M as Model context

    Note over CC,AD: registered by stacks.rs learning_hooks() (stacks.rs:27-63)<br/>into workspace/profiles/#60;name#62;/.claude/settings.json - NOT this session's root settings.json
    CC->>AD: stdin JSON, argv[2]=action (:106-109)
    AD->>AD: parsePayload(readStdin()) - malformed/empty JSON -#62; {} (:49-65)
    AD->>AD: extract tool_input.file_path, tool_input.command, prompt (:109-113)
    alt action=route
        Note over AD,CLI: DRIFT: no registration site sends action=route any more, stacks.rs learning_hooks has no<br/>UserPromptSubmit entry (stacks.rs:31-33), so the route branch runs only when invoked by hand
        AD->>CLI: spawnSync claude-flow hooks route --task PROMPT t=12000 (:117,67-73)
        CLI-->>AD: stdout (latency/alternatives dump, WASM-fallback banner possible)
        AD->>AD: filter stdout by ROUTE_SIGNAL regex (INFO|INTELLIGENCE|Agent:|Matched Pattern) (:46,74-81)
        AD-->>M: filtered lines only - full dump would pollute every turn (:41-46)
        opt ONTOLOGY_INJECT set and prompt non-empty
            AD->>OP: require ontology-push.js, getOntologyBreadcrumb(prompt) (:87-104)
            OP-->>AD: breadcrumb line or throws
            AD-->>M: [ONTOLOGY] breadcrumb appended, fail-open on any require/call error
        end
    else action=pre-edit (file present)
        AD->>CLI: hooks pre-edit --file FILE t=5000 (:120-122)
    else action=post-edit (file present)
        AD->>CLI: hooks post-edit --file FILE t=10000 (:123-125)
    else action=pre-command (command present)
        AD->>CLI: hooks pre-command --command CMD t=5000 (:126-128)
    else action=post-command (command present)
        AD->>CLI: hooks post-command --command CMD t=5000 (:129-131)
        Note over AD,CLI: pre-edit/post-edit/pre-command/post-command all call run() with NO signal<br/>arg, so the (signal && ...) stdout gate never fires for them - only<br/>route and session-restore forward filtered CLI stdout (:74-81)
    else action=session-restore
        AD->>CLI: hooks session-restore t=15000 (:132-134)
        CLI-->>AD: stdout filtered by RESTORE_SIGNAL (:47,74-81)
        AD-->>M: filtered lines
    else action=session-end
        AD->>CLI: hooks session-end t=10000 (:135-137)
    else unknown action
        AD->>AD: no-op, never signals error (:138-140)
    end
    Note over AD: TRANSFORMERS_CACHE/HF_HOME forced to writable tmpfs path (:33-39,71)<br/>else @xenova/transformers ENOENT#39;s against the read-only Nix store
    Note over AD: whole main() wrapped in try/catch, process.exit(0) unconditional (:144-149)<br/>INVARIANT: adapter holds NO learning state, intelligence lives in the CLI backend<br/>(header :13-17 cites stale ADR-015, pre-2026-consolidation id)
```

**Drift:** the adapter still handles `action=route`, but no registration site sends it any more: `learning_hooks` has no `UserPromptSubmit` entry (`../project/agentbox/services/agentbox-manifest/src/stacks.rs:31-33`), so the route branch and its ontology breadcrumb run only when invoked by hand.
## AB-48.2 trust-seed.cjs — a BOOT-TIME script now, not a hook; folder-trust, worktree discovery and per-profile stores
```mermaid
sequenceDiagram
    autonumber
    participant EP as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:1818
    participant TSD as trust-seed.cjs main()<br/>agentbox/config/hooks/trust-seed.cjs:88
    participant FS as filesystem
    participant CFG as each Claude config store<br/>trust-seed.cjs:91

    Note over EP,TSD: runs ONCE at boot, gated AGENTBOX_TRUST_SEED!=0 (entrypoint-unified.sh:1819), never as a<br/>SessionStart hook - it walked ~1,170 paths/session (avg 3.2s) and raced Claude<br/>Code's own ~/.claude.json writes (entrypoint-unified.sh:1811-1816). Callable by hand post-boot for a<br/>new worktree: node trust-seed.cjs #60;path#62; (entrypoint-unified.sh:1816)
    EP->>TSD: node trust-seed.cjs [--depth N] [--dry-run] [extra-path...] (entrypoint-unified.sh:1820, parseArgs trust-seed.cjs:38-47)
    TSD->>FS: findRepos(WORKSPACE, depth=5, []) recursive walk (trust-seed.cjs:65-76)
    loop each dir entry, skip node_modules|target|.git|dist|build|.tmp|.cache|.venv|venv (trust-seed.cjs:36,70)
        FS->>TSD: isProjectDir(p) = isGitRoot(p) OR has Cargo.toml|package.json|pyproject.toml|flake.nix|justfile (trust-seed.cjs:49-63)
        opt isProjectDir true
            TSD->>TSD: acc.push(p) (trust-seed.cjs:72)
        end
        TSD->>FS: recurse findRepos(p, depth-1, acc) (trust-seed.cjs:73)
    end
    TSD->>TSD: targets = Set(WORKSPACE, ...repos, ...extra argv resolved) (trust-seed.cjs:90)
    TSD->>FS: profileStores() lists WORKSPACE/profiles/slug/.claude/.claude.json, keeping only stores that already exist (trust-seed.cjs:78-86)
    Note over TSD,CFG: a harness-wrapper profile that has never run has no store and is skipped (trust-seed.cjs:12-13).<br/>Before this change only ~/.claude.json was seeded, and a profile reads only its own CLAUDE_CONFIG_DIR store (trust-seed.cjs:9-12)
    loop each store in [~/.claude.json, ...profile stores] (trust-seed.cjs:91)
        TSD->>CFG: seedStore(store, targets, args) JSON.parse(readFileSync(store)) - ENOENT tolerated, other errors skip this store (trust-seed.cjs:100-102)
        loop each target dir
            alt cfg.projects[dir] already hasTrustDialogAccepted AND hasCompletedProjectOnboarding
                TSD->>TSD: skip, no change (trust-seed.cjs:108)
            else
                TSD->>TSD: cfg.projects[dir] = {...entry, hasTrustDialogAccepted:true, hasCompletedProjectOnboarding:true} (trust-seed.cjs:109-110)
            end
        end
        alt --dry-run OR added===0
            TSD-->>EP: stderr summary line only, store untouched (trust-seed.cjs:112-115)
        else
            TSD->>FS: copyFileSync store to WORKSPACE/.agentbox/NAME.pre-trust-seed backup, NAME claude.json or slug.claude.json (trust-seed.cjs:119-124)
            TSD->>CFG: writeFileSync store, JSON.stringify(cfg, null, 2) + newline (trust-seed.cjs:125)
            TSD-->>EP: stderr summary line (store, N checked, M newly trusted) (trust-seed.cjs:126)
        end
    end
    Note over TSD: progress goes to STDERR ONLY (trust-seed.cjs:25) - a hook's stdout is injected into<br/>model context, and this is deliberately not a hook any more
    Note over TSD: INVARIANT: never removes/overwrites OTHER per-project keys in cfg.projects (trust-seed.cjs:18)<br/>fail-open per store: seedStore errors are caught and reported, the loop continues (trust-seed.cjs:92-94), and a top-level try/catch backs it (trust-seed.cjs:129)
```
## AB-48.3 ruvnet-brain-ground.cjs (rewritten trigger logic) and ontology-monitor.cjs (now detached)
```mermaid
sequenceDiagram
    autonumber
    participant U as UserPromptSubmit
    participant RBG as ruvnet-brain-ground.cjs<br/>agentbox/config/hooks/ruvnet-brain-ground.cjs:115
    participant HO as lib/hook-output.cjs<br/>agentbox/config/hooks/lib/hook-output.cjs:32
    participant SE as SessionEnd
    participant OMP as ontology-monitor.cjs parent<br/>agentbox/config/hooks/ontology-monitor.cjs:271
    participant OMC as ontology-monitor.cjs DETACHED CHILD<br/>agentbox/config/hooks/ontology-monitor.cjs:258
    participant ONT as local ontology route<br/>mcp/servers/lib/ontology-local.js
    participant ZAI as claude-zai CLI<br/>AGENTBOX_ZAI_BIN, ZAI_URL (:177-183)
    participant FB as forum broker gate<br/>NostrBridge kind 31402

    rect rgb(235,245,255)
    U->>RBG: stdin JSON, hook.userInput|hook.prompt (:120-121)
    RBG->>RBG: analyse(prompt) splits into sentences (:85-87)
    RBG->>RBG: DISTINCTIVE name (ruvnet, qudag, ruv-fann...) fires alone,<br/>an ESTATE name (ruvector, claude-flow, ruflo...) needs an<br/>UPSTREAM_CUES match in the SAME sentence (:36-47,74-77,96-98)
    RBG->>RBG: CLASSICAL_SUBS (pinecone, pgvector, langchain...) needs<br/>SELECTION intent (use, migrate, compare...) in the same sentence (:59-70,100-109)
    alt any pattern matched
        RBG->>HO: emitContext(ctx) — GROUNDING and/or up to 2 REDIRECT lines,<br/>capped at MAX_CONTEXT_CHARS=1200 (:34,79-82,111-112,121-124)
        HO-->>U: hookSpecificOutput.additionalContext (:22-26,32-41)
    else no match
        RBG-->>U: no stdout — emitContext resolves false, main returns (:121-122)
    end
    Note over RBG: fail-open: any JSON.parse error in main -#62; return, no process.exit(1) anywhere (:120)
    end
    rect rgb(255,240,240)
    SE->>OMP: stdin JSON payload, gated by AGENTBOX_ONTOLOGY_MONITOR=1 (:78,272-273)
    alt gatedOff(): master switch off, no ZAI key, or publish mode missing<br/>MANAGEMENT_API_KEY/NOSTR_RELAYS
        OMP-->>SE: log no-op, exit 0 (:77-82,272-273)
    else gated on
        OMP->>OMC: NOT runInForeground: spawn detached child, stdio ignored except a<br/>logfile, unref, exit 0 AT ONCE — SessionEnd never waits (:249-268, exit :274)
        Note over OMC: BUDGET_MS=180000 wall clock, read from AGENTBOX_ONTOLOGY_MONITOR_BUDGET_MS (:38,41)
        OMC->>OMC: gatherWork(payload) - git status --porcelain + transcript tail 12k chars (:93-119)
        OMC->>ONT: createLocalOntology().classList(limit:100000) (:125-129)
        OMC->>OMC: matchConcepts() word-boundary label match, MAX_CONCEPTS=8 (:39,122-151)
        alt no concepts matched
            OMC-->>SE: log no-op, exit 0 (:282)
        else
            OMC->>ZAI: spawnCli claude-zai -p PROMPT, STRICT JSON proposals[] (:154-195)
            ZAI-->>OMC: {proposals: [{iri,label,kind,title,summary,rationale}]}, MAX_PROPOSALS=5
            OMC->>OMC: fingerprint(p) = sha256(iri|kind|normalised summary), filter against seen ledger (:57-60,287-288)
            alt fresh.length===0
                OMC-->>SE: log all already seen, exit 0 (:289)
            else
                OMC->>OMC: Jev option, off unless AGENTBOX_JEV_ONTOLOGY_PROPOSALS=1:<br/>lib/ontology-proposal-jev.cjs verify(fresh) annotates each proposal,<br/>fail-open to the same array (:290-298, flag lib/ontology-proposal-jev.cjs:41)
                OMC->>OMC: verdicts appended to ontology-proposals-jev.jsonl,<br/>never part of the signed 31402 (:299, stageVerdicts :205-212)
                OMC->>OMC: held only when AGENTBOX_JEV_PROPOSAL_HOLD_BELOW is set in 0 to 0.5<br/>and supported is below it, held ones go to the local review queue<br/>(:300, lib/ontology-proposal-jev.cjs:175-186)
            end
            alt emit empty after the Jev hold
                OMC-->>SE: mark fresh seen, exit 0 (:302)
            else MODE=publish (AGENTBOX_ONTOLOGY_MONITOR_MODE, default dryrun)
                OMC->>FB: buildActionRequest(panelId, category:ontology, kind 31402) per proposal (:216-243)
                FB-->>OMC: published, NIP-33 d-tag = panelIdFor(p) so a repeat REPLACES the prior panel (:70-74)
            else MODE=dryrun
                OMC->>OMC: stageLocally(emit) appends $AGENTBOX_STATE/ontology-proposals.jsonl (:195-203, call :307)
            end
            OMC->>OMC: saveSeen(seen) ledger capped to last 5000 fingerprints (:64-69,310-311)
        end
    end
    Note over OMC: human approval via 31403 happens OUTSIDE this hook - it only proposes (PRD-014 governed elevation)
    end
```
## AB-48.4 project-tracking-publish.cjs and dream-inbox-surface.cjs — DREAM-2115: a reminder pointer, not a relay
```mermaid
sequenceDiagram
    autonumber
    participant CALLER as management API<br/>POST /v1/projects/:id/publish (NOT a Claude Code hook trigger)
    participant PTP as project-tracking-publish.cjs<br/>agentbox/config/hooks/project-tracking-publish.cjs:205
    participant MAPI as management-api<br/>GET /v1/projects on port 9090 (:136-180)
    participant NPB as nostr-pod-bridge track<br/>spawnSync binary (:186-200)
    participant U as UserPromptSubmit
    participant DI as dream-inbox-surface.cjs<br/>agentbox/config/hooks/dream-inbox-surface.cjs:34
    participant INBOX as dream-inbox.json + .surfaced stamp<br/>DREAM_INBOX_PATH override, dream-inbox-surface.cjs:28-29
    participant M as Model context

    rect rgb(245,235,255)
    Note over CALLER,NPB: RESOLVED ADR-2068 (2026-09-05): not a divergence — this is a CLI, not a hook.<br/>The management API spawns it from the /v1/projects publish route (routes/projects.js lines 28 and 322).<br/>config/hooks/README.md section 3 names every file in config/hooks/ that is a CLI or helper rather than a hook
    CALLER->>PTP: node project-tracking-publish.cjs, optional ProjectTrackingDigest on stdin (:205,213-228)
    alt AGENTBOX_PROJECT_TRACKING_PUBLISH===0, or bridge secrets absent (project-tracking-publish.cjs:60-68, project-tracking-publish.cjs:206-209)
        PTP-->>CALLER: return 0, silent no-op
    else
        alt stdin carries a valid digest (project_id present)
            PTP->>PTP: use verbatim (:218-219)
        else stdin is a TrackedProject or empty
            PTP->>MAPI: GET /v1/projects, Authorization Bearer MANAGEMENT_API_KEY (:135-146)
            MAPI-->>PTP: project list, mapped via toDigest() (:92-110, :231-235)
        end
        loop each digest
            PTP->>NPB: spawnSync nostr-pod-bridge track, digest JSON on stdin, t=30000 (:186-200)
            NPB-->>PTP: kind-30841 addressable event, d-tag=project slug, dual-write pod inbox
        end
        PTP-->>CALLER: log published N/total, return 0 (:241-245)
    end
    Note over PTP: guard setTimeout DEADLINE_MS=30000+1500 force-exits process (:248-249)
    end
    rect rgb(235,255,240)
    Note over DI: DRIFT resolved (ADR-2115): no longer relays item text. The nightly engine now<br/>publishes each item as a forum governance case, this hook adds ONE pointer line<br/>with a count, dream-inbox-surface.cjs:2-10
    U->>DI: stdin JSON, parse prompt and skip MACHINE_TURN turns (task-notification,<br/>agent-message, system-reminder) so the pointer lands on a real user turn (:40-42)
    DI->>INBOX: JSON.parse(readFileSync(INBOX)), count status===open only (:43-46)
    alt open===0
        DI-->>M: {} (no hookSpecificOutput) (:46,68-70)
    else
        DI->>INBOX: read STAMP (INBOX.surfaced), rate-limit to once per RESURFACE_HOURS=4 (:48-51)
        alt within RESURFACE_HOURS of the last stamp
            DI-->>M: {} (:51,68-70)
        else
            DI-->>M: hookSpecificOutput.additionalContext = [DREAM] N decision(s) await you in<br/>the forum governance panel, DREAM_GOVERNANCE_URL (:52-56)
            DI->>INBOX: writeFileSync(STAMP, now) ONLY after the write callback confirms flush (:57-61)
        end
    end
    Note over DI: fail-open: any error (missing file, bad JSON) -#62; exit() path returning {} (:62-64,68-70)
    end
```
## AB-48.5 trajectory-recorder.cjs — Stop/SubagentStop boundary, persisted through the Management API
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core<br/>Stop or SubagentStop
    participant TR as trajectory-recorder.cjs<br/>agentbox/config/hooks/trajectory-recorder.cjs:570
    participant UT as trajectory-util.cjs<br/>agentbox/config/hooks/lib/trajectory-util.cjs:1
    participant STASH as os.tmpdir stash<br/>agentbox-traj-#60;sha12#62;.json<br/>trajectory-recorder.cjs:220-222
    participant MAPI as management-api on 127.0.0.1, port MANAGEMENT_API_PORT or 9090<br/>trajectory-recorder.cjs:283-288

    CC->>TR: node trajectory-recorder.cjs #60;event#62;, stdin JSON (:570-577)
    Note over TR: ADR-2015: transcript-driven because PostToolUse does NOT fire<br/>for non-zero-exit Bash calls on this build (:26-30)
    Note over TR: ADR-2164: the hook holds no database client, no connection string and no gate variable,<br/>it POSTs computed rows to the API, which owns the trajectory tables (:13-17, :44-45)
    alt event is neither Stop nor SubagentStop
        TR-->>CC: return 0 (:582)
    else Stop or SubagentStop (:580-581)
        TR->>TR: no transcript_path, or the transcript cannot be read: return 0, fail-open (:407-411)
        TR->>STASH: readStash(session) - processedLines watermark + ctcPending queue (:414-415 and :224-230)
        TR->>UT: scanTranscript(lines, fromLine) grades each Bash tool_use by tool_result.is_error (:321-395)
        loop each Bash tool_result after the watermark
            TR->>UT: gradeResult(is_error,stderr,interrupted) - no grade means nothing is written (:365-366)
            TR->>UT: redact(command) - I10 fail-closed, a null command skips the step entirely (:370-371)
        end
        alt no new steps
            TR->>STASH: advance the watermark, nothing was lost (:434)
        else steps to persist
            TR->>UT: Jev option, off unless AGENTBOX_JEV_FAILURE_TAXONOMY=1: jevFailureTags pre-pass refines only failures the taxonomy left unmapped (:115-123, call :441)
            Note over TR: lib/failure-jev.cjs bounds the pre-pass to AGENTBOX_JEV_FAILURE_MAX_CALLS requests, default 3, under one deadline hard-capped at 4000 ms (lib/failure-jev.cjs:24-27)<br/>and takes the proxy and TypeSafe unless AGENTBOX_JEV_FAILURE_LOCAL_ONLY=1, off by default (lib/failure-jev.cjs:13-16, failure-jev.cjs:109), a miss keeps the base tag, and failure_jev is written to the step result only when the judge answered (trajectory-recorder.cjs:472-475)
            TR->>STASH: ctcEmits = stash.ctcPending FIFO-drained first, bound CTC_QUEUE_MAX=2000 (:446-448, :176)
            TR->>TR: build each row, id = trajectory id plus sha12(tool_use_id), step_order and quality from the stash counters (:459-491)
            TR->>TR: push ctcEmitBodyFromStep() onto ctcEmits while below CTC_QUEUE_MAX (:496-499)
            TR->>MAPI: postTrajectory(trajectory, steps, rollup) to POST /v1/agent-events/trajectory, Bearer MANAGEMENT_API_KEY or TASK_ACTION_TOKEN, 5 s timeout (:508-526, :276-295)
            alt 2xx
                TR->>STASH: watermark and cumulative counters advance ONLY now (:529-536)
            else non-2xx, timeout, unreachable or no credential
                TR->>TR: watermark NOT advanced, lines retried on the next Stop, only the carried backlog is emitted (:537-543)
            end
            TR->>MAPI: emitCtcStepsBestEffort(toEmit) POST /v1/agent-events/emit, CTC_EMIT_CAP=200/invocation, AGENTBOX_CTC_EMIT=0 turns it off (:183-194 and :548)
            TR->>TR: returns {attempted, deferred} - surplus beyond the cap is NOT dropped here (:193)
            TR->>STASH: cur.ctcPending = deferred (bounded CTC_QUEUE_MAX), cur.ctcPendingOverflow accumulates true drops (:558-560)
            Note over TR: overflow beyond CTC_QUEUE_MAX IS dropped and logged as INCOMPLETE (:554-556)<br/>this is the only stash write outside the persistence path (:549-551)
        end
    end
    Note over TR: guard setTimeout 8000ms force-exits the process regardless of HTTP state (:587)
    Note over TR: DRIFT: the Jev pre-pass comments still say it runs before the database connection opens<br/>(:110, lib/failure-jev.cjs:24), but since ADR-2164 the hook opens no database connection (failure-jev.cjs:15-17)
    Note over TR: see AB-21 for the learning loop this feeds (ReasoningBank / SONA consumers of trajectories)
```

**Drift:** the Jev pre-pass comments still say it runs before the database connection opens (`../project/agentbox/config/hooks/trajectory-recorder.cjs:110`, `../project/agentbox/config/hooks/lib/failure-jev.cjs:24`), but since ADR-2164 the hook holds no database client and persists only through `POST /v1/agent-events/trajectory` (`../project/agentbox/config/hooks/trajectory-recorder.cjs:13-17`).
## AB-48.6 lib/ shared helpers — consumers
```mermaid
flowchart TD
    LIBDIR["agentbox/config/hooks/lib/<br/>hooks/README.md:71 documents three; hook-output.cjs, jev-option.cjs, failure-jev.cjs, ontology-proposal-jev.cjs,<br/>operator-key.cjs and routing-labels.cjs are undocumented there"]
    UT["trajectory-util.cjs<br/>agentbox/config/hooks/lib/trajectory-util.cjs:1"]
    EG["egress-policy.cjs<br/>agentbox/config/hooks/lib/egress-policy.cjs:1<br/>JS half of the ADR-2026 content-egress policy egress-policy.cjs:3-10"]
    HO["hook-output.cjs<br/>agentbox/config/hooks/lib/hook-output.cjs:1<br/>the ONE stdout shape Claude Code honours hook-output.cjs:3-19"]
    LIBDIR --> UT
    LIBDIR --> EG
    LIBDIR --> HO
    F1["sha12(s) trajectory-util.cjs:18<br/>content-addressed step/trajectory ids"]
    F2["commandPattern(command) trajectory-util.cjs:38<br/>+ hasResidualSecret(text) trajectory-util.cjs:180"]
    F3["redact(command) trajectory-util.cjs:196<br/>I10 fail-closed redaction gate"]
    F4["deriveOutcome(toolResponse) trajectory-util.cjs:227<br/>+ gradeResult(isError,stderr,interrupted) trajectory-util.cjs:285"]
    F5["tokenCountOf(usage) trajectory-util.cjs:309<br/>+ usageIdentityOf(rec) trajectory-util.cjs:341"]
    F6["handoffIdFrom(env,fallbackId) trajectory-util.cjs:367<br/>CTC chain-correlation id"]
    F7["ctcEmitBodyFromStep(step,opts) trajectory-util.cjs:395<br/>builds /v1/agent-events/emit body"]
    UT --> F1 --> F2 --> F3 --> F4 --> F5 --> F6 --> F7
    G1["OUTCOME skipped|attempted|accepted|failed<br/>egress-policy.cjs:29-34"]
    G2["recipientAllowed(recipient) :59-68<br/>64-hex grammar + mandatory non-empty AGENTBOX_MIRROR_RECIPIENTS allowlist :46-53"]
    G3["redactForEgress(text) :115-124<br/>null return means fail-closed SKIP"]
    G4["egressDecision(pathId, opts) :136-162<br/>global, per-path and redaction-disabled switches"]
    EG --> G1 & G2 & G3 & G4
    H1["contextPayload() :22-26 — the ONLY honoured shape:<br/>hookSpecificOutput.hookEventName/.additionalContext"]
    H2["emitContext() hook-output.cjs:32-44 — resolves true only once the write flushes"]
    HO --> H1 & H2
    TR["trajectory-recorder.cjs<br/>agentbox/config/hooks/trajectory-recorder.cjs:54<br/>ONLY consumer of trajectory-util.cjs"]
    NM["nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:46<br/>+ activity-sink.cjs:315 redacts with it too — TWO consumers of egress-policy.cjs"]
    RBG["ruvnet-brain-ground.cjs:123<br/>+ hooks/skill-route.cjs:28 — BOTH consumers of hook-output.cjs"]
    F7 --> TR
    G1 & G2 & G3 & G4 --> NM
    H1 & H2 --> RBG
    PAIR["INVARIANT ADR-2026: the Rust twin services/nostr-pod-bridge/src/egress_policy.rs<br/>must satisfy the SAME fixture tests/fixtures/egress-redaction.v1.json<br/>egress-policy.cjs:6-10 — a shared comment is not a shared contract"]
    EG -.-> PAIR
    OTHER["ontology-monitor requires only lib/ontology-proposal-jev.cjs (ontology-monitor.cjs:296) and<br/>trajectory-recorder also lazily requires lib/failure-jev.cjs (trajectory-recorder.cjs:121).<br/>These import no lib/ module: claude-flow-hook-adapter, trust-seed (a boot-time script, see AB-48.2),<br/>dream-inbox-surface (composes its own JSON, does not call hook-output.cjs),<br/>project-tracking-publish, fleet-session-start and fleet-tab-name"]
    LIBDIR -.-> OTHER
```

Session-mirror egress: see AB-13.9.

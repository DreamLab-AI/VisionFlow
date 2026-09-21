---
id: AB-22
title: Skills estate — discovery, lint gate, the live router, harness and colloquy MCP bridges
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2020, ADR-2021, ADR-2028, ADR-2056, ADR-2057, ADR-2083, ADR-2085, ADR-2086, ADR-2089, ADR-2090, ADR-2091, ADR-2092]
sources:
  - ../project/agentbox/skills/lint-skills.sh
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/CLAUDE.md
  - ../project/agentbox/skills/lint-skills.mjs
  - ../project/agentbox/skills/SKILL-DIRECTORY.md
  - ../project/agentbox/skills/skill-router/SKILL.md
  - ../project/agentbox/skills/skill-router/references/routing-table.md
  - ../project/agentbox/.github/workflows/invariants.yml
  - ../project/agentbox/skills/tree-search-coder/SKILL.md
  - ../project/agentbox/skills/tree-search-coder/references/algorithm.md
  - ../project/agentbox/scripts/skill-count-check.js
  - ../project/agentbox/tests/fixtures/skill-router-prompts.json
  - ../project/agentbox/mcp/servers/harness-bridge.js
  - ../project/agentbox/mcp/servers/continual-harness.cjs
  - ../project/agentbox/mcp/servers/lib/continual-harness.js
  - ../project/agentbox/skills/mcp.json
  - ../project/agentbox/mcp/mcp.json
  - ../project/agentbox/scripts/project-mcp-servers.mjs
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/mcp/servers/lib/vault-frontmatter.js
  - ../project/agentbox/scripts/ci/check-no-logseq-paths.sh
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/docs/adr/ADR-2020-capability-gating.md
  - ../project/agentbox/docs/adr/ADR-2021-skills-jit-context-lint.md
  - ../project/agentbox/docs/adr/ADR-2028-vault-manifest-path-authority.md
  - ../project/agentbox/services/agentbox-ops/src/cost_cap/mod.rs
  - ../project/agentbox/services/agentbox-ops/src/bin/tree-search-cap.rs
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/flake.nix
  - ../project/agentbox/mcp/servers/mcp-ws-relay.js
  - ../project/agentbox/skills/registered-skills.txt
  - ../project/agentbox/skills/codex-registered-skills.txt
  - ../project/agentbox/skills/gen-routing-table.mjs
  - ../project/agentbox/skills/skill-router/references/section-map.json
  - ../project/agentbox/config/hooks/skill-route.cjs
  - ../project/agentbox/config/hooks/lib/skill-route.cjs
  - ../project/agentbox/agents/registered-agents.txt
  - ../project/agentbox/scripts/reconcile-agents.sh
  - ../project/agentbox/config/registered-commands.txt
  - ../project/agentbox/scripts/reconcile-commands.sh
  - ../project/agentbox/crates/colloquy/colloquy-mcp/src/server.rs
  - ../project/agentbox/crates/colloquy/colloquy-mcp/src/main.rs
  - ../project/agentbox/crates/colloquy/colloquy-core/src/cluster.rs
  - ../project/agentbox/docs/adr/ADR-2083-skills-estate-authoring-contract-and-generated-discovery.md
  - ../project/agentbox/docs/adr/ADR-2085-colloquy-knowledge-units-on-the-forum.md
  - ../project/agentbox/docs/adr/ADR-2086-confirmation-weight-follows-authorising-principals.md
  - ../project/agentbox/docs/adr/ADR-2089-skill-status-and-measured-discovery.md
  - ../project/agentbox/docs/adr/ADR-2090-skill-routing-prompt-egress.md
  - ../project/agentbox/docs/adr/ADR-2091-live-skill-router.md
  - ../project/agentbox/docs/adr/ADR-2092-govern-the-agent-and-command-registries.md
verified_commit: 1639f86ab
---

## AB-22.1 Skill discovery and JIT load at a turn

```mermaid
sequenceDiagram
    autonumber
    participant U as User turn
    participant H as Claude Code harness
    participant FS as /opt/agentbox/skills<br/>agentbox/skills/SKILL-DIRECTORY.md
    participant SK as SKILL.md<br/>agentbox/skills/tree-search-coder/SKILL.md:1
    participant REF as references/*<br/>agentbox/skills/tree-search-coder/references/algorithm.md

    Note over H,FS: INVARIANT: skills self-trigger from description frontmatter (ADR-2021) — bake root is<br/>/opt/agentbox/skills, NEVER ~/.claude/skills. 127 skills at this revision (agentbox/CLAUDE.md:63)
    Note over H,FS: INVARIANT ADR-2083 authoring contract — name equals the directory and description is<br/>at most DESC_MAX 1024 chars with what/when/when-not (agentbox/skills/lint-skills.mjs:72)<br/>Registration is by manifest: skills/registered-skills.txt:1 always-loaded for Claude Code,<br/>skills/codex-registered-skills.txt:1 for Codex — everything else is reached through the router
    H->>FS: scan description frontmatter of every SKILL.md
    alt manifest gate off — e.g. [skills.tree_search_coder] enabled = false
        Note right of H: DIVERGENCE: byte-identical-when-off (GOVERNANCE-capabilities.md Invariants) —<br/>instructional file still bakes with the skills tree, only the executable<br/>package/supervised process is omitted
        H-->>U: skill absent from routable set this turn
    else trigger phrase matches description
        H->>SK: load entry-context SKILL.md (<=250 lines, MAX_ENTRY_LINES)
        SK-->>H: name, description, triggers, manifest_gate, related_skills
        opt entry references deeper material
            H->>REF: load references/algorithm.md on demand
            REF-->>H: 7-step algorithm, manifest gate, URN schema
        end
        H-->>U: skill invoked in-turn
    end
    alt explicit /route dispatch
        U->>H: "/route <task>"
        Note right of H: see AB-22.2 for the full dispatch sequence
    end
```

## AB-22.2 Routing — the live Jev router, and the table it falls open to

```mermaid
sequenceDiagram
    autonumber
    participant U as User turn
    participant HK as skill-route hook<br/>agentbox/config/hooks/skill-route.cjs:29
    participant LIB as shared route library<br/>agentbox/config/hooks/lib/skill-route.cjs:1
    participant J as TypeSafe System One, model Jev
    participant TBL as routing-table.md<br/>agentbox/skills/skill-router/references/routing-table.md
    participant M as model

    Note over HK: registered on UserPromptSubmit ONLY when [skills.routing].router is jev<br/>and hook is true (agentbox/agentbox.toml:895-896). The gate is inlined into the<br/>command as AGENTBOX_SKILL_ROUTER=jev, so gate-off means NOT REGISTERED<br/>(byte-identical-when-off, config/hooks/skill-route.cjs:3-7)
    U->>HK: hook JSON on stdin
    HK->>LIB: config(process.env) then route(prompt, cfg)
    alt prompt shorter than min_prompt_chars 24
        LIB-->>HK: never sent, never routed (agentbox/agentbox.toml:899)
    else sent
        LIB->>J: ONE Choice question over every routable skill description, budget timeout_ms 4000 (agentbox/agentbox.toml:898)
        alt a pick comes back
            J-->>LIB: the chosen skill
            HK-->>M: additionalContext — ADVISORY ONLY, the hook recommends and never dispatches (config/hooks/skill-route.cjs:10-11)
        else error, timeout, 429 or 529, missing key, or a none pick
            J-->>LIB: nothing usable
            HK-->>M: exit 0 with NO injection — FAIL OPEN (config/hooks/skill-route.cjs:15-16)
            M->>TBL: the pre-2091 path — always-loaded descriptions self-trigger and /route reads the generated table
        end
    end
    U->>M: /route describe what you need
    M->>LIB: the slash command shares the SAME library (agentbox/agentbox.toml:886-888)
    Note over HK,J: INVARIANT ADR-2090 — the pick's probability is never a gate. Egress of the routing<br/>prompt is an accepted operator decision for skill routing ONLY<br/>(docs/adr/ADR-2090-skill-routing-prompt-egress.md:1, agentbox.toml:883-884)
    Note over LIB,TBL: DEBT — router jev is the default at this revision (agentbox.toml:895), so a turn that<br/>routes at all leaves the container. The per-project gate is explicitly deferred (agentbox.toml:884)
```

## AB-22.3 Lint gate (ADR-2021) — findings and the advisory divergence

```mermaid
sequenceDiagram
    autonumber
    participant CI as invariants.yml<br/>agentbox/.github/workflows/invariants.yml:75
    participant SH as lint-skills.sh<br/>agentbox/skills/lint-skills.sh:24
    participant MJS as main<br/>agentbox/skills/lint-skills.mjs:625
    participant TXT as checkTextPatterns<br/>agentbox/skills/lint-skills.mjs:373
    participant SKC as checkSkill<br/>agentbox/skills/lint-skills.mjs:406

    CI->>SH: bash skills/lint-skills.sh
    SH->>MJS: exec node ./lint-skills.mjs (skills/lint-skills.sh:24)
    MJS->>TXT: checkTextPatterns() over every *.md under skills/
    alt STALE — banned string, no suppress context
        TXT-->>MJS: fail STALE (BANNED regex agentbox/skills/lint-skills.mjs:59-60)
    else STALE — suppressed by DEAD/retired/legacy/lint-ok context
        TXT-->>MJS: suppressed STALE (agentbox/skills/lint-skills.mjs:61,395)
    else ABSPATH — literal ~/.claude/skills/ path, dir not in skip-list
        TXT-->>MJS: fail ABSPATH (agentbox/skills/lint-skills.mjs:394-396)
    else RETIRED-PATH — literal /workspace/ prefix, no lint-ok
        TXT-->>MJS: fail RETIRED-PATH (agentbox/skills/lint-skills.mjs:398-400)
    end
    MJS->>SKC: checkSkill(skill) for each skills/<name>/SKILL.md
    alt FRONTMATTER — missing/empty name or description, or block never opens/closes
        SKC-->>MJS: fail FRONTMATTER (agentbox/skills/lint-skills.mjs:411-422)
    else BUDGET — over MAX_ENTRY_LINES=250, no references/ or references/ has no readable file
        SKC-->>MJS: fail BUDGET (agentbox/skills/lint-skills.mjs:478-496)
    else RESOURCE — cited references|scripts|assets path does not resolve
        SKC-->>MJS: fail RESOURCE (agentbox/skills/lint-skills.mjs:501-510)
    end
    MJS->>MJS: checkEstate(entries) — REGISTERED and DIRECTORY, added with the ADR-2083 contract (agentbox/skills/lint-skills.mjs:530)
    MJS-->>CI: exit 0 clean / exit 1 with byCode summary (agentbox/skills/lint-skills.mjs:649-655)
    Note over MJS,CI: DIVERGENCE: "Skill lint is advisory — lint-skills.sh gates estate hygiene but is not a<br/>runtime capability gate — an enabled skill with clean frontmatter is trusted"<br/>(agentbox/docs/GOVERNANCE-capabilities.md:292)
```

## AB-22.4 SKILL-DIRECTORY.md maintenance vs the machine-checked count

```mermaid
sequenceDiagram
    autonumber
    participant AUTHOR as Estate audit (human/Opus swarm)
    participant DIR as SKILL-DIRECTORY.md<br/>agentbox/skills/SKILL-DIRECTORY.md:1
    participant CNT as skill-count-check.js<br/>agentbox/scripts/skill-count-check.js:115
    participant DISK as skills/*/SKILL.md<br/>agentbox/skills/

    Note over AUTHOR,DIR: SKILL-DIRECTORY.md is hand-maintained prose (categorised inventory, decision tree, MCP<br/>table) — no generator script found under agentbox/scripts for its body text
    CNT->>DISK: countSkills() — one SKILL.md per top-level skills/ dir<br/>(agentbox/scripts/skill-count-check.js:52-68)
    DISK-->>CNT: count = 127 (agentbox/scripts/skill-count-check.js output, this tree)
    CNT->>DIR: scanDoc() for "N active skills" / "N+ skills" / "for N skills" claims<br/>(agentbox/scripts/skill-count-check.js:41-45,77-106)
    alt claim matches truth count
        CNT-->>AUTHOR: ok=true (SKILL-DIRECTORY.md:3,35 both state 127 active skills — matches)
    else claim diverges from truth count
        CNT-->>AUTHOR: E-SKILL1 skill-count drift, exit 1 (agentbox/scripts/skill-count-check.js:154-161)
    end
    Note over AUTHOR,DISK: INVARIANT ADR-2056 — skill-count-check.js is the single count authority and it is<br/>wired into CI (agentbox/.github/workflows/invariants.yml:77), so a hand-written count claim<br/>in SKILL-DIRECTORY.md cannot drift silently
    Note over AUTHOR,DISK: RESOLVED — agentbox/CLAUDE.md:63 now says "The image bakes /opt/agentbox/skills (127<br/>skills)", matching skill-count-check.js and SKILL-DIRECTORY.md:3
```

## AB-22.5 harness-bridge MCP — list/inspect/validate

```mermaid
sequenceDiagram
    autonumber
    participant AG as Agent
    participant HB as harness-bridge<br/>agentbox/mcp/servers/harness-bridge.js:263
    participant LT as loadTemplates<br/>agentbox/mcp/servers/harness-bridge.js:57
    participant TD as HARNESS_TEMPLATE_DIR<br/>agentbox/mcp/servers/harness-bridge.js:25

    Note over HB: Note tools+lines — harness_list:197, harness_inspect:212, harness_validate:227,<br/>harness_audit:246 (see AB-22.6)
    AG->>HB: CallTool harness_list {maturity_filter?}
    HB->>LT: loadTemplates() — read *.json from TEMPLATE_DIR
    LT->>TD: fs.readdirSync(TEMPLATE_DIR)
    alt TEMPLATE_DIR missing
        LT-->>HB: {templates: [], warnings: ["Template directory does not exist"]}
    else templates present, each schema-validated (Ajv or manual fallback)
        LT-->>HB: {templates, warnings}
    end
    HB-->>AG: {topology, version, maturity, substrates, guide_count, sensor_count, pairing_ratio}[]<br/>(harness-bridge.js:273-288)

    AG->>HB: CallTool harness_inspect {topology}
    HB->>LT: loadTemplates()
    alt topology not found
        HB-->>AG: {error: "not_found", available: [...]} (harness-bridge.js:298-305)
    else found
        HB->>HB: computePairingAnalysis(template) (harness-bridge.js:149-179)
        HB-->>AG: full template + computed{pairing_ratio, unpaired_guides, unpaired_sensors,<br/>coverage_summary}
    end

    AG->>HB: CallTool harness_validate {topology, output_summary}
    HB->>LT: loadTemplates()
    alt topology not found
        HB-->>AG: {error: "not_found"} (harness-bridge.js:323-329)
    else found — check required_substrates and blocked_patterns per guide, plus structure.substrates
        HB-->>AG: {compliant, violations[], template_version} (harness-bridge.js:363-369)
    end
    Note over TD: RESOLVED (ADR-2057 gap 4): agentbox.toml:741-742 sets [skills.harness] enabled/template_dir =<br/>"/home/devuser/workspace/VisionFlow/docs/engineering/templates" — entrypoint-unified.sh's<br/>harness-bridge registration block now READS it (agentbox-manifest toml-string --path<br/>skills.harness.template_dir, entrypoint-unified.sh:1707-1710) and projects it into the<br/>server's HARNESS_TEMPLATE_DIR env (:1721) — an empty/absent manifest value still falls back<br/>to the server's own default (:1711), but a set value now propagates instead of being ignored
```

## AB-22.6 harness_audit — pairing ratio across all templates

```mermaid
sequenceDiagram
    autonumber
    participant AG as Agent
    participant HB as harness-bridge<br/>agentbox/mcp/servers/harness-bridge.js:372
    participant LT as loadTemplates<br/>agentbox/mcp/servers/harness-bridge.js:57
    participant PA as computePairingAnalysis<br/>agentbox/mcp/servers/harness-bridge.js:149

    AG->>HB: CallTool harness_audit {verbose?}
    HB->>LT: loadTemplates()
    loop for each template t
        HB->>PA: computePairingAnalysis(t) — pairedGuideIds/pairedSensorIds via pairings[] cross-ref
        PA-->>HB: {pairing_ratio, unpaired_guides, unpaired_sensors, coverage_summary}
        opt verbose = true
            HB->>HB: entry.unpaired_guides / entry.unpaired_sensors attached (harness-bridge.js:390-393)
        end
    end
    HB-->>AG: {audit: [{topology, guides_total, sensors_total, paired, ratio, maturity}], summary: "N<br/>templates, M% average pairing ratio"} (harness-bridge.js:399-408)
```

## AB-22.7 Colloquy knowledge-unit lifecycle — the replacement for the precedent bridge

```mermaid
stateDiagram-v2
    [*] --> Proposed
    Proposed --> Active: confirm by another authorising principal
    Proposed --> Flagged: flag
    Active --> Active: query returns it, no state change
    Active --> Flagged: flag
    Flagged --> [*]
    note right of Proposed
        propose mints the unit. colloquy-mcp/src/server.rs:43
        confirm and flag are the same attest verb with
        one boolean. colloquy-mcp/src/server.rs:166-167
    end note
    note right of Active
        INVARIANT ADR-2086 - confidence counts distinct
        AUTHORISING PRINCIPALS, never accounts, so an
        operator's fifty agents are one voice.
        colloquy-core/src/cluster.rs:49
        min_distinct_principals defaults to 3.
        colloquy-core/src/cluster.rs:66
    end note
    note right of Flagged
        DEBT - precedent-bridge.js and precedent-service.js were
        DELETED rather than migrated. The governance-precedents
        namespace was empty and nothing called the tools, so there
        was no migration to keep revertible. Commit 70d017a3b,
        ADR-2085 docs/adr/ADR-2085-colloquy-knowledge-units-on-the-forum.md:1
    end note
```

## AB-22.8 colloquy-mcp — six verbs over stdio, tier chosen by env

```mermaid
sequenceDiagram
    autonumber
    participant AG as Agent
    participant CQ as colloquy-mcp handle<br/>agentbox/crates/colloquy/colloquy-mcp/src/server.rs:141
    participant V as verbs<br/>agentbox/crates/colloquy/colloquy-mcp/src/server.rs:164
    participant ST as store, tier chosen by COLLOQUY_TIER<br/>agentbox/crates/colloquy/colloquy-mcp/src/main.rs:130
    participant EP as entrypoint registration<br/>agentbox/config/entrypoint-unified.sh:1800

    AG->>CQ: tools/list
    CQ-->>AG: query :29, propose :43, confirm :59, flag :71, reflect :83, status :107
    AG->>CQ: tools/call propose
    CQ->>V: verbs.propose(store, identity, args, now) (crates/colloquy/colloquy-mcp/src/server.rs:165)
    V->>ST: append to the tier's store
    alt COLLOQUY_TIER is not local, shared or public
        ST-->>AG: refused, the tier name is validated not defaulted (crates/colloquy/colloquy-mcp/src/main.rs:169)
    end
    AG->>CQ: tools/call confirm or flag
    CQ->>V: the SAME attest verb, the boolean is the only difference (crates/colloquy/colloquy-mcp/src/server.rs:166-167)
    Note over EP: the server is registered only when [skills.colloquy].enabled is true<br/>(agentbox/agentbox.toml:906) AND the container has both a member pubkey and an<br/>authorising principal — otherwise the boot SKIPS it and says why<br/>(config/entrypoint-unified.sh:1802-1806)
    Note over EP: INVARIANT — the recorded command is compared against the canonical baked binary and<br/>rewritten when they differ, so a stale /nix/store path self-heals on rebuild<br/>(config/entrypoint-unified.sh:1813-1815). The previous grep-once guard could not<br/>correct itself. see AB-09.5
    Note over ST,AG: tier shared is the default on purpose — agents in one container are one<br/>operator's, so a private tier-1 store would lose every learning at session end<br/>(agentbox/agentbox.toml:908-913)
```

## AB-22.9 continual-harness — evidence-anchored signed refine

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator/agent
    participant CLI as continual-harness.cjs<br/>agentbox/mcp/servers/continual-harness.cjs:55
    participant CH as createHarness().refine<br/>agentbox/mcp/servers/lib/continual-harness.js:153
    participant IB as immutableBase guard<br/>agentbox/mcp/servers/lib/continual-harness.js:138
    participant GIT as harnessDir git repo<br/>agentbox/mcp/servers/lib/continual-harness.js:118
    participant SIGN as defaultSign<br/>agentbox/mcp/servers/lib/continual-harness.js:87

    OP->>CLI: continual-harness refine <layer> <key> --value V --evidence E [--reason R]
    CLI->>CH: h.refine({layer, key, value, evidence, reason, actor})
    alt layer not in supplemental-prompt|memory|skill-spec|subagent-spec
        CH-->>CLI: throw "layer must be one of: ..." (continual-harness.js:154)
    else key fails [a-z0-9-]{1,128} slug
        CH-->>CLI: throw "key must match [a-z0-9-]" (continual-harness.js:155)
    else evidence missing or blank — INVARIANT: no anchor, no refine
        CH-->>CLI: throw "evidence is required — a refine must cite the transcript span / commit / test<br/>that justifies it" (continual-harness.js:157-159)
    else evidence present
        CH->>IB: targetPath(layer, key) resolves under harnessDir
        alt resolved path escapes harnessDir
            IB-->>CH: throw "refine target escapes the harness dir" (continual-harness.js:141)
        else resolved path IS one of the immutable-base files (CLAUDE.md tiers)
            IB-->>CH: throw "refuse to write an immutable base file" (continual-harness.js:144) — TWO-LAYER<br/>hard invariant, never a convention
        else path is inside the mutable layer
            CH->>GIT: write <layer>/<key>.md with layer/key/operator/actor/evidence/updated header<br/>(continual-harness.js:164-178)
            GIT-->>CH: git add, then diff --cached --name-only
            alt no staged diff — identical content
                CH-->>CLI: {changed: false, commit: HEAD, signature: null, signed: false}<br/>(continual-harness.js:184)
            else content changed
                CH->>SIGN: sign(canonical, {key, evidence}) — Schnorr via management-api Nostr key<br/>(REFINE_KIND=30841)
                alt signer/key material unavailable (dev shell) — fail-open, not signature-invalid
                    SIGN-->>CH: null
                    CH->>GIT: commit with Refine-Signature: deferred, Refine-Pubkey: (deferred)<br/>(continual-harness.js:199-200)
                    CH-->>CLI: {changed: true, signature: "deferred", signed: false}
                else signer present
                    SIGN-->>CH: {sig, pubkey}
                    CH->>GIT: commit with Refine-Evidence/Refine-Operator/Refine-Signature/Refine-Pubkey trailers
                    CH-->>CLI: {changed: true, commit, signature: sig, signed: true}
                end
            end
        end
    end
    CLI-->>OP: JSON result to stdout
    Note over CH,GIT: INVARIANT: immutable base = CLAUDE.md tiers (~/.claude/CLAUDE.md, ~/workspace/CLAUDE.md,<br/>vault CLAUDE.md via VAULT_ROOT — ADR-2028) — a refine resolving onto any of them is<br/>REJECTED, enforced as a path guard not a convention (continual-harness.js:50-65,143-144)
```

## AB-22.10 Skills-side MCP projection boundary — skills/mcp.json to workspace .mcp.json

```mermaid
flowchart TB
    REG["skills/mcp.json<br/>agentbox/skills/mcp.json:3<br/>CANONICAL BOOT-PROJECTION SOURCE"]
    MIRROR["mcp/mcp.json<br/>agentbox/mcp/mcp.json:3<br/>INFRASTRUCTURE REGISTRY — consumed by agentbox.toml, flake.nix, mcp-ws-relay.js"]
    BAKE["baked at /opt/agentbox/skills/mcp.json<br/>entrypoint _MCP_REGISTRY"]
    PROJ["project-mcp-servers.mjs<br/>agentbox/scripts/project-mcp-servers.mjs:1"]
    GATE["gateOpen(x-agentbox-gate)<br/>agentbox/scripts/project-mcp-servers.mjs:234"]
    REQ["requiresMet(x-agentbox-requires)<br/>agentbox/scripts/project-mcp-servers.mjs:248"]
    LEDGER["ownership ledger<br/>agentbox/scripts/project-mcp-servers.mjs:268"]
    WS[".mcp.json workspace target<br/>entrypoint-unified.sh:1031 WORKSPACE/.mcp.json"]
    BESPOKE["bespoke entries hand-written earlier in entrypoint<br/>harness-bridge entrypoint-unified.sh:1873, colloquy :1813,<br/>claude-flow and browser-gpu — precedent-bridge is GONE, see AB-22.7"]

    REG -- "mirror shared server wiring (skills/mcp.json:3 comment)" --> MIRROR
    REG -- "nix bake" --> BAKE
    BAKE -- "MCP_REGISTRY env" --> PROJ
    PROJ --> GATE
    PROJ --> REQ
    GATE -- "x-agentbox-managed-by == projector only" --> PROJ
    REQ -- "bin on PATH / file exists / envset non-empty" --> PROJ
    PROJ --> LEDGER
    LEDGER -- "D1: owned name whose definition vanished is removed + recorded" --> WS
    BESPOKE -- "written first, entrypoint-unified.sh:2252 runs the projector AFTER" --> WS
    PROJ -- "reconcile: upsert projector-managed servers, remove closed-gate ones" --> WS
```

## AB-22.11 Vault path authority for skill-side corpus readers (ADR-2028)

```mermaid
sequenceDiagram
    autonumber
    participant EP as entrypoint _ab_vault_resolve<br/>agentbox/config/entrypoint-unified.sh:81
    participant TOML as agentbox.toml [vault]<br/>agentbox/agentbox.toml:706
    participant ENV as exported VAULT_ROOT/PAGES/FORMAT/TUI<br/>agentbox/config/entrypoint-unified.sh:124
    participant SK as vault-writing skills<br/>podcast-knowledge-ingest, web-summary note-link mode
    participant VF as vault-frontmatter.js<br/>agentbox/mcp/servers/lib/vault-frontmatter.js:242
    participant CI as check-no-logseq-paths.sh<br/>agentbox/scripts/ci/check-no-logseq-paths.sh:18

    EP->>TOML: _ab_toml_val vault root (entrypoint-unified.sh:86)
    alt no [vault] in agentbox.toml — VAULT_ROOT empty
        EP->>EP: unset VAULT_ROOT/PAGES/FORMAT/TUI/WORKING_ROOT/WORKING_PAGES/TRANSCRIPTS<br/>(entrypoint-unified.sh:88)
        EP-->>ENV: echo "[vault] disabled — no [vault] in agentbox.toml" (entrypoint-unified.sh:91)
        alt AGENTBOX_VAULT_LEGACY_PATHS=1 opt-in
            EP-->>ENV: RETAIN deprecated ONTOLOGY_PAGES_DIR with a warning (entrypoint-unified.sh:96-98)
        else no opt-in
            EP-->>ENV: WARNING then clear ONTOLOGY_PAGES_DIR="" (entrypoint-unified.sh:103-107)
        end
        SK->>SK: consumer sees no VAULT_PAGES and no retained override — disables itself, one clear line<br/>(ADR-2028 D3, fail-loud)
    else [vault].root present
        EP->>EP: VAULT_PAGES = root/pages, VAULT_FORMAT default obsidian, VAULT_TUI default none<br/>(entrypoint-unified.sh:111-115)
        EP->>ENV: export VAULT_ROOT VAULT_PAGES VAULT_FORMAT VAULT_TUI ... (entrypoint-unified.sh:124)
        opt explicit ONTOLOGY_PAGES_DIR differs from VAULT_PAGES
            EP-->>ENV: note override honoured for legacy consumers, VAULT_PAGES remains the authority<br/>(entrypoint-unified.sh:130-131)
        end
        ENV-->>SK: every supervised program, tmux window, MCP server inherits<br/>VAULT_ROOT/VAULT_PAGES/VAULT_FORMAT
        SK->>VF: ensureFrontmatter(text, extraProps, opts) before writing a page<br/>(vault-frontmatter.js:242)
        VF-->>SK: V2 YAML frontmatter — public as real boolean, wikilinks quoted, legacy key:: value block<br/>converted
    end
    CI->>CI: grep -rn "workspace/logseq" outside docs/archive and docs/adr<br/>(check-no-logseq-paths.sh:18-24)
    alt hard-coded corpus literal found
        CI-->>EP: FAIL — exit 1, points at ADR-2028 remediation (check-no-logseq-paths.sh:29-40)
    else clean
        CI-->>EP: PASS (check-no-logseq-paths.sh:43)
    end
```

## AB-22.12 tree-search-coder — execution-gated branching with enforced spend cap

```mermaid
sequenceDiagram
    autonumber
    participant U as User/coordinator (explicit invocation only, never auto-routed)
    participant TSC as tree-search-coder SKILL<br/>agentbox/skills/tree-search-coder/SKILL.md:80
    participant CAP as tree-search-cap CLI<br/>agentbox/services/agentbox-ops/src/bin/tree-search-cap.rs
    participant LED as cost_cap ledger (flock)<br/>agentbox/services/agentbox-ops/src/cost_cap/mod.rs:361
    participant SC as sparc:coder
    participant KS as KernelSession (code-interpreter MCP)

    U->>TSC: /tree-search-coder <task> (manifest_gate [skills.tree_search_coder] enabled=true)
    loop candidate k = 1..max_candidates (default 5, agentbox.toml:644)
        TSC->>CAP: tree-search-cap reserve --run RUN_ID --estimate 0.13 (algorithm.md:60-63)
        CAP->>LED: reserve_at(run_id, estimate, now) under exclusive flock (cost_cap/mod.rs:361-366)
        break exit 3 REFUSED — spend_cap_exceeded, candidate_limit_exceeded, branch_timeout, or capability_disabled
            LED-->>CAP: refuse reservation
            CAP-->>TSC: exit 3 (algorithm.md:72-75)
            TSC-->>U: return best candidate found so far, annotated halted true reason spend_cap<br/>(algorithm.md:21-24)
        end
        CAP-->>TSC: exit 0 granted — reservation id, candidate_index, remaining_usd
        TSC->>SC: sparc:coder with varied temperature/framing (algorithm.md:6-8)
        SC-->>TSC: candidate program
        TSC->>KS: kernel.reset (fresh session per branch, algorithm.md:9-11)
        TSC->>KS: kernel.exec candidate's assertions/tests (algorithm.md:12-14)
        KS-->>TSC: ExecutionTrace — assertion-pass count
        alt branch succeeds within per_branch_timeout_s
            TSC->>CAP: tree-search-cap settle --run RUN_ID --reservation res-id --actual 0.11 (algorithm.md:65)
        else branch fails or is cancelled — hold must still be released
            TSC->>CAP: tree-search-cap settle --run RUN_ID --reservation res-id --actual 0.00 --failed<br/>(algorithm.md:67)
        end
        CAP->>LED: settle_at(...) — release hold, charge actual (cost_cap/mod.rs:434-444)
    end
    TSC->>TSC: score by assertion-pass count, select highest, tie-break shortest code<br/>(algorithm.md:16-18)
    TSC-->>U: chosen candidate + audit trajectory JSONL for ExpeL distillation (algorithm.md:27-28)
    Note over TSC,LED: INVARIANT (ADR-2020): no default-unlimited mode — an absent spend_cap_usd falls back to<br/>the documented 0.50 USD default, never to infinity (algorithm.md:24,<br/>agentbox.toml:644-646)
    Note over TSC,U: INVARIANT: never auto-routed — SKILL-DIRECTORY.md and skill-router's routing table<br/>exclude tree-search-coder from automatic dispatch (algorithm.md manifest gate section)
```

## AB-22.13 [skills.*] manifest gate table (ADR-2020, catalogue parity closed)

```mermaid
flowchart TB
    TOML["agentbox.toml [skills.*] blocks<br/>agentbox/agentbox.toml:589"]
    MANI["system-manifest.js catalogue<br/>agentbox/management-api/lib/system-manifest.js:41"]

    subgraph rebuild["apply_class rebuild — Nix package set plus supervisor block, image rebuild required"]
        CI2["code_interpreter agentbox.toml:589<br/>system-manifest.js:143"]
        CODEACT["codeact agentbox.toml:605"]
        ACI["aci_shell agentbox.toml:729<br/>system-manifest.js:256"]
        TSCB["tree_search_coder agentbox.toml:771<br/>system-manifest.js:259"]
        RES["research.web_researcher agentbox.toml:582<br/>system-manifest.js:146"]
    end
    subgraph boot["apply_class boot — manifest re-read at container boot, no rebuild"]
        RVB["ruvnet_brain agentbox.toml:737<br/>system-manifest.js:210"]
        ONT["ontology agentbox.toml:780<br/>system-manifest.js:227"]
        HARN["harness agentbox.toml:873<br/>system-manifest.js:234"]
        CQG["colloquy agentbox.toml:906<br/>system-manifest.js:237"]
        RTR["routing.router — off_values table<br/>agentbox.toml:895, system-manifest.js:224"]
    end
    subgraph vault["[vault] — sibling top-level section, not [skills.*]"]
        VLT["vault agentbox.toml:838 — ADR-2028 path authority"]
    end

    TOML --> MANI
    MANI --> rebuild
    MANI --> boot
    TOML --> vault

    HARN -->|"gate read at boot, config/entrypoint-unified.sh:1862, THEN the file must exist"| REG2["harness-bridge registered in .mcp.json only when [skills.harness].enabled is true"]
    CQG -->|"gate read at boot, config/entrypoint-unified.sh:1761, plus a member and an authorising principal"| REG3["colloquy registered in .mcp.json — it REPLACED the precedent bridge, see AB-22.7"]
    RTR -->|"router table de-registers the UserPromptSubmit hook entirely"| REG4["byte-identical-when-off, see AB-22.2"]

    TSCB -->|"enabled false refuses every tree-search-cap reservation"| CAPGATE["cost_cap ledger — manifest gate check, algorithm.md Enforced cost cap"]

    MANI --> RESOLVED["RESOLVED ADR-2057 gap 1/2 — harness and the retired precedent gate were the two<br/>that had no catalogue entry. harness now carries one (system-manifest.js:234) and<br/>colloquy took the second slot (system-manifest.js:237), so the ADR-039 honesty rule<br/>holds for every [skills.*] gate this diagram names"]
    RESOLVED --> INV["INVARIANT ADR-039 — the apply class follows WHERE the gate is consumed:<br/>boot for a gate the entrypoint reads, rebuild for one that decides baked text<br/>(management-api/lib/system-manifest.js:229-232)"]
```

## AB-22.14 Generated discovery — the table nobody hand-edits (ADR-2083)

```mermaid
flowchart TB
    FM["every skills/<name>/SKILL.md frontmatter description"] --> GEN["gen-routing-table.mjs<br/>agentbox/skills/gen-routing-table.mjs:20-21"]
    MAP["section-map.json — the router's single source of sections<br/>agentbox/skills/skill-router/references/section-map.json:1"] --> GEN
    GEN --> OUT["skill-router/references/routing-table.md<br/>marked Generated artefact, do not hand-edit<br/>agentbox/skills/gen-routing-table.mjs:72-74"]
    GEN --> MISS["a skill directory absent from the map is an ERROR,<br/>not a silent omission (agentbox/skills/gen-routing-table.mjs:61)"]
    CI["invariants.yml routing-table freshness<br/>agentbox/.github/workflows/invariants.yml:79"] --> GEN
    DIRC["lint DIRECTORY check — the inventory MCP column is DERIVED from<br/>frontmatter, so it is gated rather than trusted<br/>agentbox/skills/lint-skills.mjs:569-575"] --> OUT
    REGC["lint REGISTERED check — a name in either manifest with no<br/>SKILL.md, or pointing at a deprecated redirect stub, fails<br/>agentbox/skills/lint-skills.mjs:537-541"] --> OUT
    OUT --> AG["ADR-2092 gives agents and slash-commands the same manifest shape —<br/>agents/registered-agents.txt:1 projected by scripts/reconcile-agents.sh:1,<br/>config/registered-commands.txt:1 pruned by scripts/reconcile-commands.sh:1"]
    OUT --> RM["INVARIANT ADR-2089 — a whole skill may be REMOVED on measurement plus an<br/>operator decision; four tooling-free thinking lenses went that way<br/>agentbox/CLAUDE.md:63, docs/adr/ADR-2089-skill-status-and-measured-discovery.md:1"]
```

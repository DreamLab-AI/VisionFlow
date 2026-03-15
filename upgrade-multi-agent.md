upgrade multi-agent

Below is a full Product Requirements Document for upgrading `./multi-agent-docker` using `./turbo-flow` as the reference/migration source, with the expectation that `./turbo-flow` will be deleted after the upgrade. The available context confirms the current multi-agent workstation structure, key files, and build path, including the unified build flow and service orchestration files [1].

# PRD: Upgrade `./multi-agent-docker` Using `./turbo-flow` as Temporary Migration Source

## 1. Document Summary

### Title
Upgrade and Modernization of `./multi-agent-docker` via Temporary `./turbo-flow` Reference

### Status
Proposed

### Author
AI-generated implementation PRD

### Primary Goal
Use `./turbo-flow` as a temporary reference implementation and source of patterns, configs, scripts, and architecture decisions to upgrade `./multi-agent-docker`, then remove `./turbo-flow` entirely once parity or improvement is achieved.

### Confirmed Existing Context
The current `multi-agent-docker` setup includes:
- `docker-compose.unified.yml` as the agentic workstation compose entrypoint [1]
- `Dockerfile.unified` as the primary container build file [1]
- `unified-config/supervisord.unified.conf` for 9+ managed services [1]
- `unified-config/supervisord.simple.conf` for minimal services [1]
- `unified-config/entrypoint-unified.sh` for container initialization [1]
- `unified-config/tmux-autostart.sh` for an 8-window workspace [1]
- `.env` and `.env.example` for configuration and API keys [1]
- Build command `./multi-agent-docker/build-unified.sh` [1]

## 2. Background

`./multi-agent-docker` appears to already be a structured multi-service workstation environment. The upgrade objective is not to preserve `./turbo-flow` as a runtime dependency, but to mine it for:
- better configuration conventions
- improved startup orchestration
- agent workflow improvements
- service definitions
- shell ergonomics
- observability and health checks
- security hardening
- developer experience improvements

After migration, `./multi-agent-docker` should be self-sufficient and `./turbo-flow` should be deletable with no regressions.

## 3. Problem Statement

The current workstation is functional, but likely needs modernization and consolidation. Typical issues this kind of upgrade should solve:

- duplicated or drifted config between projects
- unclear startup ordering across many supervised services
- environment variable inconsistency
- missing health checks and failure visibility
- weak separation between build-time and run-time configuration
- tmux workspace setup that may be rigid or non-role-aware
- insufficient onboarding and reproducibility
- inability to safely remove old reference project without losing valuable behavior

## 4. Product Vision

Create a cleaner, more modular, more observable, and easier-to-maintain `./multi-agent-docker` that:
- absorbs the best ideas from `./turbo-flow`
- standardizes service lifecycle management
- improves local and CI build reliability
- supports future extension without copying another repo again
- becomes the canonical workstation implementation

## 5. Scope

## In Scope

- Audit `./turbo-flow` and map useful components to `./multi-agent-docker`
- Refactor `./multi-agent-docker` configs and scripts
- Upgrade Docker, supervisor, entrypoint, env templating, and tmux startup
- Add validation, health checks, documentation, and migration guardrails
- Create a one-time migration framework or checklist
- Ensure `./multi-agent-docker/build-unified.sh` remains the canonical build path [1]

## Out of Scope

- Preserving `./turbo-flow` long term
- Building a generalized package manager for arbitrary workstation repos
- Rewriting the platform away from Docker unless explicitly desired
- Replacing supervisor/tmux unless the migration reveals strong justification

## 6. Assumptions

Because the provided context is limited, the following are informed assumptions rather than confirmed facts:
- `./turbo-flow` contains overlapping workstation or agent orchestration logic
- both directories likely share shell, Docker, or agent patterns
- the target runtime should stay centered on `multi-agent-docker/docker-compose.unified.yml` [1]
- service orchestration should continue using supervisor configs unless a migration decision explicitly changes that [1]

## 7. Users / Stakeholders

### Primary Users
- developers running the multi-agent workstation locally
- maintainers of `multi-agent-docker`
- platform engineers integrating local and CI workflows

### Secondary Users
- new contributors onboarding the workstation
- operators debugging service startup failures
- future maintainers who need a self-contained canonical implementation

## 8. Success Criteria

A successful upgrade means:

1. `./multi-agent-docker` contains all desired functionality previously only found in `./turbo-flow`
2. `./turbo-flow` can be deleted with no required runtime references
3. `./multi-agent-docker/build-unified.sh` still builds the workstation successfully [1]
4. Startup through `docker-compose.unified.yml` remains reliable [1]
5. All required services are documented and health-checked
6. `.env.example` fully represents required configuration surface [1]
7. Tmux and entrypoint logic are deterministic and configurable [1]
8. Supervisor-managed services are clearly grouped, restart-safe, and loggable [1]

## 9. Key Upgrade Principles

### 9.1 Canonicalize, Don’t Copy Blindly
Do not wholesale copy `./turbo-flow`. Instead:
- inventory features
- classify them as adopt/adapt/reject
- port only what strengthens `./multi-agent-docker`

### 9.2 Delete-by-Design
Every imported pattern from `./turbo-flow` must land in:
- a documented file
- an owned module
- a tested config
So the source directory can be removed immediately afterward.

### 9.3 Runtime Simplicity
Keep the runtime centered around the existing compose + Dockerfile + supervisor + entrypoint stack unless there is a compelling reason to alter it [1].

### 9.4 Explicit Configuration
Everything configurable should live in:
- `.env.example`
- compose environment sections
- well-documented shell variables
- optional role-based preset files

## 10. Functional Requirements

## FR1. Turbo-Flow Audit and Mapping
Create a migration inventory document that compares:
- scripts
- Dockerfiles
- compose files
- env files
- startup scripts
- agent definitions
- shell aliases
- tmux layout logic
- supervisor or service management config
- model/provider integrations
- testing scripts
- health-check logic

Output:
- `multi-agent-docker/docs/turbo-flow-migration-map.md`

Each item in `./turbo-flow` should be labeled:
- Adopt
- Adapt
- Ignore
- Deprecated
- Unknown / review needed

## FR2. Docker Build Modernization
Upgrade `multi-agent-docker/Dockerfile.unified` [1] to incorporate useful improvements found in `./turbo-flow`, potentially including:
- clearer layer ordering
- dependency caching
- package grouping
- ARG vs ENV discipline
- non-root user setup
- smaller final image
- consistent shell defaults
- deterministic installs
- better cache bust strategy
- explicit version pins for critical tools

Acceptance:
- build remains runnable via `./multi-agent-docker/build-unified.sh` [1]
- no hard dependency on `./turbo-flow`
- image builds reproducibly from `multi-agent-docker` alone

## FR3. Compose Standardization
Refactor `multi-agent-docker/docker-compose.unified.yml` [1] to:
- standardize service naming
- improve volume mounts
- centralize env file usage
- add health checks where possible
- clarify ports and network topology
- support profiles if useful
- separate optional vs core services

Acceptance:
- default launch path remains simple
- optional complexity is hidden behind profiles or env toggles
- all mounted paths and env assumptions are documented

## FR4. Supervisor Service Rationalization
The current setup includes a supervisor config managing 9+ services [1]. Upgrade it to:
- group services logically
- define startup priorities
- use consistent restart policy
- emit logs to predictable locations
- support debug and minimal modes
- extract repeated config patterns if feasible

Potential outputs:
- `supervisord.unified.conf` remains primary [1]
- `supervisord.simple.conf` remains minimal baseline [1]
- optional split configs under `unified-config/supervisord.d/`

Acceptance:
- services start in deterministic order
- a failed optional service should not always block the whole workspace
- logs and restart loops are easier to diagnose

## FR5. Entrypoint Hardening
Upgrade `unified-config/entrypoint-unified.sh` [1] to:
- validate required env vars
- create required directories
- initialize permissions
- detect first-run vs subsequent-run behavior
- render templates if used
- emit startup summary
- support safe debug mode
- fail fast on critical misconfiguration
- remain idempotent

Acceptance:
- rerunning startup should not corrupt config
- missing required env vars produce clear errors
- startup behavior is traceable and documented

## FR6. Tmux Workspace Upgrade
Upgrade `unified-config/tmux-autostart.sh` [1] to provide:
- role-based layouts
- optional window presets
- service status pane or dashboard
- shell/session naming consistency
- ability to disable autostart
- support for task-focused layouts such as coding, monitoring, agents, logs

Accepted example modes:
- `TMUX_LAYOUT=default`
- `TMUX_LAYOUT=debug`
- `TMUX_LAYOUT=minimal`
- `TMUX_AUTOSTART=false`

Acceptance:
- layout script is modular and readable
- users can override without editing core script
- autostart remains helpful but not intrusive

## FR7. Environment Model Cleanup
Standardize `.env` and `.env.example` [1] with:
- categorized variables
- required/optional distinctions
- sane defaults where safe
- comments explaining purpose
- validation rules
- deprecation notes for renamed vars migrated from `./turbo-flow`

Suggested sections:
- Core runtime
- User / UID / GID
- API providers
- Agent config
- Logging / observability
- Tmux behavior
- Feature flags
- Debug options

Acceptance:
- a new developer can bootstrap with `.env.example`
- legacy names from `./turbo-flow` are mapped or rejected explicitly
- no secret values are hardcoded

## FR8. Migration Compatibility Layer
For one release cycle, optionally support deprecated `./turbo-flow` env names or script assumptions through a compatibility layer:
- env alias mapping
- deprecation warnings
- shim scripts where necessary

Examples:
- if `TURBO_*` vars exist, map them to `MULTI_AGENT_*` or equivalent
- if old path assumptions exist, log migration warnings

Acceptance:
- compatibility is temporary and documented
- all warnings point users to the updated names
- compatibility layer can be removed within a defined milestone

## FR9. Documentation Upgrade
Add or update:
- `multi-agent-docker/README.md`
- `docs/architecture.md`
- `docs/services.md`
- `docs/configuration.md`
- `docs/migration-from-turbo-flow.md`
- `docs/troubleshooting.md`

Must include:
- build instructions using `./multi-agent-docker/build-unified.sh` [1]
- service model
- env setup
- startup/attach commands
- tmux behavior
- common failures
- deletion plan for `./turbo-flow`

## FR10. Validation and Test Coverage
Add lightweight validation for:
- shell scripts
- compose syntax
- supervisor config sanity
- env completeness
- critical file existence
- basic container startup smoke test

Potential additions:
- `scripts/validate.sh`
- `scripts/smoke-test.sh`
- CI workflow for build + startup validation

Acceptance:
- maintainers can validate before deleting `./turbo-flow`
- failures are actionable and early

## 11. Non-Functional Requirements

## NFR1. Maintainability
Configs should be modular, named clearly, and documented.

## NFR2. Observability
Logs for service startup and failure should be easy to find.

## NFR3. Reproducibility
Builds should be deterministic enough for team use.

## NFR4. Security
- avoid overbroad permissions
- clearly separate secrets from defaults
- favor non-root runtime where possible
- minimize unnecessary packages and open ports

## NFR5. Deletability of Turbo-Flow
No final code path in `multi-agent-docker` may require:
- importing files from `./turbo-flow`
- bind mounting `./turbo-flow`
- shelling into `./turbo-flow` scripts
- doc references that imply it must remain

## 12. Proposed Architecture

## 12.1 Target Structure
Suggested upgraded structure:

```text
multi-agent-docker/
  build-unified.sh
  docker-compose.unified.yml
  Dockerfile.unified
  .env
  .env.example
  README.md
  docs/
    architecture.md
    services.md
    configuration.md
    troubleshooting.md
    migration-from-turbo-flow.md
    turbo-flow-migration-map.md
  scripts/
    validate.sh
    smoke-test.sh
    migrate-env.sh
    doctor.sh
  unified-config/
    entrypoint-unified.sh
    supervisord.unified.conf
    supervisord.simple.conf
    supervisord.d/
      core.conf
      optional.conf
      debug.conf
    tmux-autostart.sh
    tmux/
      layouts/
        default.sh
        debug.sh
        minimal.sh
      lib.sh
    env/
      required.env.schema
      optional.env.schema
```

## 12.2 Logical Layers
1. Build layer
   - Dockerfile
   - package/tool installation
2. Runtime orchestration layer
   - compose
   - entrypoint
   - supervisor
3. Workspace layer
   - tmux
   - shell UX
4. Configuration layer
   - `.env`
   - schema/validation
5. Validation/documentation layer
   - scripts
   - docs

## 13. Migration Strategy from `./turbo-flow`

## Phase 1: Discovery
Perform a structured diff between `./turbo-flow` and `./multi-agent-docker`:

Checklist:
- Compare Dockerfiles
- Compare compose files
- Compare env names
- Compare startup scripts
- Compare service inventory
- Compare shell setup
- Compare tmux layouts
- Compare logging strategy
- Compare health checks
- Compare helper scripts
- Compare docs

Deliverable:
- migration spreadsheet or markdown map

## Phase 2: Classify
For each `./turbo-flow` asset:
- Is it superior?
- Is it still needed?
- Is it compatible?
- Is it secure?
- Is it generic enough?
- Can it be folded into existing files cleanly?

## Phase 3: Port
Port in small slices:
1. env model
2. Dockerfile improvements
3. compose improvements
4. entrypoint hardening
5. supervisor reorganizing
6. tmux UX
7. docs and validation

## Phase 4: Compatibility
Add temporary aliases/shims for anything likely to break current users.

## Phase 5: Verification
Build and run using only `multi-agent-docker` paths:
- build via `./multi-agent-docker/build-unified.sh` [1]
- start via compose [1]
- verify all expected services
- verify tmux/autostart
- verify docs from fresh clone workflow

## Phase 6: Turbo-Flow Deletion
Delete `./turbo-flow` only after:
- no direct references remain
- shims are internalized
- docs point only to `multi-agent-docker`
- smoke test passes

## 14. Detailed Implementation Plan

## Workstream A: Repository Audit
### Tasks
- create a file inventory of `./turbo-flow`
- annotate ownership and purpose
- identify duplicate functionality already in `multi-agent-docker`

### Outputs
- migration map
- risk register
- decision log

## Workstream B: Build System Refactor
### Tasks
- compare package/tooling installation patterns
- merge any superior shell tooling, caches, package mirrors, user setup
- standardize build arguments
- add labels/metadata if useful

### Risks
- image bloat
- package incompatibilities
- hidden runtime deps copied from turbo-flow

## Workstream C: Runtime Config Upgrade
### Tasks
- normalize compose service conventions
- add healthchecks
- ensure volumes/ports are explicit
- define minimal and full launch modes

### Risks
- startup ordering issues
- host path assumptions
- environment drift

## Workstream D: Entrypoint and Supervisor
### Tasks
- harden entrypoint
- split supervisor configs if too large
- define core vs optional services
- improve stderr/stdout handling

### Risks
- service restart storms
- race conditions on startup
- permissions problems

## Workstream E: UX / Tmux Upgrade
### Tasks
- modularize tmux script
- add layout presets
- add attach/help messaging
- support disabling automatic session startup

### Risks
- brittle pane commands
- poor noninteractive behavior
- interference with debugging

## Workstream F: Docs, Validation, and Deletion Readiness
### Tasks
- write migration docs
- add validation scripts
- add smoke test
- create deletion checklist for `./turbo-flow`

### Risks
- undocumented assumptions
- deletion uncovers hidden dependency

## 15. Acceptance Criteria

The project is complete when all of the following are true:

### Build
- `./multi-agent-docker/build-unified.sh` successfully builds from a clean environment [1]

### Runtime
- `docker-compose.unified.yml` launches the workstation without any dependency on `./turbo-flow` [1]

### Config
- `.env.example` is sufficient to configure a fresh setup [1]

### Service Management
- supervisor configs are reliable, documented, and easier to debug [1]

### Startup
- `entrypoint-unified.sh` is idempotent and validates configuration [1]

### Workspace UX
- `tmux-autostart.sh` supports configurable layouts and can be disabled [1]

### Migration
- all valuable functionality from `./turbo-flow` is either migrated, intentionally rejected, or documented as obsolete

### Deletion
- removing `./turbo-flow` does not break build, startup, docs, or workflows

## 16. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hidden dependency on `./turbo-flow` | High | grep all references before deletion; add smoke test |
| Conflicting env names | Medium | env alias map + deprecation warnings |
| Service startup regression | High | add ordered supervisor priorities and health checks |
| Tmux script becomes too complex | Medium | split layouts into separate files |
| Docker image growth | Medium | track image size before/after |
| Poor onboarding after migration | High | update README and config docs first |
| Build cache instability | Medium | pin key versions and simplify layers |
| Over-copying from `./turbo-flow` | Medium | require adopt/adapt/reject decision per asset |

## 17. Metrics

Track:
- clean build success rate
- time to first successful startup
- number of required manual setup steps
- number of undocumented env vars
- number of direct references to `./turbo-flow`
- number of startup failures due to env/config validation
- image size delta
- number of services with health checks

## 18. Open Questions

These are not answerable from the provided context alone:
1. What exactly exists in `./turbo-flow`?
2. Which services are the 9+ supervisor-managed processes? [1]
3. Is the system intended for local-only use, CI use, or both?
4. Are there GPU, browser, or IDE-specific dependencies?
5. Are there user-specific workflow requirements for the 8-window tmux workspace? [1]
6. Is backward compatibility required for existing env names?
7. Are there secrets/provider integrations beyond standard API keys in `.env`? [1]

## 19. Recommended Milestones

## Milestone 1: Audit and Design
- complete migration map
- produce architecture decisions
- identify no-go items from `./turbo-flow`

## Milestone 2: Build and Config Foundation
- Dockerfile refactor
- env cleanup
- entrypoint validation
- docs skeleton

## Milestone 3: Runtime and Service Reliability
- compose cleanup
- supervisor improvements
- health checks
- smoke tests

## Milestone 4: UX and Compatibility
- tmux layouts
- deprecation warnings
- migration helper scripts

## Milestone 5: Deletion Readiness
- remove all references to `./turbo-flow`
- run validation suite
- delete `./turbo-flow`

## 20. Deletion Checklist for `./turbo-flow`

Before deleting `./turbo-flow`, verify:

- no import/path references in scripts
- no compose mounts from `./turbo-flow`
- no Docker COPY commands from `./turbo-flow`
- no env file sourcing from `./turbo-flow`
- no docs instruct users to inspect `./turbo-flow`
- all desired configs copied/adapted into `multi-agent-docker`
- validation scripts pass
- smoke test passes
- startup and tmux behavior verified manually

## 21. Suggested Immediate Next Steps

1. Create `docs/turbo-flow-migration-map.md`
2. Inventory all files in `./turbo-flow`
3. Add a “no external turbo-flow dependency” rule to the upgrade
4. Refactor `.env.example` first
5. Harden `entrypoint-unified.sh` next [1]
6. Split or rationalize supervisor config after that [1]
7. Then improve tmux layouts and docs [1]
8. Add validation and smoke tests
9. Delete `./turbo-flow` only after a clean build/startup cycle using `multi-agent-docker` alone [1]

If you want, I can next turn this into one of these forms:
1. an engineering task breakdown with epics/stories/subtasks
2. a file-by-file implementation plan for `multi-agent-docker`
3. a concrete migration checklist assuming a typical `turbo-flow` layout
4. a draft `README` and `migration-from-turbo-flow.md` for the repo
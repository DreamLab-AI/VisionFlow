---
id: OM-12
title: Shared filesystem checks and consumer features
area: ontomark
governing:
  - ../ontomark/docs/adr/ADR-OM-011-shared-filesystem-layout-checks.md
  - ../project/docs/VAULT-corpus-format.md
adrs: [ADR-OM-008, ADR-OM-011, ADR-2168]
sources:
  - ../ontomark/crates/ontomark-compile/src/profile/layout.rs
  - ../ontomark/crates/ontomark-compile/src/profile.rs
  - ../ontomark/crates/ontomark-compile/src/input/vocabulary.rs
  - ../ontomark/crates/ontomark-compile/src/input/page.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/migrate/plan.rs
  - ../ontomark/crates/ontomark-compile/src/migrate/apply.rs
  - ../ontomark/crates/ontomark-compile/src/check/bodies.rs
  - ../ontomark/crates/ontomark-compile/src/check/repair.rs
  - ../ontomark/crates/ontomark-cli/src/main.rs
  - ../ontomark/crates/ontomark-cli/src/migrate_cmd.rs
  - ../project/src/services/corpus_source/local.rs
verified_commit: {ontomark: 6159bb6089753105ef839fd4aac46c1598a64bd2, visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

Ontomark owns the path checks; VisionClaw consumes `ontomark-compile` through default crate features. The [section assessment and regression evidence](../../estate-review/2026-11-01-ontomark-layout-and-shared-checks.md) distinguish configuration rejection, publication policy and runtime activation. The consumer source pin precedes the dependency-only update from compiler 0.6.2 to 0.6.3; the final lock and release receipt are recorded in that review.

## For the business

A misplaced or linked source should produce an error before publication or partial ingestion. These checks make that feedback consistent across the compiler and its consumer. They do not establish publication privacy, replace producer parity checks or prove a deployed service is running the new code.

## OM-12.1 One owner resolves roots before assigning roles

```mermaid
flowchart TD
    REPO["Canonical repository must be a directory<br/>layout.rs:140-151"] --> DECL["Relative non-empty declarations<br/>no absolute or parent components<br/>layout.rs:91-101"]
    DECL --> RES["Resolve existing ancestors<br/>retain missing optional leaves<br/>layout.rs:54-89"]
    RES --> IN["Resolved root stays inside repository<br/>layout.rs:111-117"]
    IN --> DIS["Reject equal or nested roots<br/>layout.rs:153-164"]
    DIS --> ROLE["Exact resolved root selects the role<br/>layout.rs:199-213"]
    ROLE -.-> LIMIT["Point-in-time configuration check<br/>concurrent replacement remains outside the guarantee<br/>layout.rs:123-124"]
```

**What it shows.** A single flat `.` is valid. Adding a nested working role, an alias or an escaping root fails. An absent optional leaf is checked through its existing ancestors.

**Why it is this way.** Suffix inference confused `nested/drafts` with `drafts`. The shared resolver compares complete paths and preserves the declared ordering.

## OM-12.2 Discovery, validation and build agree on the layout

```mermaid
flowchart TD
    START["Canonicalise discovery start<br/>vocabulary.rs:507-510"] --> FIND["Walk ancestors for vocabulary<br/>vocabulary.rs:512-518"]
    FIND --> CLI["CLI validates profile layout<br/>main.rs:659"]
    CLI --> MIG["Selected migration role must exist in layout<br/>migrate_cmd.rs:142-145"]
    BUILD["Build validates profile layout<br/>mod.rs:645"] --> EX["Built-in exclusions plus profile additions<br/>profile.rs:530-537"]
    EX --> LOAD["Load under exact resolved role<br/>mod.rs:655-667"]
    CLI --> LOAD
```

**What it shows.** Relative and implicit discovery reach the same ancestor vocabulary. Build role assignment uses the same layout contract, and profile furniture exclusions are additive.

**Why it is this way.** A validator's refusal is useful only if the publication path applies the same role rules. Arbitrary required-key enforcement and custom public-gate policy remain separate open findings.

## OM-12.3 VisionClaw consumes checks at each local-source boundary

```mermaid
flowchart TD
    LIST["Default feature checks all roots before listing<br/>local.rs:230-232"] --> SHARED["Call owner resolve_roots<br/>local.rs:120-123"]
    SHARED --> PREFIX["Resolve requested prefix within configured roots<br/>local.rs:94-108"]
    PREFIX --> WALK["Skip child symlinks, including cycles<br/>local.rs:173-182"]
    WALK --> MISSING["Missing optional source can still be skipped<br/>local.rs:235-245"]
    FETCH["Fetch canonicalises the page path<br/>local.rs:262-267"] --> BOUND["Require membership in a configured root<br/>local.rs:268-278"]
    BOUND --> READ["Read the resolved path<br/>local.rs:280-283"]
```

**What it shows.** The server's default `shared-corpus-layout` feature uses the registry compiler without its default reasoner feature. The vault CLI separately enables `shared-layout`. The actual local module and real vault binary have regression tests; the full server has a compile check.

**Why it is this way.** The owner maintains path semantics once. Feature-off builds retain legacy behaviour for compatibility comparison, including weaker link handling. This is not a server deployment receipt.

## OM-12.4 Journal walks and migration writes also check their roots

```mermaid
flowchart TD
    JOURNAL["Reject linked journal root before separate walk<br/>page.rs:524-535"] --> PLAN["Planner applies the same journal-root guard<br/>plan.rs:341-349"]
    REPAIR["Body and fence repair reject linked scope roots<br/>bodies.rs:791; repair.rs:387"]
    APPLY["Resolve migration log directory and children<br/>apply.rs:145-146"] --> TARGET["Check planned target containment<br/>apply.rs:182"]
    ROLLBACK["Canonical journal must belong to repository<br/>apply.rs:298-305"] --> RESTORE["Check target before restoring bytes<br/>apply.rs:347"]
    APPLY --> SHARED["Reuse owner resolver for paths and log children<br/>apply.rs:80-90"]
    TARGET --> SHARED
    RESTORE --> SHARED
```

**What it shows.** Separate walk roots need explicit guards: child-link traversal settings did not protect a linked `journals/`. A matching content hash likewise did not prevent a parent directory from being redirected outside the repository between planning and application.

**Why it is this way.** The follow-up checks stable path redirection before reading or writing. It does not claim race-safe filesystem capabilities. Repair deliberately refuses linked scope roots even where a configured read-role root could resolve to a permitted in-repository target.

---
title: Capability instructions and enforced limits
status: source-and-isolated-probe-verified
date: 2026-09-04
type: explanation
---

# Capability instructions and enforced limits

Skills describe how an agent should use tools. The manifest, image builder, registration projector and runtime executor each control a different part of availability. A skill's instruction to stop at a budget is not equivalent to a limiter that rejects the next paid operation. The closeout roadmap must identify which boundary enforces each promise.

[Source and isolated lint receipts](evidence/skill-lint-probes.json) establish this review's scope. No tree-search skill was invoked, no agent delegated work, and no provider or rebuild ran.

## Tree search is an orchestration instruction

The tree-search skill explicitly states that it carries no code of its own. It describes candidate generation, clean kernels, assertion scoring and a spend-cap halt. Its algorithm says to stop remaining branches if the cap is exceeded mid-search. That is an instruction to the orchestrating agent; it does not establish pre-authorisation, reservation or cancellation at the billing boundary.

The inspected management-api, services, manifest implementation and flake paths show configuration and an ENABLE_TREE_SEARCH_CODER projection, but did not establish a runtime consumer enforcing the named spend/candidate/timeout fields. This is a bounded negative source finding, not proof that no external orchestrator could enforce them. Nor was a cost overrun reproduced. Historical research lift figures in the skill are not evidence of this estate's performance and are not re-verified here.

For a hard limit, require the actual executor to reserve budget before work, account for concurrent and in-flight calls, apply candidate/time ceilings and report actual versus estimated cost. Explicit-only routing must be tested at invocation entry points. Separate these requirements from the useful instructional policy already present.

## Disabled has several meanings

The flake copies the entire skills input into the image, while capability-specific package and supervisor gates control other artefacts. A disabled optional capability can therefore still have instructional files in the image. The inspected code does not justify a universal zero-footprint or byte-identical image claim from a single false flag. Build closure, process absence, registration absence and execution denial need separate acceptance evidence.

The [configuration review](configuration-projection.md) explains why gate declarations and projected registrations are not loaded-state receipts. Disabling a feature also needs a policy for old sessions and previously projected entries. No Nix closure comparison or live activation test ran here.

## Lint checks conventions, not semantic validity

The current skill lint checks text patterns, counts top-level SKILL.md lines and accepts any existing references directory for a long entry file. It checks the first line for an opening delimiter and searches the whole file for name/description lines; it does not parse a frontmatter block.

The [actual-script fixtures](evidence/skill-lint-probes.py) show:

| Fixture | Result |
|---|---|
| 304-line entry without references directory | Fails monolith rule |
| Same 304-line entry with empty references directory | Passes |
| Empty frontmatter with name/description in the body | Passes |

The workflow invokes this lint, so it has value as a CI convention check. Its success does not prove bounded trigger context, valid frontmatter, referenced-file completeness or runtime sandboxing. A pre-rebuild policy also needs evidence that every supported build entry point invokes the gate; the presence of a CI step alone cannot establish that.

## Closeout requirements

CP-01/07/08 require typed frontmatter validation, actual entry-context limits, reference existence checks, and explicit treatment of suppressions and nested skills. Test positive and negative fixtures and bind the gate result to the baked skills revision. Runtime capability admission and spending require separate executable receipts.

ADR-2020 becomes partial for the broad off-state and hard-spend guarantees. ADR-2021 becomes partial for frontmatter and bounded-context guarantees while preserving the useful lint conventions. Maintainers must define the intended contract before extending the tests; merely creating an empty references directory cannot satisfy the original context-budget purpose.

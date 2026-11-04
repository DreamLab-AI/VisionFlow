---
id: ADR-2023
title: Adapt three ideas from NVIDIA auto-ontology — field-level certification, question coverage grading and per-question benchmark directions — and upgrade the corpus header to carry them
date: 2026-11-04
decision_status: accepted
implementation_status: partial
activation_status: staged
supersedes: []
superseded_by: []
depends_on: [ADR-2013, ADR-2014, ADR-2022]
verified_commit: add1aa703
verified_paths: [ontomark:crates/ontomark-ingest/src/certification.rs, ontomark:crates/ontomark-ingest/src/coverage.rs, ontomark:crates/ontomark-ingest/src/accept.rs, ontomark:crates/ontomark-cli/src/ground_cmd.rs, ontomark:crates/ontomark-compile/src/build/okf.rs, ontomark:crates/ontomark-compile/src/build/page_api.rs, ontomark:docs/adr/ADR-OM-012-certification-grounding-and-context-v2.md, visiongraph:ontology/vocabulary.yaml, visiongraph:vault.toml, visiongraph:.github/workflows/publish.yml, visiongraph:docs/adr/ADR-VG-009-header-v2-keyed-certification-and-generated-context.md, visionclaw:crates/visionclaw-domain/src/vault/mod.rs, visionclaw:crates/visionclaw-ontology/src/services/jsonld_validator/frame.rs, loom:crates/loom-bench/src/lib.rs, loom:crates/loom-bench/src/main.rs, agentbox:skills/ontology-augment/SKILL.md]
owner: jjohare
review_trigger: Activation — an ontomark release carrying ADR-OM-012 is pinned by the visionGraph publisher, narrativegoldmine.com serves identical bytes at /context/v2.jsonld and /ns/v2.jsonld, and the agentbox image ships an ontomark with `ground`. Reopen certification when the first `verified` stamp with `keys` lands, or when a hand edit after certification is found. Reopen the deferred items (relational-schema Source, Ossie interchange) when a real database or a semantic-model consumer appears.
repo: visionflow
domain: BASELINE-visionflow.md
evidence_class: source+test
acceptance_boundary: Established by crate tests in ontomark, VisionClaw and loom-bench, and by validate/build runs over the 9,609-page visionGraph corpus with the pinned ontomark v0.7.3, VisionClaw `vault` and the ADR-OM-012 branch. Not established — no ontomark release has been cut, nothing has been published, the agentbox image has not been rebuilt, and no Loom benchmark has been re-run to produce a direction report.
---

# ADR-2023 — Adapt three ideas from NVIDIA auto-ontology, and upgrade the corpus header to carry them

## Context

[NVIDIA auto-ontology](https://github.com/NVIDIA/auto-ontology) (Apache-2.0,
published 2026-11) compiles relational database metadata into business terms,
reviewed SQL attributes and relationships. People certify the result in a UI, and
an agent answers questions from it over MCP. Its domain is text-to-SQL, not a
reasoned OWL corpus, and its extraction is less strict than `ontomark-ingest`:

- auto-ontology writes LLM drafts straight to its store.
- ontomark asks the model for names (never identifiers), requires a verbatim
  evidence quote for every value, records each refusal with its reason, ties every
  patch to the exact version of the note, and lands external sources as proposals
  (ADR-OM-003, ADR-OM-010).

The comparison found three ideas worth adapting:

| auto-ontology | Where | Gap on our side |
|---|---|---|
| Certification per field (`name_certified`, …) kept across recompiles; derived `certified / partial / pending` | `dal/terms.py:157-232, 370` | Review is page-level only (`verified` stamps); ingest cannot tell a reviewed key from an unreviewed one |
| `check_answerable`: covered ÷ named entities, with the uncovered ones listed | `retrieval/entity_coverage/agents/coverage.py` | Ingest grounds names against the vocabulary, but nothing grades a *question* |
| The eval report lists the questions an ontology fixed and the ones it broke | auto-ontology-eval | `loom-bench` reports only the mean paired difference and its CI |

Deferred: a relational-schema `Source`, until there is a real database to ingest,
and Apache Ossie interchange, until a consumer asks for it.

Declined:
- the Python, Postgres and pgvector stack;
- unauthenticated ingestion endpoints and identity forwarded in a header;
- uploading tables to a hosted predictor (KumoRFM).

The operator asked for the corpus header and JSON-LD context to be upgraded to carry
this, with every parser updated. That work surfaced two existing defects:

- The publisher served a stale, hand-written `ns/v2.jsonld` over the generated
  one.
- VisionClaw's governed write-back deleted every frontmatter value its typed
  fields could not hold: `verified`, `sources` items and `generated.process`.

## Decision

1. **A human `verified` stamp may name the keys it certifies, and ingest never
   adds a value to a certified key** (ontomark ADR-OM-012, ontomark `ef48510`).
   - An entry without `keys` certifies the whole page.
   - Only `human:` actors certify. Ingest and the compiler are pinned to the same
     rule by a shared test table.
   - The page API reports `certification: {state, keys}`.
   - Certification is written one stamp at a time by a human, through
     `ontomark edit`. There is no bulk certification.
2. **Agents grade a question's coverage before answering.**
   - `ontomark ground <NAME>...` sorts each name into grounded, unclear or
     uncovered, and gives `coverage` (`null` when no names were given) (ontomark
     `17edddb`).
   - The `ontology-augment` skill tells agents to cite grounded names, ask about
     unclear ones and say so for uncovered ones (agentbox `ce899e126`).
3. **`loom-bench score` lists per-question directions.** Beside the
   preregistered mean and CI, each question is listed as improved, regressed or
   unchanged (loom `741c5df`).
   - It compares arm 3 (scaffold injected) with arm 4 (the model drives `/mcp`).
     Both use the ontology, so this is not a with/without-ontology comparison.
   - The preregistered metric is continuous recall with no threshold, so
     direction is the strict sign of the difference. It is a secondary
     diagnostic.
4. **Corpus header v2** (visionGraph ADR-VG-009, `add1aa703`).
   - Vocabulary v2 adds `verified[].keys`.
   - The canonical context is `context/v2.jsonld`, generated by ontomark and
     closed over the page API, and is also served at `ns/v2.jsonld`.
     `context/v1.jsonld` keeps its exact bytes.
   - The hand-written override is deleted.
   - The page retrofit set is **empty**: no header change both earns its place
     and leaves `ontology.ttl` byte-identical, and certification cannot be
     applied by a script.
5. **VisionClaw's frontmatter writer is lossless** (VisionClaw `d74b25143`).
   - `PageMeta` keeps the parsed mapping and writes back what its typed fields
     cannot hold.
   - All 9,609 corpus pages pass a parse → render round-trip with no change.
   - VisionClaw's JSON-LD validator accepts `context/v2`.
   - The other writers have not been tested against `verified[].keys`. The
     census found that they round-trip the frontmatter mapping (ontomark
     `edit`/`apply`/`create`, `vault-core`) or edit lines (`fmlib.py`,
     `amend.py`). For ontomark `edit`, a test proves that `keys` survives.

## Consequences

- **No triples change, measured.**
  - v0.7.3 builds of visionGraph `main` (vocabulary v1) and of the v2 branch
    differ only in `vocabulary_version` and the content digest, in the
    generation markers and `okf/index.md`.
  - The ADR-OM-012 build differs from v0.7.3 only in the page API's
    `certification` field and the context files.
  - VisionClaw `vault build` and `vault certify` produce byte-identical
    derivation certificates.

  So VisionFlow ADR-2022's corpus history records nothing from this change.
- **Every page API document gains one field.** ontomark's legacy-parity golden
  was re-blessed for this in a targeted way. A refusal script checked that
  nothing else changed.
- **Certification's day-one effect: one page.** *Fungible Token* has a
  whole-page human stamp, so ingest stops proposing changes to it and its page
  document reports `certified`. No page has `keys` yet, and the effect grows as
  reviews land.
- **Cost of `ground`:** about 3.7 s cold over visionGraph per call.
- **Activation steps** (operator decisions, all outward-facing):
  1. Cut an ontomark release.
  2. Bump the publisher pin (tag and digest), add `context/v2.jsonld` to its
     required-artefact gate and to `check-urls.sh`, and point the Quartz footer
     at v2.
  3. Rebuild the agentbox image.
  4. Re-capture the `corpus-visiongraph` golden at the release.
- **Follow-ups recorded, not done:**
  - VisionClaw's legacy `vault build` still has its own context generator (v1
    only). It feeds the Solid pod, not narrativegoldmine.com, so no URL serves two
    documents. Retire it with the producer migration (ADR-OM-005).
  - Nothing compares `vocabulary.yaml` `version` with `vault.toml`.
  - *Presentation Concluding Synthesis Slide* has an unquoted-wikilink alias that
    YAML reads as a nested list. It needs a reviewed edit.

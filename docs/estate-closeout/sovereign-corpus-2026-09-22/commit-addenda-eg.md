# Prepared commit-message addenda (NOT committed)

## VisionClaw (/home/devuser/workspace/project)

```
corpus(writers): decision and class pages are OKF frontmatter, not json-ld fences

PRD-sovereign-corpus Q4/Q5/Q16. draft_decision_page and draft_class_page emit
typed Obsidian Properties only: type/resource/status/generated, relations as
wikilink lists under keys ontology/vocabulary.yaml declares. Both pages pass
`vault validate --vault knowledge` with 0 errors; the rationale and the draft
definition go to body prose, which is where the vocabulary's own migration rule
C puts a value with no declared home.

PageMeta grows the OKF fields and extra_lists, so a relation survives
render->parse as a list. ontology_mutation_service appended to a comma-joined
scalar, collapsing two edges into one malformed target; it now appends to the
list.

Writers repointed off the retired mainKnowledgeGraph/ layout onto
KNOWLEDGE_PAGES_DIR (knowledge/pages), the prefix DECISIONS_DIR already used.

The assert-graph rebuild filtered on owl_class_iri.is_some(), which shipped any
node with an ontology-shaped IRI to Whelk — a public working/ Episode among
them. It now filters on a declaration: knowledge/ AND type in
{Class, Property, Individual}. The KG graph ingest still reads both base paths.

Tests: a public working Episode is a graph node and never an ontology class; no
frontmatter value is ever an environment value; every key the writers emit is
declared in the live vocabulary.yaml.

Co-Authored-By: jjohare <github@thedreamlab.uk>
```

## agentbox (/home/devuser/workspace/project/agentbox)

```
skills(podcast-ingest): ledger pages are typed Episodes in working/podcast-evidence

PRD Q16. SKILL.md, both references, podcasts.yaml, crontab and run-promote.sh
now describe the subdirectory layout, frontmatter-only metadata and `**key:**`
prose; the documented json-ld template fence and every `key::` example are gone.
Ledger Episodes are documented as `public: false` always, with publication an
explicit human act (owner, 2026-09-22 14:00). podcast-bulk-ingest's review-dir
example wrote the rejected `public:: false` Logseq form; corrected. Records the
podcast-promote --ledger-dir follow-up.

Co-Authored-By: jjohare <github@thedreamlab.uk>
```

## visionGraph (/home/devuser/workspace/visionGraph)

```
transcripts: write evidence ledgers to working/pages/podcast-evidence

Typed Episode, `public: false` always (owner, 2026-09-22 14:00): Q16 publishes
`public: true` from EITHER vault, so this writer must never emit it. Publishing
an Episode is a human act, and a re-ledgered page is reset to false because new
machine claims have landed since the human read it. Every field a frontmatter
key, no `key::` line and no json-ld fence. render_page refuses any frontmatter
carrying an unexpanded environment reference (Invariant 11). 24 tests, incl. a
vault-validate contract test against the live vocabulary and manifest.

Co-Authored-By: jjohare <github@thedreamlab.uk>
```

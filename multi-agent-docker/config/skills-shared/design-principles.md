# Skills Design Principles

These principles govern the design, structure, and maintenance of all skills.
They are non-negotiable for new skills and should be incrementally applied to
existing skills during migration.

---

## 1. Single Responsibility

Each skill does one thing well.

- A skill addresses one tool, one workflow, or one bounded domain.
- If a skill description requires "and" to connect two unrelated capabilities,
  split it into two skills.
- Cross-cutting concerns (e.g. "use browser to test a deployment") belong in
  composition, not in a single monolithic skill.

**Test**: Can you describe the skill in one sentence without the word "and"
connecting unrelated clauses? If not, split it.

---

## 2. Progressive Disclosure

Keep the entrypoint lean. Put depth elsewhere.

- `SKILL.md` is the entrypoint. It must be scannable in under 2 minutes.
- Detailed tutorials, deep-dive explanations, and reference tables go in
  `docs/` subdirectory files.
- `SKILL.md` links to `docs/` files rather than embedding their content.
- A user who reads only `SKILL.md` should be able to complete the Quick Start
  successfully.

**Test**: Is `SKILL.md` under 50 KB? Can a new user reach a working result from
Quick Start alone? If not, trim and delegate.

---

## 3. No Duplicate Content

Say it once, reference it everywhere.

- If two skills share setup instructions, extract the shared content into a
  common skill or a `shared/` document and reference it.
- Never copy-paste sections between skills. Use cross-references:
  `See [Browser Prerequisites](../browser/SKILL.md#prerequisites)`.
- The `find-duplicates.py` script enforces this: pairs exceeding 30% text
  similarity are flagged for consolidation.

**Test**: Run `find-duplicates.py`. Zero pairs above 30% similarity.

---

## 4. Explicit Boundaries

Every skill states when to use it AND when not to use it.

- **When to Use**: Concrete scenarios where this skill is the correct choice.
  Not vague ("when you need automation") but specific ("when testing a web
  application's login flow across multiple browsers").
- **When Not to Use**: Scenarios where this skill is the wrong choice, with a
  pointer to the correct alternative. "For API-only testing without a browser,
  use the `playwright` skill with headless mode or plain `curl`."

**Test**: Does the skill have both "When to Use" and "When Not to Use" sections?
Do they contain specific, actionable guidance?

---

## 5. Executable and Tested

Every script, command, and example must work.

- Code blocks in Quick Start must produce the documented output when executed
  in the documented environment.
- Scripts in `scripts/` must be executable (`chmod +x`) and include a shebang
  line.
- Scripts must handle missing dependencies gracefully with clear error messages,
  not stack traces.
- Every skill should be validatable by the lint and validation scripts without
  errors.

**Test**: Follow the Quick Start from a clean environment. Does it work? Run
`lint-skills.py` on the skill. Zero errors.

---

## 6. Metadata-Driven Discovery

Frontmatter is the primary index for skill discovery.

- The `name` and `description` fields in frontmatter must be sufficient for
  a search tool to determine relevance without reading the full document.
- Tags provide secondary categorisation. Use specific tags (`web-scraping`)
  not generic ones (`tool`).
- `depends_on` enables dependency resolution. If your skill cannot function
  without another skill, declare the dependency.

**Test**: Search for the skill using only its `description` text. Does the
description alone convey what it does and when to reach for it?

---

## 7. Graceful Degradation

Skills must handle missing prerequisites without crashing.

- The Prerequisites section must list every external dependency.
- Quick Start should include a check: "Verify `tool --version` returns X.Y+."
- When a dependency is missing, the skill's instructions should lead to a clear
  error message and a fix, not silent failure or cryptic stack traces.

**Test**: Remove one prerequisite and follow the instructions. Is the failure
clear? Does the Troubleshooting section cover it?

---

## Summary Checklist

Before submitting a new or revised skill, verify:

- [ ] Single responsibility: one thing, described in one sentence
- [ ] Progressive disclosure: SKILL.md under 50 KB, details in docs/
- [ ] No duplicates: similarity < 30% with all other skills
- [ ] Explicit boundaries: "When to Use" and "When Not to Use" present
- [ ] Executable: Quick Start works, scripts are tested
- [ ] Metadata complete: frontmatter has `name` and `description` with what+when
- [ ] Graceful degradation: prerequisites listed, failures handled

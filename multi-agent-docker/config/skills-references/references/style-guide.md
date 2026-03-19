# Skills Style Guide

This document defines the writing and structural standards for all skills in the
`~/.claude/skills/` directory. Every skill must conform to these rules before
merging.

---

## Naming Conventions

### Directory Names

- **Format**: `lowercase-kebab-case`
- **Length**: 3-64 characters
- **Characters**: `a-z`, `0-9`, `-` only. No underscores, no uppercase.
- **Examples**: `browser`, `rust-development`, `github-code-review`
- **Anti-examples**: `Browser`, `rust_development`, `GitHub-Code-Review`

### File Names

| File | Required | Purpose |
|------|----------|---------|
| `SKILL.md` | Yes | Primary skill document (the entrypoint) |
| `docs/*.md` | No | Extended documentation, tutorials, deep-dives |
| `scripts/*` | No | Executable helper scripts |
| `examples/*` | No | Worked examples and sample configs |
| `tests/*` | No | Validation tests for the skill |

All markdown files use lowercase-kebab-case: `setup-guide.md`, not
`SetupGuide.md`.

---

## SKILL.md Required Sections (in order)

Every `SKILL.md` must contain these sections in this exact order:

1. **YAML frontmatter** -- See `frontmatter-schema.md`
2. **Title** -- `# <Skill Name>` (H1, exactly one)
3. **Summary** -- One to three sentences immediately below the title. States
   what the skill does and when to invoke it.
4. **Prerequisites** -- Tools, services, or configuration required before use.
5. **Quick Start** -- The simplest working invocation (under 10 lines).
6. **Usage** -- Detailed commands, flags, parameters. Use tables for reference.
7. **When to Use** -- Bullet list of concrete scenarios.
8. **When Not to Use** -- Bullet list of scenarios where this skill is the wrong
   choice, with pointers to the correct alternative.
9. **Troubleshooting** -- Common failure modes and their fixes.

Optional sections (placed after Troubleshooting):

- **Configuration** -- Environment variables, config files
- **Advanced Usage** -- Complex workflows, composition with other skills
- **References** -- Links to upstream documentation

---

## Writing Style

### Tone and Voice

- **Procedural**: Write instructions as steps. "Run `cargo build`." not "You
  can try running cargo build."
- **Trigger-based**: State the condition, then the action. "When the build fails
  with error E0308, check type annotations."
- **Direct**: No hedging. "This script validates frontmatter." not "This script
  aims to help validate frontmatter."

### Language

- **UK English throughout**:
  - `colour` not `color` (in prose; code identifiers follow their language)
  - `optimise` not `optimize`
  - `behaviour` not `behavior`
  - `organisation` not `organization`
  - `licence` (noun) / `license` (verb)
  - `catalogue` not `catalog`
  - `analyse` not `analyze`
  - `centre` not `center`
  - `defence` not `defense`
  - `programme` (noun, non-computing) / `program` (computing)
- **Exception**: Code identifiers, CLI flags, and API fields retain their
  original spelling regardless of origin (e.g. `--color` flag stays as-is).

### Prohibited Patterns

| Pattern | Reason |
|---------|--------|
| Marketing language ("revolutionary", "powerful", "seamless") | Not verifiable |
| Emojis in headings or body text | Noise; reduces scannability |
| `TODO`, `FIXME`, `HACK` | Incomplete work must not ship |
| "In a real implementation..." | All content must be real |
| Vague claims ("improves performance") | Must be specific: "reduces latency by ~40%" |
| Nested blockquotes (> > ) | Hard to parse |
| HTML tags in markdown | Use standard markdown only |

### Formatting

- Line length: 80 characters soft limit for prose. Code blocks exempt.
- One blank line between sections.
- Code blocks must specify the language: ````bash`, ````python`, ````yaml`.
- Tables must have a header row and alignment.
- Lists use `-` (not `*` or `+`).
- Indent nested lists with 2 spaces.

---

## Testability

Every claim in a skill must be testable:

- "Supports Python 3.10+" -- Testable: run `python3 --version`.
- "Fast" -- Not testable. Replace with a measurable statement.
- "Run `npm test` to validate" -- Testable if the command exists and exits 0.

Every Quick Start section must produce observable output when followed exactly.

---

## File Size Limits

| File | Maximum |
|------|---------|
| `SKILL.md` | 50 KB |
| Any single markdown file in `docs/` | 100 KB |
| Total skill directory | 1 MB |

If a `SKILL.md` approaches 50 KB, split detailed content into `docs/`
subdirectory files and reference them from the main document.

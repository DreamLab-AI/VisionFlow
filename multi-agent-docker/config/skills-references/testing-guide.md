# Skills Testing Guide

This document describes how to validate that a skill conforms to the standards
defined in the style guide and frontmatter schema.

---

## Validation Layers

Validation is structured in four layers, from fastest to most thorough:

### Layer 1: Frontmatter Check

**Script**: `scripts/validate-frontmatter.py`
**Runtime**: < 1 second for all skills

Checks:
- YAML frontmatter block exists (delimited by `---`)
- YAML parses without errors
- Required fields present: `name`, `description`
- `name` is 64 characters or fewer
- `description` is 1024 characters or fewer
- `description` contains both a capability statement and a conditional clause
- `version` matches semver pattern if present
- `status` is one of `active`, `deprecated`, `experimental` if present
- `upstream` is a valid URL if present
- `tags` contains 10 or fewer lowercase kebab-case strings if present
- `depends_on` entries are valid skill directory names if present
- No disallowed fields (`triggers`, `tools`, `capabilities`, `author`,
  `category`, `requires`)

### Layer 2: Section Completeness

**Script**: `scripts/lint-skills.py`
**Runtime**: < 5 seconds for all skills

Checks:
- `SKILL.md` exists in the skill directory
- Exactly one H1 heading
- Required sections present (as H2 headings):
  - Prerequisites
  - Quick Start
  - Troubleshooting
- Recommended sections present (warning, not error):
  - Usage
  - When to Use
  - When Not to Use
- No `TODO`, `FIXME`, `HACK`, or `XXX` markers
- File size under 50 KB
- UK English spot-checks (flags US spellings in prose)
- Code blocks specify a language

### Layer 3: Link Validation

**Script**: `scripts/check-links.py`
**Runtime**: < 10 seconds for all skills (filesystem only, no HTTP)

Checks:
- Internal markdown links (`[text](path)`) resolve to existing files
- Cross-skill references (`depends_on`, `supersedes`) point to existing skill
  directories
- Anchor links (`#section-name`) match actual heading anchors in the target file
- Image references point to existing files
- No broken relative paths

### Layer 4: Duplicate Detection

**Script**: `scripts/find-duplicates.py`
**Runtime**: < 30 seconds for all skills

Checks:
- Pairwise Jaccard similarity on 3-gram shingles of text content
- Flags pairs exceeding 30% similarity
- Groups overlapping skills into clusters
- Reports which skill should be the canonical one (by age or completeness)

---

## Running Validation

### Full Suite

```bash
cd /home/devuser/workspace/skills-revamp

# Run all checks against the skills directory
python3 scripts/validate-frontmatter.py ~/.claude/skills/
python3 scripts/lint-skills.py ~/.claude/skills/
python3 scripts/check-links.py ~/.claude/skills/
python3 scripts/find-duplicates.py ~/.claude/skills/
```

### Single Skill

```bash
# Validate one skill
python3 scripts/validate-frontmatter.py ~/.claude/skills/browser/
python3 scripts/lint-skills.py ~/.claude/skills/browser/
python3 scripts/check-links.py ~/.claude/skills/browser/
```

### CI Integration

All scripts exit with code 0 on success, non-zero on failure. They produce:
- **stdout**: Human-readable summary
- **reports/**: JSON files for machine consumption

JSON output files are written to the `reports/` directory relative to the
scripts' working directory. Set `--reports-dir` to override.

---

## Interpreting Results

### Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| `error` | Blocks merge. Must fix. | Fix before committing. |
| `warning` | Does not block. Should fix. | Fix in follow-up or justify in PR. |
| `info` | Informational. | No action required. |

### Common Failures and Fixes

| Failure | Fix |
|---------|-----|
| Missing `description` in frontmatter | Add a description that states what + when |
| Description lacks "when" clause | Append a sentence: "Use when [condition]." |
| US English detected ("optimize") | Replace with UK English ("optimise") |
| Missing Prerequisites section | Add `## Prerequisites` with tool requirements |
| TODO marker found | Complete the TODO or remove the section |
| Link to non-existent file | Fix the path or remove the link |
| Similarity > 30% with another skill | Consolidate or deduplicate content |

---

## Adding New Validation Rules

To add a check:

1. Identify which layer it belongs to (frontmatter, structure, links, content).
2. Add the check to the appropriate script.
3. Assign a severity level (`error`, `warning`, `info`).
4. Add a test case to `tests/` that exercises the new rule.
5. Document the rule in this file.

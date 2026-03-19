# Frontmatter Schema

Every `SKILL.md` must begin with a YAML frontmatter block delimited by `---`.
This document defines the canonical schema.

---

## Schema Definition

```yaml
---
name: "string"           # Required. Max 64 characters.
description: "string"    # Required. Max 1024 characters. Must state WHAT + WHEN.
version: "string"        # Optional. Semver format: "1.0.0", "0.2.1".
status: "string"         # Optional. One of: active, deprecated, experimental.
upstream: "string"       # Optional. URL to source repository if skill was imported.
tags: [list]             # Optional. List of lowercase strings. Max 10 tags.
depends_on: [list]       # Optional. List of other skill directory names.
supersedes: [list]       # Optional. List of deprecated skill names this replaces.
---
```

---

## Field Details

### `name` (required)

- **Type**: String
- **Max length**: 64 characters
- **Format**: Human-readable display name. May contain spaces, hyphens, and
  mixed case.
- **Examples**: `"Browser Automation"`, `"Rust Development"`,
  `"GitHub Code Review"`
- **Validation**: Must not be empty. Must not exceed 64 characters.

### `description` (required)

- **Type**: String
- **Max length**: 1024 characters
- **Content requirements**: Must include both:
  1. **What** the skill does (the capability)
  2. **When** to use it (the trigger condition)
- **Good example**: `"Automates web browser interactions using accessibility
  snapshots. Use when you need to navigate, scrape, or test web pages
  programmatically."`
- **Bad example**: `"Browser automation skill"` (missing "when")
- **Bad example**: `"Use this when you need to browse the web"` (missing "what")
- **Validation**: Must not be empty. Must not exceed 1024 characters. Lint
  scripts flag descriptions that lack both a capability verb and a conditional
  clause.

### `version` (optional)

- **Type**: String
- **Format**: Semantic versioning -- `MAJOR.MINOR.PATCH`
- **Examples**: `"1.0.0"`, `"0.3.2"`, `"2.1.0"`
- **Validation**: Must match pattern `^\d+\.\d+\.\d+$` if present.

### `status` (optional)

- **Type**: String, enumerated
- **Allowed values**:
  - `active` -- Fully supported and maintained (default assumption if absent)
  - `deprecated` -- No longer recommended; see `supersedes` for replacement
  - `experimental` -- Under development; interface may change
- **Validation**: Must be one of the three allowed values if present.

### `upstream` (optional)

- **Type**: String (URL)
- **Purpose**: Points to the original source repository when a skill was
  imported or forked from an external project.
- **Examples**: `"https://github.com/anthropics/claude-code"`
- **Validation**: Must be a valid URL (starts with `http://` or `https://`) if
  present.

### `tags` (optional)

- **Type**: List of strings
- **Max items**: 10
- **Format**: Each tag must be lowercase, alphanumeric with hyphens.
  No spaces, no uppercase.
- **Examples**: `[browser, automation, web-scraping, testing]`
- **Validation**: Each tag must match `^[a-z0-9-]+$`. Maximum 10 tags.

### `depends_on` (optional)

- **Type**: List of strings
- **Format**: Each entry must be a valid skill directory name (lowercase
  kebab-case).
- **Purpose**: Declares runtime dependencies on other skills.
- **Examples**: `[browser, github-workflow-automation]`
- **Validation**: Each entry must match `^[a-z0-9-]+$`.

### `supersedes` (optional)

- **Type**: List of strings
- **Format**: Each entry must be a skill directory name.
- **Purpose**: Lists deprecated skills that this skill replaces.
- **Examples**: `[old-browser-skill, legacy-web-automation]`
- **Validation**: Each entry must match `^[a-z0-9-]+$`.

---

## Disallowed Fields

The following fields are **not part of the canonical schema** and must be
removed during migration:

| Field | Reason | Migration |
|-------|--------|-----------|
| `triggers` | Replaced by description "when" clause | Move trigger words into description |
| `tools` | Implementation detail, not metadata | Move to Prerequisites or Usage |
| `capabilities` | Redundant with description and body | Move to SKILL.md body |
| `author` | Not tracked at skill level | Remove |
| `category` | Use `tags` instead | Convert to first tag |
| `requires` | Use `depends_on` with skill names | Convert to `depends_on` |

---

## Examples

### Minimal Valid Frontmatter

```yaml
---
name: "Browser Automation"
description: "Automates web browser interactions using accessibility snapshots. Use when navigating, scraping, or testing web pages programmatically."
---
```

### Complete Frontmatter

```yaml
---
name: "GitHub Code Review"
description: "Coordinates multi-agent code review for GitHub pull requests. Use when reviewing PRs that need security, performance, and style analysis."
version: "1.0.0"
status: "active"
upstream: "https://github.com/example/code-review-skill"
tags: [code-review, github, pull-request, automation]
depends_on: [browser, github-workflow-automation]
supersedes: [legacy-code-review]
---
```

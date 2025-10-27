# Documentation Contributing Guidelines

This document explains how to contribute to VisionFlow documentation using the **Diátaxis** framework.

## 📐 The Diátaxis Framework

Our documentation is organized into four distinct types:

| Type | Purpose | Audience | Style |
|------|---------|----------|-------|
| **Getting Started** | Learn by doing | New users | Step-by-step, hands-on |
| **Guides** | Accomplish a goal | Experienced users | Problem-focused, goal-oriented |
| **Concepts** | Understand why | Learners seeking knowledge | Explanatory, background context |
| **Reference** | Look up information | Developers needing details | Technical, comprehensive, dry |

## 📍 Where to Put Documentation

### Getting Started (`getting-started/`)
✅ **Add here if:**
- You're writing a tutorial
- The content is learning-oriented
- You're introducing new concepts step-by-step
- Users are following along sequentially

**Examples:** Installation guide, "First Graph" tutorial, "Getting Started with Agents"

### Guides (`guides/user/` or `guides/developer/`)
✅ **Add here if:**
- You're solving a specific problem
- The content is goal-oriented
- Users might skip around
- You're answering "How do I...?"

**Examples:** "How to deploy to production", "Adding a custom agent", "Debugging issues"

### Concepts (`concepts/`)
✅ **Add here if:**
- You're explaining background knowledge
- The content builds understanding, not just procedures
- You're answering "Why does...?" or "What is...?"
- Users read this to understand architecture

**Examples:** "Hexagonal Architecture Explained", "How the GPU Compute works", "Understanding CQRS"

### Reference (`reference/`)
✅ **Add here if:**
- You're documenting an API
- You're providing complete technical specifications
- Information is looked up, not read sequentially
- The content is comprehensive and technical

**Examples:** API endpoints, protocol specifications, configuration options, schema definitions

## 🗂️ Directory Structure

```
docs/
├── README.md                          # Main entry point (DON'T EDIT unless updating framework)
├── CONTRIBUTING_DOCS.md               # This file
├── getting-started/
│   ├── 01-installation.md
│   ├── 02-first-graph.md
│   └── README.md                      # (optional) Overview of getting started
├── guides/
│   ├── README.md                      # (optional) Guide overview
│   ├── user/
│   │   ├── working-with-agents.md
│   │   └── xr-setup.md
│   └── developer/
│       ├── development-setup.md
│       ├── adding-a-feature.md
│       └── testing-guide.md
├── concepts/
│   ├── README.md                      # (optional) Concepts overview
│   ├── architecture.md
│   ├── agentic-workers.md
│   ├── gpu-compute.md
│   └── security-model.md
├── reference/
│   ├── api/
│   │   ├── README.md
│   │   ├── rest-api.md
│   │   ├── websocket-api.md
│   │   └── binary-protocol.md
│   ├── architecture/
│   │   ├── hexagonal-cqrs.md
│   │   ├── database-schema.md
│   │   └── actor-system.md
│   └── agents/
│       └── (well-organized agent reference)
├── deployment/
│   ├── README.md
│   └── (deployment guides)
├── archive/
│   ├── analysis/                      # Old analysis documents
│   ├── migration/                     # Migration planning (historical)
│   └── planning/                      # Planning documents (historical)
└── research/                          # (optional) Research and advanced topics
```

## ✏️ Writing Style Guidelines

### Consistency Across Sections

**Getting Started:**
- Use imperative mood: "Install the package", not "You should install"
- Be very explicit about each step
- Include expected output/verification
- Keep paragraphs short

**Guides:**
- Assume user has basic knowledge
- Focus on the task at hand
- Provide context but don't digress
- Include examples
- Link to Concepts for deeper understanding

**Concepts:**
- Explain the reasoning behind decisions
- Use diagrams where helpful (Mermaid preferred)
- Provide examples to illustrate ideas
- Can be more narrative and detailed

**Reference:**
- Be precise and complete
- Use tables and structured formats
- Include all parameters/options
- Keep prose minimal
- Use consistent formatting

## 🔗 Navigation Guidelines

Always include these in your file frontmatter or top section:

```markdown
# [Page Title]

*[Parent Category](../README.md) > [Section](./README.md) > [This Page]*

[description...]
```

Example:
```markdown
# Adding a Feature

*[Guides](../../README.md) > [Developer Guides](./README.md) > Adding a Feature*

This guide walks you through adding a new feature to VisionFlow.
```

## ✅ Quality Checklist

Before submitting documentation:

- [ ] File is in the correct directory (Getting Started/Guides/Concepts/Reference)
- [ ] File is correctly named (kebab-case, descriptive)
- [ ] File has navigation breadcrumbs at the top
- [ ] Content matches its section's style guide
- [ ] All code examples are tested and work
- [ ] All links are relative and verified
- [ ] Diagrams use Mermaid format where possible
- [ ] Ground truth is verified (port 3030, React, SQLite, etc.)
- [ ] File is formatted correctly (proper markdown, no spelling errors)
- [ ] Cross-references link to the correct sections

## 🚫 What NOT to Do

- ❌ Don't create new root-level `.md` files (use the structure above)
- ❌ Don't duplicate content across categories
- ❌ Don't mix procedural and explanatory content
- ❌ Don't create files in the wrong category
- ❌ Don't hardcode information that can drift (use links instead)
- ❌ Don't include outdated port numbers (always 3030, verify in code)

## 🔍 Before Committing

Run these checks to ensure documentation quality:

```bash
# Check for incorrect port references
grep -r "localhost:\(3001\|8080\)" docs/ --include="*.md" | grep -v archive | grep -v "docker"

# Check for broken links (relative paths)
# Check for monolithic files > 6000 words (split them up)
find docs -name "*.md" -exec wc -w {} \; | awk '$1 > 6000'

# Check for consistent formatting
# Check for missing breadcrumbs
grep -L "^\*\[" docs/**/*.md
```

## 📋 Updating the Main README

The main `docs/README.md` serves as the single entry point. Update it when:
1. Adding a new major section
2. Changing the framework or structure
3. Moving a file to a different category

**Don't edit the main README for:**
- Adding individual pages (those should auto-organize)
- Fixing typos in individual pages (edit the page itself)

## 🏗️ Evolution of Documentation

This structure is designed to grow:
- **New Getting Started**: Add to `getting-started/`
- **New User Guide**: Add to `guides/user/`
- **New Developer Guide**: Add to `guides/developer/`
- **New Concept**: Add to `concepts/`
- **New API Reference**: Add to `reference/api/`
- **New Architecture Details**: Add to `reference/architecture/`

## 📞 Questions?

If you're unsure where something belongs:
1. Check if it's procedural (guides), explanatory (concepts), or technical (reference)
2. Read similar files in each category to feel the tone
3. When in doubt, err toward the more specific category

---

**Last Updated**: 2025-10-27  
**Framework**: Diátaxis  
**Status**: Active

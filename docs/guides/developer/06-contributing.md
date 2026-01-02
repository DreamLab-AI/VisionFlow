---
layout: default
title: Contributing Guidelines
parent: Developer
grand_parent: Guides
nav_order: 5
description: Guidelines for contributing to VisionFlow including code style and PR process
---


# Contributing Guidelines

## Welcome Contributors!

We appreciate your interest in contributing to VisionFlow. This guide will help you get started.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Submit a pull request

## Development Process

### 1. Create an Issue

Before starting work, create an issue describing:
- Problem or feature request
- Proposed solution
- Implementation approach

### 2. Discuss Approach

Wait for maintainer feedback before implementing large changes.

### 3. Branch Naming

```bash
feature/add-notifications
fix/resolve-upload-bug
docs/update-api-guide
refactor/improve-auth
```

### 4. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(api): add notification endpoints
fix(web): resolve file upload timeout
docs(readme): update installation steps
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### 5. Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure CI passes
4. Request review
5. Address feedback
6. Merge when approved

## Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No new warnings
- [ ] Tests pass locally
```

## Code Style

- Follow existing patterns
- Use ESLint and Prettier
- Write meaningful comments
- Keep functions small
- Avoid magic numbers

---

---

## Related Documentation

- [Semantic Forces User Guide](../features/semantic-forces.md)
- [Goalie Integration - Goal-Oriented AI Research](../infrastructure/goalie-integration.md)
- [Project Structure](02-project-structure.md)
- [Natural Language Queries Tutorial](../features/natural-language-queries.md)
- [Intelligent Pathfinding Guide](../features/intelligent-pathfinding.md)

## Questions?

- GitHub Issues
- Community Forum
- Email: dev@visionflow.example

Thank you for contributing!

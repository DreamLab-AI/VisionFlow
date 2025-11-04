# Documentation Link Fix - Quick Reference Guide

## 🎯 Top 10 Missing Files (Fix These First)

### 1. `docs/readme.md` - Main Documentation Hub
**Referenced by**: 21 files
**Files that link here**:
- `docs/getting-started/01-installation.md` (6 references)
- `docs/getting-started/02-first-graph-and-agents.md` (4 references)
- `docs/guides/contributing.md` (2 references)
- `docs/guides/migration/json-to-binary-protocol.md`
- All navigation guides

**Quick Fix**:
```bash
cat > docs/readme.md << 'EOF'
# VisionFlow Documentation

## Quick Links
- [Getting Started](getting-started/01-installation.md)
- [API Reference](reference/api/readme.md)
- [Guides](guides/readme.md)
- [Architecture](concepts/architecture/hexagonal-cqrs-architecture.md)
- [Multi-Agent Docker](multi-agent-docker/readme.md)

## Documentation Sections
- **Getting Started**: Installation and first steps
- **Guides**: How-to guides for common tasks
- **Reference**: API and configuration reference
- **Concepts**: Architecture and design concepts
- **Multi-Agent Docker**: Container environment setup
EOF
```

### 2. `docs/index.md` - Knowledge Base Entry
**Referenced by**: 10 files
**Files that link here**:
- `docs/guides/deployment.md`
- `docs/guides/development-workflow.md`
- `docs/guides/extending-the-system.md`
- `docs/guides/orchestrating-agents.md`
- `docs/guides/troubleshooting.md`
- `docs/guides/working-with-gui-sandbox.md`
- `docs/guides/xr-setup.md`

**Quick Fix**:
```bash
# Symlink to readme.md or create separate index
ln -s readme.md docs/index.md
```

### 3. `docs/reference/configuration.md` - Configuration Reference
**Referenced by**: 9 files
**Files that link here**:
- `docs/getting-started/01-installation.md` (2 references)
- `docs/getting-started/02-first-graph-and-agents.md` (3 references)
- `docs/guides/configuration.md` (3 references)
- `docs/guides/deployment.md`
- `docs/guides/troubleshooting.md`

**Quick Fix**:
```bash
mkdir -p docs/reference
cat > docs/reference/configuration.md << 'EOF'
# Configuration Reference

## Environment Variables

### Server Configuration
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `PORT` - Server port (default: 8080)
- `HOST` - Server host (default: 0.0.0.0)

### GPU Configuration
- `CUDA_VISIBLE_DEVICES` - GPU device selection
- `GPU_MEMORY_FRACTION` - Memory allocation limit

### Authentication
- `JWT_SECRET` - JWT signing secret
- `SESSION_SECRET` - Session encryption key

See `.env.example` for complete configuration options.
EOF
```

### 4. `docs/concepts/architecture/00-ARCHITECTURE-overview.md`
**Referenced by**: 15 files
**Files that link here**:
- `docs/concepts/architecture/ontology-storage-architecture.md`
- `docs/concepts/architecture/xr-immersive-system.md`
- `docs/guides/migration/json-to-binary-protocol.md`
- `docs/guides/navigation-guide.md` (4 references)
- `docs/reference/api/03-websocket.md`
- `docs/reference/error-codes.md`

**Quick Fix**:
```bash
# This file is referenced as 00-ARCHITECTURE-overview.md
# Check if it exists with different casing
ls -la docs/concepts/architecture/ | grep -i architecture

# If 00-architecture-overview.md exists, create symlink
cd docs/concepts/architecture
ln -s 00-architecture-overview.md 00-ARCHITECTURE-overview.md
```

### 5. `docs/reference/readme.md` - Reference Hub
**Referenced by**: 7 files
**Files that link here**:
- `docs/guides/extending-the-system.md`
- `docs/guides/index.md`
- `docs/guides/orchestrating-agents.md`
- `docs/guides/user/working-with-agents.md` (2 references)

**Quick Fix**:
```bash
cat > docs/reference/readme.md << 'EOF'
# Reference Documentation

## API Reference
- [REST API Complete](api/rest-api-complete.md)
- [WebSocket API](api/03-websocket.md)
- [Authentication](api/01-authentication.md)

## Configuration
- [Configuration Reference](configuration.md)
- [Error Codes](error-codes.md)

## Architecture
- [Hexagonal CQRS Architecture](../concepts/architecture/hexagonal-cqrs-architecture.md)
- [System Architecture](../concepts/architecture/hexagonal-cqrs-architecture.md)

## Ontology
- [Semantic Physics](semantic-physics-implementation.md)
- [WebSocket Protocol](websocket-protocol.md)
EOF
```

### 6. `docs/reference/architecture/readme.md`
**Referenced by**: 12 files
**Files that link here**:
- `docs/guides/security.md`
- `docs/guides/troubleshooting.md`
- `docs/guides/vircadia-multi-user-guide.md`
- `docs/guides/working-with-gui-sandbox.md`
- `docs/multi-agent-docker/port-configuration.md`
- `docs/multi-agent-docker/readme.md` (4 references)

**Quick Fix**:
```bash
mkdir -p docs/reference/architecture
cat > docs/reference/architecture/readme.md << 'EOF'
# Architecture Reference

This section has been moved to:
- [Architecture Overview](../../concepts/architecture/hexagonal-cqrs-architecture.md)
- [Multi-Agent Docker](../../multi-agent-docker/architecture.md)

## Quick Links
- [System Architecture](../../concepts/architecture/hexagonal-cqrs-architecture.md)
- [Database Schemas](../../concepts/architecture/04-database-schemas.md)
- [Multi-Agent Setup](../../multi-agent-docker/readme.md)
EOF
```

### 7. `docs/reference/agents/templates/index.md`
**Referenced by**: 11 references across 2 files
**Files that link here**:
- `docs/guides/extending-the-system.md` (9 references)

**Quick Fix**:
```bash
mkdir -p docs/reference/agents/templates
cat > docs/reference/agents/templates/index.md << 'EOF'
# Agent Templates

## Available Templates

### Core Agents
- [Automation Smart Agent](automation-smart-agent.md) - Automated task execution
- [SPARC Coder](implementer-sparc-coder.md) - SPARC methodology implementation
- [Task Orchestrator](orchestrator-task.md) - Task coordination
- [Memory Coordinator](memory-coordinator.md) - Memory management
- [GitHub PR Manager](github-pr-manager.md) - PR automation

## Creating Custom Templates
See [Extending the System](../../../guides/extending-the-system.md) for details.
EOF

# Create stub files
for template in automation-smart-agent implementer-sparc-coder orchestrator-task memory-coordinator github-pr-manager; do
cat > docs/reference/agents/templates/${template}.md << EOF
# ${template}

Documentation in progress.

See: [Agent Templates Index](index.md)
EOF
done
```

### 8. `docs/concepts/system-architecture.md`
**Referenced by**: 6 files
**Files that link here**:
- `docs/getting-started/01-installation.md` (2 references)
- `docs/getting-started/02-first-graph-and-agents.md` (2 references)
- `docs/guides/navigation-guide.md` (2 references)

**Quick Fix**:
```bash
# Likely should redirect to hexagonal architecture
cd docs/concepts
ln -s architecture/hexagonal-cqrs-architecture.md system-architecture.md
```

### 9. `docs/getting-started/readme.md`
**Referenced by**: 2 files
**Files that link here**:
- `docs/getting-started/01-installation.md` (2 references)

**Quick Fix**:
```bash
cat > docs/getting-started/readme.md << 'EOF'
# Getting Started with VisionFlow

## Quick Start
1. [Installation](01-installation.md) - Install and configure VisionFlow
2. [First Graph and Agents](02-first-graph-and-agents.md) - Create your first project

## Next Steps
- [Configuration Guide](../guides/configuration.md)
- [Development Workflow](../guides/development-workflow.md)
- [Architecture Overview](../concepts/architecture/hexagonal-cqrs-architecture.md)
EOF
```

### 10. `docs/reference/agents/readme.md`
**Referenced by**: 2 files
**Files that link here**:
- `docs/guides/navigation-guide.md` (2 references)

**Quick Fix**:
```bash
mkdir -p docs/reference/agents
cat > docs/reference/agents/readme.md << 'EOF'
# Agent Reference

## Agent Templates
See [Agent Templates](templates/index.md) for available templates.

## Agent Types
- **Core Development**: coder, reviewer, tester
- **Swarm Coordination**: hierarchical-coordinator, mesh-coordinator
- **GitHub Integration**: pr-manager, issue-tracker
- **SPARC Methodology**: sparc-coord, specification, architecture

## Documentation
- [Orchestrating Agents](../../guides/orchestrating-agents.md)
- [Extending the System](../../guides/extending-the-system.md)
EOF
```

---

## 🔧 Batch Fix Commands

### Create All Missing Index Files
```bash
#!/bin/bash
# Run from project root

# Create main docs hub
cat > docs/readme.md << 'EOF'
# VisionFlow Documentation
See [link-fix-quick-reference.md](link-fix-quick-reference.md) for content.
EOF

# Create index (symlink to readme)
cd docs && ln -s readme.md index.md && cd ..

# Create reference section
mkdir -p docs/reference
cat > docs/reference/readme.md << 'EOF'
# Reference Documentation
See link-fix-quick-reference.md for content.
EOF

cat > docs/reference/configuration.md << 'EOF'
# Configuration Reference
See link-fix-quick-reference.md for content.
EOF

# Create agent templates
mkdir -p docs/reference/agents/templates
cat > docs/reference/agents/readme.md << 'EOF'
# Agent Reference
See link-fix-quick-reference.md for content.
EOF

cat > docs/reference/agents/templates/index.md << 'EOF'
# Agent Templates
See link-fix-quick-reference.md for content.
EOF

# Create stub agent templates
for template in automation-smart-agent implementer-sparc-coder orchestrator-task memory-coordinator github-pr-manager; do
  echo "# ${template}" > docs/reference/agents/templates/${template}.md
done

# Create architecture references
mkdir -p docs/reference/architecture
cat > docs/reference/architecture/readme.md << 'EOF'
# Architecture Reference (Moved)
See ../../concepts/architecture/hexagonal-cqrs-architecture.md
EOF

# Create getting started index
cat > docs/getting-started/readme.md << 'EOF'
# Getting Started
See link-fix-quick-reference.md for content.
EOF

# Create system architecture symlink
cd docs/concepts
ln -sf architecture/hexagonal-cqrs-architecture.md system-architecture.md
cd ../..

echo "✅ Created all missing index files"
echo "⚠️  Remember to fill in actual content!"
```

---

## 📝 Common Path Fix Patterns

### Pattern 1: `docs/src/` → `../../src/`
**Issue**: Code references from `docs/concepts/architecture/` incorrectly use `../../src/`
**Fix**: Should be `../../../src/` (go up 3 levels)

```bash
# Find affected files
grep -r "../../src/" docs/concepts/architecture/

# Fix pattern (BE CAREFUL - test first!)
# find docs/concepts/architecture -name "*.md" -exec sed -i 's|../../src/|../../../src/|g' {} \;
```

### Pattern 2: `../api/` from `docs/concepts/`
**Issue**: Links from `docs/concepts/architecture/` to `../api/` expect `docs/concepts/api/`
**Fix**: Should be `../../reference/api/`

```bash
# Find affected files
grep -r "\.\./api/" docs/concepts/

# Manual fix required - check each case
```

### Pattern 3: `/project/` prefix
**Issue**: Absolute paths with `/project/` prefix
**Fix**: Remove `/project/` or use relative paths

```bash
# Find files with /project/ paths
grep -r "/project/" docs/

# These need manual review
```

---

## ✅ Validation

After fixing, run:
```bash
python3 scripts/analyze_doc_links.py
```

Expected improvement:
- **Before**: 32.5% health, 257 broken links
- **After Priority 1-2**: ~85% health, ~40 broken links
- **After all fixes**: ~95% health, <10 broken links

---

*Quick Reference for fixing top documentation link issues*
*See: link-analysis-executive-summary.md for full analysis*

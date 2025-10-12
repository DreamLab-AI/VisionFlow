# Upstream Merge Strategy for CachyOS Workstation

This document explains how to maintain your custom CachyOS workstation configuration while staying synchronized with upstream Agentic Flow updates.

## Overview

The CachyOS workstation is isolated in `docker/cachyos/` to avoid conflicts with upstream changes. This allows you to:

- ✅ Pull upstream updates safely
- ✅ Maintain custom configuration
- ✅ Merge selectively
- ✅ Keep your modifications separate

---

## Directory Structure

### Custom Configuration (Not Tracked Upstream)

```
docker/cachyos/                    # Your custom workstation
├── Dockerfile.workstation         # CachyOS-specific
├── docker-compose.workstation.yml # Custom compose config
├── config/
│   ├── .zshrc                     # Custom shell config
│   ├── router.config.json         # Model router config
│   ├── providers.env.template     # API key template
│   └── claude-flow.json           # MCP configuration
├── scripts/
│   ├── init-workstation.sh        # Startup script
│   └── test-all-providers.sh      # Testing script
├── README.workstation.md          # Custom documentation
├── UPSTREAM.md                    # This file
└── .env                           # Your API keys (gitignored)
```

### Upstream Files (Will Be Updated)

```
docker/
├── base/                          # Upstream base images
├── cloud-run/                     # Upstream GCP deployment
├── configs/                       # Upstream config templates
├── Dockerfile.test-*              # Upstream test containers
├── docker-compose.yml             # Upstream compose config
└── README.md                      # Upstream docker documentation
```

---

## Git Configuration

### 1. Add Upstream Remote

```bash
cd /mnt/mldata/githubs/agentic-flow

# Add upstream (if not already added)
git remote add upstream https://github.com/ruvnet/agentic-flow.git

# Verify remotes
git remote -v
# origin    your-fork-url (fetch)
# origin    your-fork-url (push)
# upstream  https://github.com/ruvnet/agentic-flow.git (fetch)
# upstream  https://github.com/ruvnet/agentic-flow.git (push)
```

### 2. Configure Gitignore

Ensure your `.gitignore` includes:

```bash
# Add to .gitignore (if not present)
cat >> .gitignore <<EOF

# CachyOS workstation private files
docker/cachyos/.env
docker/cachyos/**/*.local.*
docker/cachyos/**/secrets/

# Docker volumes
/var/lib/docker/volumes/agentic-cachyos-*
EOF
```

---

## Merging Upstream Updates

### Step 1: Fetch Upstream Changes

```bash
# Fetch latest upstream
git fetch upstream

# Check what changed
git log HEAD..upstream/main --oneline
```

### Step 2: Review Changes

```bash
# See what files changed upstream
git diff HEAD..upstream/main --name-only

# Focus on docker directory
git diff HEAD..upstream/main --name-only -- docker/

# Check if cachyos directory is affected (should be none)
git diff HEAD..upstream/main --name-only -- docker/cachyos/
```

### Step 3: Merge Selectively

#### Option A: Merge Everything (Recommended)

```bash
# Create a backup branch first
git checkout -b backup-$(date +%Y%m%d)
git checkout main

# Merge upstream
git merge upstream/main --no-commit

# Check for conflicts
git status

# If conflicts exist in cachyos directory, resolve them
# (Usually there won't be any since it's custom)

# Commit the merge
git commit -m "Merge upstream updates from agentic-flow"
```

#### Option B: Cherry-Pick Specific Commits

```bash
# List upstream commits
git log upstream/main --oneline -20

# Cherry-pick specific commits
git cherry-pick <commit-hash>

# Or cherry-pick a range
git cherry-pick <start-hash>..<end-hash>
```

### Step 4: Test After Merge

```bash
# Rebuild container
cd docker/cachyos
docker-compose -f docker-compose.workstation.yml build --no-cache

# Start container
docker-compose -f docker-compose.workstation.yml up -d

# Test
docker exec -it agentic-flow-cachyos zsh -c "test-providers"
```

---

## Handling Merge Conflicts

### Common Conflict Scenarios

#### 1. Docker Compose Files

**Conflict**: `docker/docker-compose.yml` updated upstream

**Resolution**: Keep both files separate

```bash
# Upstream changes are in docker/docker-compose.yml
# Your changes are in docker/cachyos/docker-compose.workstation.yml
# No conflict - they're different files
```

#### 2. Dockerfile Changes

**Conflict**: `docker/base/Dockerfile` updated upstream

**Resolution**: Review and optionally incorporate changes

```bash
# Compare differences
git diff upstream/main:docker/base/Dockerfile docker/cachyos/Dockerfile.workstation

# Manually incorporate useful changes
vim docker/cachyos/Dockerfile.workstation
```

#### 3. Configuration Templates

**Conflict**: `docker/configs/*.env.template` updated upstream

**Resolution**: Compare and update your templates

```bash
# Check what changed
git diff HEAD:docker/configs/claude.env.template upstream/main:docker/configs/claude.env.template

# Update your template if needed
vim docker/cachyos/config/providers.env.template
```

---

## Update Workflow

### Regular Update Schedule

**Recommended**: Check for upstream updates weekly

```bash
#!/bin/bash
# Save as: docker/cachyos/scripts/update-from-upstream.sh

echo "Checking for upstream updates..."

# Fetch upstream
git fetch upstream

# Check if updates exist
UPDATES=$(git log HEAD..upstream/main --oneline | wc -l)

if [ $UPDATES -eq 0 ]; then
    echo "✅ No upstream updates available"
    exit 0
fi

echo "📦 $UPDATES new commits upstream:"
git log HEAD..upstream/main --oneline

echo ""
echo "Review changes with:"
echo "  git diff HEAD..upstream/main"
echo ""
echo "Merge with:"
echo "  git merge upstream/main"
```

Make it executable:

```bash
chmod +x docker/cachyos/scripts/update-from-upstream.sh
```

---

## Preserving Custom Changes

### 1. Keep Configuration Separate

Store all custom configuration in `docker/cachyos/config/`:

- `.zshrc` - Custom shell
- `router.config.json` - Model routing
- `providers.env.template` - API keys template
- `claude-flow.json` - MCP servers

### 2. Use Environment Variables

Override upstream defaults with environment variables:

```bash
# In your .env file
PRIMARY_PROVIDER=gemini
ROUTER_MODE=performance
GPU_ACCELERATION=true
```

### 3. Extend, Don't Modify

Rather than modifying upstream files, extend them:

```bash
# Don't modify: docker/base/Dockerfile
# Instead create: docker/cachyos/Dockerfile.workstation

FROM cachyos/base:latest  # Your custom base
# Add your customizations
```

---

## Advanced: Subtree Strategy

For more complex scenarios, use git subtree:

### Setup

```bash
# Add upstream as subtree
git subtree add --prefix=docker/upstream \
    https://github.com/ruvnet/agentic-flow.git main --squash
```

### Update

```bash
# Pull upstream updates into subtree
git subtree pull --prefix=docker/upstream \
    https://github.com/ruvnet/agentic-flow.git main --squash
```

### Benefit

This keeps upstream code in a separate directory (`docker/upstream/`) while your custom code stays in `docker/cachyos/`.

---

## Rollback Strategy

### If Something Breaks After Merge

#### Option 1: Revert the Merge

```bash
# Find the merge commit
git log --oneline -10

# Revert the merge
git revert -m 1 <merge-commit-hash>
```

#### Option 2: Reset to Before Merge

```bash
# Check reflog
git reflog

# Reset to before merge
git reset --hard HEAD@{1}
```

#### Option 3: Use Backup Branch

```bash
# Switch to backup branch
git checkout backup-20251011

# Create new branch from backup
git checkout -b main-restored
git branch -D main
git branch -m main
```

---

## Testing Changes

### Automated Testing Script

Create `docker/cachyos/scripts/test-after-update.sh`:

```bash
#!/bin/bash
set -e

echo "🧪 Testing Agentic Flow Workstation After Update"

# Build fresh image
docker-compose -f docker-compose.workstation.yml build

# Start container
docker-compose -f docker-compose.workstation.yml up -d

# Wait for initialization
sleep 10

# Run tests
echo "Testing providers..."
docker exec agentic-flow-cachyos test-providers

echo "Testing GPU..."
docker exec agentic-flow-cachyos test-gpu

echo "Testing MCP servers..."
docker exec agentic-flow-cachyos mcp-status

echo "✅ All tests passed!"
```

---

## Best Practices

### 1. Always Backup Before Merging

```bash
# Create backup branch
git checkout -b backup-$(date +%Y%m%d-%H%M%S)
git checkout main
```

### 2. Review Changes Before Merging

```bash
# Review all changes
git diff HEAD..upstream/main

# Focus on critical files
git diff HEAD..upstream/main -- docker/ agentic-flow/
```

### 3. Test in Isolated Environment

```bash
# Create test branch
git checkout -b test-upstream-merge
git merge upstream/main

# Test thoroughly
cd docker/cachyos && ./scripts/test-after-update.sh

# If successful, merge to main
git checkout main
git merge test-upstream-merge
```

### 4. Document Custom Changes

Keep a `CHANGES.md` in `docker/cachyos/`:

```markdown
# Custom Changes Log

## 2025-01-11
- Added CachyOS workstation configuration
- Integrated with RAGFlow network (Xinference)
- Added intelligent model router
- Configured GPU passthrough

## 2025-01-15
- Updated router.config.json with new Gemini models
- Added cost tracking to .zshrc aliases
```

---

## Maintaining Compatibility

### API Version Pinning

In your `.env`:

```bash
# Pin specific versions to avoid breaking changes
AGENTIC_FLOW_VERSION=1.4.1
CLAUDE_FLOW_VERSION=latest
```

### Feature Flags

Use feature flags for experimental features:

```bash
# In docker/cachyos/config/router.config.json
{
  "experimentalFeatures": {
    "multiModal": false,
    "streaming": true,
    "toolCalling": true
  }
}
```

---

## Troubleshooting Merge Issues

### Issue: Container Won't Build After Merge

```bash
# Check Docker build logs
docker-compose -f docker-compose.workstation.yml build 2>&1 | tee build.log

# Common fixes:
# 1. Clear Docker cache
docker builder prune -af

# 2. Rebuild without cache
docker-compose -f docker-compose.workstation.yml build --no-cache

# 3. Check for breaking changes in upstream Dockerfile
git diff HEAD^:docker/base/Dockerfile HEAD:docker/base/Dockerfile
```

### Issue: API Keys Not Working

```bash
# Verify .env file
cat .env | grep -E "(API_KEY|BASE_URL)"

# Rebuild with fresh .env
docker-compose -f docker-compose.workstation.yml down
docker-compose -f docker-compose.workstation.yml up -d --build
```

### Issue: MCP Servers Failing

```bash
# Check MCP logs
docker exec agentic-flow-cachyos cat /tmp/mcp-startup.log

# Manually restart MCP
docker exec agentic-flow-cachyos mcp-stop
docker exec agentic-flow-cachyos mcp-start
```

---

## Contributing Back Upstream

If you develop a feature in your CachyOS workstation that could benefit the upstream project:

### 1. Create a Fork

```bash
# Fork on GitHub UI, then:
git remote add myfork https://github.com/YOUR-USERNAME/agentic-flow.git
```

### 2. Create Feature Branch

```bash
git checkout -b feature/cachyos-workstation
```

### 3. Commit Changes

```bash
git add docker/cachyos/
git commit -m "feat: Add CachyOS workstation configuration

- CachyOS-based Docker workstation
- Intelligent model router
- RAGFlow network integration
- GPU support with NVIDIA/AMD
- 6 model providers
- Interactive CLI environment"
```

### 4. Push and Create PR

```bash
git push myfork feature/cachyos-workstation

# Create PR on GitHub UI
```

---

## Summary

**Key Principles:**

1. ✅ Keep custom code in `docker/cachyos/`
2. ✅ Never modify upstream files directly
3. ✅ Use `.gitignore` for private files
4. ✅ Test after every merge
5. ✅ Backup before merging
6. ✅ Review changes carefully
7. ✅ Document custom modifications

**Quick Reference:**

```bash
# Check for updates
git fetch upstream && git log HEAD..upstream/main --oneline

# Merge updates
git merge upstream/main

# Test after merge
cd docker/cachyos && ./scripts/test-after-update.sh

# Rollback if needed
git reset --hard HEAD^
```

---

For questions or issues, open an issue at: https://github.com/ruvnet/agentic-flow/issues

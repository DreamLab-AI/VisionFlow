# Enhanced Entrypoint Script

## Location
```
/home/devuser/workspace/project/scripts/entrypoint-unified-enhanced.sh
```

## What It Does

This enhanced entrypoint script:

1. ✅ **Runs all original setup** (directories, credentials, services)
2. ✅ **Initializes Claude Flow** (`claude-flow init --force` as devuser)
3. ✅ **Enhances CLAUDE.md** with compact project-specific documentation:
   - 610 Claude Sub-Agents reference
   - Z.AI Service usage
   - Gemini Flow commands
   - Multi-User System table
   - tmux workspace layout
   - Management API endpoints
   - Diagnostic commands
   - Service ports

## How to Use

### Copy to Host System

```bash
# From host machine, copy this script
cp /path/to/workspace/project/scripts/entrypoint-unified-enhanced.sh \
   /path/to/multi-agent-docker/unified-config/

# Update Dockerfile.unified to use it
# Change the COPY line:
COPY unified-config/entrypoint-unified-enhanced.sh /usr/local/bin/entrypoint.sh
```

### Update Dockerfile

In your `Dockerfile.unified`, ensure:

```dockerfile
# Copy enhanced entrypoint
COPY unified-config/entrypoint-unified-enhanced.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set as container entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

### Rebuild Container

```bash
cd /path/to/turbo-flow-claude
docker build -f Dockerfile.unified -t turbo-flow-unified:latest .
docker-compose -f docker-compose.unified.yml up -d
```

## What Gets Appended to CLAUDE.md

The script appends ~100 lines of compact documentation to `/home/devuser/CLAUDE.md`:

- **610 Sub-Agents**: GitHub link + location
- **Z.AI Service**: Port, curl examples, user switching
- **Gemini Flow**: All gf-* commands
- **Multi-User System**: 4-user table with access methods
- **tmux Workspace**: 8-window layout reference
- **Management API**: Key endpoints with auth
- **Diagnostics**: Essential troubleshooting commands
- **Service Ports**: Complete port mapping table
- **Security Notes**: Default credentials warning

## Testing

After container starts, verify the enhancement:

```bash
# Inside container
cat /home/devuser/CLAUDE.md | tail -100

# Should see project-specific sections starting with:
# "## 🚀 Project-Specific: Turbo Flow Claude"
```

## Benefits

- ✅ Claude Code gets full project context automatically
- ✅ No manual documentation copying needed
- ✅ 610 agents integrated via reference
- ✅ All services documented compactly
- ✅ Ready for production use

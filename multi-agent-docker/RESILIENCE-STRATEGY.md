# CLAUDE.md Resilience Strategy

## Problem
Claude can regenerate/overwrite CLAUDE.md, removing system tool awareness.

## Solution: Multi-Layer Protection

### Layer 1: Watcher Service (Primary)
**File**: `core-assets/scripts/claude-md-watcher.sh`
**Service**: `claude-md-watcher` in supervisord
**Mechanism**:
- Checks every 30 seconds
- Detects missing `<!-- SYSTEM_TOOLS_MANIFEST -->` marker
- Auto-reapplies patch via `claude-md-patcher.sh`
- Logs to `/app/mcp-logs/claude-md-watcher.log`

### Layer 2: Checksum Verification
**File**: `core-assets/config/.tools-manifest.lock`
**Mechanism**:
- Stores MD5 checksum of patched CLAUDE.md
- Version tracking for manifest updates
- Timestamp of last application
- Prevents duplicate patching

### Layer 3: Idempotent Patcher
**File**: `core-assets/scripts/claude-md-patcher.sh`
**Mechanism**:
- Marker-based detection (HTML comment)
- Safe to run multiple times
- Updates lock file on each run
- Minimal 8-line addition (~150 tokens)

### Layer 4: Immutable Reference
**File**: `core-assets/config/SYSTEM_TOOLS.md`
**Mechanism**:
- Source of truth for tool descriptions
- Never modified by runtime
- Can be read by agents if CLAUDE.md lost
- Fallback documentation

## Recovery Time
- **Detection**: ≤30 seconds (watcher interval)
- **Repair**: <1 second (patch application)
- **Total**: ≤31 seconds to restore awareness

## Manual Recovery
```bash
# Check if manifest present
grep "SYSTEM_TOOLS_MANIFEST" /workspace/CLAUDE.md

# Manually reapply
/app/core-assets/scripts/claude-md-patcher.sh

# Check watcher status
supervisorctl status claude-md-watcher

# View watcher logs
tail -f /app/mcp-logs/claude-md-watcher.log
```

## Token Efficiency
- **Original CLAUDE.md additions**: 2000+ tokens (verbose)
- **New compact manifest**: ~150 tokens (93% reduction)
- **Restoration overhead**: Negligible (~1 token/sec watcher)

## Design Principles
1. **Idempotent**: Safe to apply repeatedly
2. **Minimal**: Smallest possible footprint
3. **Automatic**: No manual intervention needed
4. **Auditable**: Lock file tracks state
5. **Recoverable**: Multiple fallback layers

## Monitoring
```bash
# Check last patch time
cat /app/core-assets/config/.tools-manifest.lock

# Verify marker exists
grep -c "SYSTEM_TOOLS_MANIFEST" /workspace/CLAUDE.md

# Watch for changes
watch -n 5 'tail -3 /app/mcp-logs/claude-md-watcher.log'
```
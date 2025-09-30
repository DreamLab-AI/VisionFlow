#!/bin/bash
# Claude.md Token-Efficient System Awareness Patcher
# Appends ultra-compact tool manifest reference instead of verbose documentation
# Idempotent and checksum-verified for resilient re-application

CLAUDE_MD="/workspace/CLAUDE.md"
MANIFEST="/app/core-assets/config/tools-manifest.json"
MARKER="<!-- SYSTEM_TOOLS_MANIFEST -->"
LOCK_FILE="/app/core-assets/config/.tools-manifest.lock"

if [ ! -f "$CLAUDE_MD" ]; then
    echo "CLAUDE.md not found, skipping patch"
    exit 0
fi

# Calculate checksum of current CLAUDE.md
current_checksum=$(md5sum "$CLAUDE_MD" 2>/dev/null | cut -d' ' -f1)

# Check if already patched
if grep -q "$MARKER" "$CLAUDE_MD"; then
    # Update lock file
    echo "VERSION=1.0.0" > "$LOCK_FILE"
    echo "LAST_APPLIED=$(date +%s)" >> "$LOCK_FILE"
    echo "CHECKSUM=$current_checksum" >> "$LOCK_FILE"
    echo "CLAUDE.md already patched"
    exit 0
fi

# Append ultra-compact reference
cat >> "$CLAUDE_MD" << 'EOF'

<!-- SYSTEM_TOOLS_MANIFEST -->
## Available Tools

**Research**: `goalie search "query"` - Deep research with GOAP (20-30 sources, $0.006-0.10)
**Agents**: `claude-flow goal|neural` - AI orchestration
**Browser**: `playwright` - Automation (headless + VNC:5901)
**Graphics**: Blender(9876), QGIS(9877), PBR(9878) - All on VNC:5901

Full manifest: `cat /app/core-assets/config/tools-manifest.json`
Goalie docs: `/app/GOALIE-INTEGRATION.md`
EOF

# Update lock file
new_checksum=$(md5sum "$CLAUDE_MD" 2>/dev/null | cut -d' ' -f1)
echo "VERSION=1.0.0" > "$LOCK_FILE"
echo "LAST_APPLIED=$(date +%s)" >> "$LOCK_FILE"
echo "CHECKSUM=$new_checksum" >> "$LOCK_FILE"

echo "CLAUDE.md patched with compact tool manifest"
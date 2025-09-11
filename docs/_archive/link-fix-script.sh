#!/bin/bash

# VisionFlow Documentation Link Fix Script
# This script fixes broken internal markdown links

echo "VisionFlow Documentation Link Fix Script"
echo "========================================"
echo

# Function to fix links in a file
fix_links_in_file() {
    local file="$1"
    echo "Fixing links in: $(basename "$file")"
    
    # Create a backup
    cp "$file" "$file.backup"
    
    # Fix patterns of broken links
    
    # 1. Fix missing configuration.md references - now that file exists
    sed -i 's|](./getting-started/configuration\.md|](./getting-started/configuration.md|g' "$file"
    
    # 2. Fix references to AGENT_TYPE_CONVENTIONS.md (moved to reference/agents/conventions.md)
    sed -i 's|]\.\./AGENT_TYPE_CONVENTIONS\.md)|../reference/agents/conventions.md)|g' "$file"
    sed -i 's|]\(\./AGENT_TYPE_CONVENTIONS\.md)|./reference/agents/conventions.md)|g' "$file"
    sed -i 's|]\(\.\./\.\./\.\./AGENT_TYPE_CONVENTIONS\.md)|../../../reference/agents/conventions.md)|g' "$file"
    
    # 3. Fix missing troubleshooting.md references - now that file exists
    sed -i 's|](../troubleshooting\.md)|](../troubleshooting.md)|g' "$file"
    sed -i 's|](./troubleshooting\.md)|](./troubleshooting.md)|g' "$file"
    
    # 4. Fix missing contributing.md references - now that file exists
    sed -i 's|](../contributing\.md)|](../contributing.md)|g' "$file"
    sed -i 's|](./contributing\.md)|](./contributing.md)|g' "$file"
    
    # 5. Fix references to missing files that should point to existing alternatives
    # MCP_AGENT_VISUALIZATION.md doesn't exist - remove broken links or point to existing docs
    sed -i 's|]\(\.\./MCP_AGENT_VISUALIZATION\.md)|../api/multi-mcp-visualization-api.md)|g' "$file"
    sed -i 's|]\(\.\./\.\./\.\./MCP_AGENT_VISUALIZATION\.md)|../../../api/multi-mcp-visualization-api.md)|g' "$file"
    
    # multi-mcp-agent-visualization.md doesn't exist - same fix
    sed -i 's|]\(\.\./multi-mcp-agent-visualization\.md)|../api/multi-mcp-visualization-api.md)|g' "$file"
    sed -i 's|]\(\.\./\.\./\.\./multi-mcp-agent-visualization\.md)|../../../api/multi-mcp-visualization-api.md)|g' "$file"
    
    # 6. Fix references to missing agent-visualization-architecture.md
    sed -i 's|]\(\.\./agent-visualization-architecture\.md)|../architecture/system-overview.md)|g' "$file"
    sed -i 's|]\(\.\./\.\./\.\./agent-visualization-architecture\.md)|../../../architecture/system-overview.md)|g' "$file"
    
    # 7. Fix references to missing architecture_analysis_report.md
    sed -i 's|]\(\.\./architecture_analysis_report\.md)|../architecture/system-overview.md)|g' "$file"
    
    # 8. Fix references to missing migration-guide.md
    sed -i 's|]\(\.\./architecture/migration-guide\.md)|../architecture/system-overview.md)|g' "$file"
    
    echo "  ✓ Fixed links in $(basename "$file")"
}

# Find and fix all markdown files (except those in _archive)
find /workspace/ext/docs -name "*.md" -not -path "/workspace/ext/docs/_archive/*" | while read -r file; do
    fix_links_in_file "$file"
done

echo
echo "========================================"
echo "Link fixing complete!"
echo "Backup files created with .backup extension"
echo

# Run the link audit again to see improvements
echo "Running link audit to verify fixes..."
echo
/workspace/ext/docs/_archive/link-audit-script.sh
#!/bin/bash
#
# MCP Tool Management CLI
# Simplifies adding, removing, and managing MCP tools
#

set -e

MCP_CONFIG="/home/devuser/.config/claude/mcp.json"
TEMP_CONFIG="/tmp/mcp.json.tmp"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# List all configured MCP tools
list_tools() {
    log_info "Configured MCP Tools:"
    echo ""

    jq -r '.mcpServers | to_entries[] | "  \(.key)\n    Command: \(.value.command) \(if .value.args then .value.args | join(" ") else "" end)\n    Type: \(.value.type)\n    Category: \(.value.description // "N/A")\n"' "$MCP_CONFIG"

    echo ""
    log_info "Total tools: $(jq '.mcpServers | length' "$MCP_CONFIG")"
}

# Show detailed info about a specific tool
show_tool() {
    local tool_name="$1"

    if ! jq -e ".mcpServers.\"$tool_name\"" "$MCP_CONFIG" >/dev/null 2>&1; then
        log_error "Tool not found: $tool_name"
        exit 1
    fi

    log_info "Tool: $tool_name"
    echo ""
    jq ".mcpServers.\"$tool_name\"" "$MCP_CONFIG"
}

# Add a new MCP tool
add_tool() {
    local tool_name="$1"
    local command="$2"
    local args="$3"
    local description="$4"
    local env_vars="$5"
    local category="${6:-other}"

    if [ -z "$tool_name" ] || [ -z "$command" ]; then
        log_error "Usage: mcp add <name> <command> [args] [description] [env_vars] [category]"
        log_info "Example: mcp add weather 'npx' '-y @modelcontextprotocol/server-weather' 'Weather data'"
        exit 1
    fi

    # Check if tool already exists
    if jq -e ".mcpServers.\"$tool_name\"" "$MCP_CONFIG" >/dev/null 2>&1; then
        log_error "Tool already exists: $tool_name"
        log_info "Use 'mcp update $tool_name' or 'mcp remove $tool_name' first"
        exit 1
    fi

    # Build the tool configuration
    local tool_config=$(cat <<EOF
{
  "command": "$command",
  "type": "stdio",
  "description": "${description:-Custom MCP tool}"
}
EOF
)

    # Add args if provided
    if [ -n "$args" ]; then
        tool_config=$(echo "$tool_config" | jq ".args = $(echo "$args" | jq -R 'split(" ")')")
    fi

    # Add environment variables if provided
    if [ -n "$env_vars" ]; then
        tool_config=$(echo "$tool_config" | jq ".env = $env_vars")
    fi

    # Add tool to configuration
    jq ".mcpServers.\"$tool_name\" = $tool_config" "$MCP_CONFIG" > "$TEMP_CONFIG"
    mv "$TEMP_CONFIG" "$MCP_CONFIG"

    # Add to category
    jq ".toolCategories.\"$category\" += [\"$tool_name\"] | .toolCategories.\"$category\" |= unique" "$MCP_CONFIG" > "$TEMP_CONFIG"
    mv "$TEMP_CONFIG" "$MCP_CONFIG"

    log_success "Added tool: $tool_name"
    log_info "Tool added to category: $category"

    # Test the tool
    log_info "Testing tool availability..."
    if command -v "$command" >/dev/null 2>&1 || [[ "$command" == "npx" ]] || [[ "$command" == "/opt/venv/bin/python3" ]]; then
        log_success "Tool command is available"
    else
        log_warning "Tool command not found: $command"
        log_info "Make sure to install it before using"
    fi
}

# Remove an MCP tool
remove_tool() {
    local tool_name="$1"

    if [ -z "$tool_name" ]; then
        log_error "Usage: mcp remove <name>"
        exit 1
    fi

    if ! jq -e ".mcpServers.\"$tool_name\"" "$MCP_CONFIG" >/dev/null 2>&1; then
        log_error "Tool not found: $tool_name"
        exit 1
    fi

    # Remove from mcpServers
    jq "del(.mcpServers.\"$tool_name\")" "$MCP_CONFIG" > "$TEMP_CONFIG"
    mv "$TEMP_CONFIG" "$MCP_CONFIG"

    # Remove from all categories
    jq ".toolCategories |= map_values(select(. != null) | map(select(. != \"$tool_name\")))" "$MCP_CONFIG" > "$TEMP_CONFIG"
    mv "$TEMP_CONFIG" "$MCP_CONFIG"

    log_success "Removed tool: $tool_name"
}

# Update an existing tool
update_tool() {
    local tool_name="$1"
    shift

    if [ -z "$tool_name" ]; then
        log_error "Usage: mcp update <name> [--command <cmd>] [--args <args>] [--description <desc>]"
        exit 1
    fi

    if ! jq -e ".mcpServers.\"$tool_name\"" "$MCP_CONFIG" >/dev/null 2>&1; then
        log_error "Tool not found: $tool_name"
        exit 1
    fi

    local updated=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --command)
                jq ".mcpServers.\"$tool_name\".command = \"$2\"" "$MCP_CONFIG" > "$TEMP_CONFIG"
                mv "$TEMP_CONFIG" "$MCP_CONFIG"
                updated=true
                shift 2
                ;;
            --args)
                jq ".mcpServers.\"$tool_name\".args = $(echo "$2" | jq -R 'split(" ")')" "$MCP_CONFIG" > "$TEMP_CONFIG"
                mv "$TEMP_CONFIG" "$MCP_CONFIG"
                updated=true
                shift 2
                ;;
            --description)
                jq ".mcpServers.\"$tool_name\".description = \"$2\"" "$MCP_CONFIG" > "$TEMP_CONFIG"
                mv "$TEMP_CONFIG" "$MCP_CONFIG"
                updated=true
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    if [ "$updated" = true ]; then
        log_success "Updated tool: $tool_name"
    else
        log_warning "No changes made"
    fi
}

# Validate MCP configuration
validate_config() {
    log_info "Validating MCP configuration..."

    # Check JSON syntax
    if ! jq empty "$MCP_CONFIG" 2>/dev/null; then
        log_error "Invalid JSON syntax in $MCP_CONFIG"
        exit 1
    fi

    # Check required fields
    local tools=$(jq -r '.mcpServers | keys[]' "$MCP_CONFIG")
    local errors=0

    for tool in $tools; do
        if ! jq -e ".mcpServers.\"$tool\".command" "$MCP_CONFIG" >/dev/null 2>&1; then
            log_error "$tool: Missing 'command' field"
            errors=$((errors + 1))
        fi

        if ! jq -e ".mcpServers.\"$tool\".type" "$MCP_CONFIG" >/dev/null 2>&1; then
            log_error "$tool: Missing 'type' field"
            errors=$((errors + 1))
        fi
    done

    if [ $errors -eq 0 ]; then
        log_success "Configuration is valid"
    else
        log_error "Found $errors validation errors"
        exit 1
    fi
}

# Backup configuration
backup_config() {
    local backup_dir="/home/devuser/.config/claude/backups"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$backup_dir/mcp-${timestamp}.json"

    mkdir -p "$backup_dir"
    cp "$MCP_CONFIG" "$backup_file"

    log_success "Backup created: $backup_file"
}

# Restore configuration from backup
restore_config() {
    local backup_file="$1"

    if [ -z "$backup_file" ]; then
        log_info "Available backups:"
        ls -1t /home/devuser/.config/claude/backups/mcp-*.json 2>/dev/null || log_warning "No backups found"
        echo ""
        log_info "Usage: mcp restore <backup-file>"
        exit 1
    fi

    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi

    # Validate backup before restoring
    if ! jq empty "$backup_file" 2>/dev/null; then
        log_error "Invalid backup file (not valid JSON)"
        exit 1
    fi

    # Create safety backup of current config
    backup_config

    cp "$backup_file" "$MCP_CONFIG"
    log_success "Configuration restored from: $backup_file"
}

# Display usage
usage() {
    cat << EOF
MCP Tool Management CLI

Usage: mcp <command> [arguments]

Commands:
    list                        List all configured MCP tools
    show <name>                 Show detailed info about a tool
    add <name> <cmd> [args]     Add a new MCP tool
    remove <name>               Remove an MCP tool
    update <name> [options]     Update tool configuration
    validate                    Validate MCP configuration
    backup                      Backup current configuration
    restore <file>              Restore configuration from backup

Add Tool Examples:
    # Simple NPM package
    mcp add weather npx "-y @modelcontextprotocol/server-weather" "Weather data"

    # With environment variable
    mcp add github npx "-y @modelcontextprotocol/server-github" "GitHub API" '{"GITHUB_TOKEN":"\${GITHUB_TOKEN}"}'

    # Custom Python script
    mcp add custom-tool /opt/venv/bin/python3 "-u /app/tools/my-tool.py" "Custom tool"

Update Examples:
    mcp update github --description "GitHub API integration"
    mcp update weather --command "node" --args "/path/to/weather-server.js"

Configuration File:
    $MCP_CONFIG

EOF
}

# Main execution
case "${1:-help}" in
    list)
        list_tools
        ;;
    show)
        show_tool "$2"
        ;;
    add)
        add_tool "$2" "$3" "$4" "$5" "$6" "$7"
        ;;
    remove|rm)
        remove_tool "$2"
        ;;
    update)
        shift
        update_tool "$@"
        ;;
    validate)
        validate_config
        ;;
    backup)
        backup_config
        ;;
    restore)
        restore_config "$2"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        usage
        exit 1
        ;;
esac

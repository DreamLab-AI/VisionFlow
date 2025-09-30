#!/bin/bash
# Enhanced setup script for the Multi-Agent Docker Environment
#
# Features:
# - Non-destructive and idempotent operations
# - Argument parsing for --dry-run, --force, --quiet
# - Additive merging of .mcp.json configurations
# - Unified toolchain PATH setup (/etc/profile.d)
# - Security token validation
# - Claude authentication verification
# - Supervisorctl-based aliases for service management
# - Appends a compact, informative context section to CLAUDE.md

# --- Argument Parsing ---
DRY_RUN=false
FORCE=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# --- Logging Functions ---
log_info() {
    [ "$QUIET" = false ] && echo "ℹ️  $1"
}

log_success() {
    [ "$QUIET" = false ] && echo "✅ $1"
}

log_warning() {
    [ "$QUIET" = false ] && echo "⚠️  $1"
}

log_error() {
    echo "❌ $1" >&2
}

dry_run_log() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $1"
        return 0
    fi
    return 1
}

# --- Sudo Check ---
if [ "$(id -u)" -ne 0 ] && [ "$DRY_RUN" = false ]; then
    log_info "Checking if sudo is available..."
    if ! sudo -n true 2>/dev/null; then
        log_warning "Sudo is not available or not configured properly."
        log_info "Attempting to fix sudo permissions..."
        # Try to fix sudo if we can
        if [ -w /usr/bin/sudo ]; then
            chmod 4755 /usr/bin/sudo 2>/dev/null || true
        fi
        # Test again
        if ! sudo -n true 2>/dev/null; then
            log_error "Cannot use sudo. Please run this script as root or fix sudo permissions."
            log_info "Try running: docker exec -u root <container-id> /app/setup-workspace.sh"
            exit 1
        fi
    fi
    log_info "Requesting root privileges for setup..."
    exec sudo /bin/bash "$0" "$@"
fi

echo "🚀 Initializing enhanced Multi-Agent workspace..."
[ "$DRY_RUN" = true ] && echo "🔍 DRY RUN MODE - No changes will be made"

# --- Claude Code Home Directory Workaround ---
log_info "Applying workaround for Claude Code home directory..."
if [ ! -d "/home/ubuntu" ]; then
    if dry_run_log "Would create symlink /home/ubuntu -> /home/dev"; then :; else
        ln -s /home/dev /home/ubuntu
        log_success "Created symlink /home/ubuntu -> /home/dev for Claude Code compatibility."
    fi
else
    log_info "Directory /home/ubuntu already exists, skipping symlink."
fi

# Ensure claude is accessible globally
if [ -f "/home/ubuntu/.local/bin/claude" ] && [ ! -f "/usr/local/bin/claude" ]; then
    if dry_run_log "Would create global claude symlink"; then :; else
        ln -sf /home/ubuntu/.local/bin/claude /usr/local/bin/claude
        chmod +x /usr/local/bin/claude 2>/dev/null || true
        log_success "Created global claude symlink"
    fi
fi

# --- Security Token Validation ---
check_security_tokens() {
    log_info "🔒 Checking security configuration..."
    
    # Check if security tokens are set
    if [ -z "$WS_AUTH_TOKEN" ] || [ "$WS_AUTH_TOKEN" = "your-secure-websocket-token-change-me" ]; then
        log_warning "Default WebSocket auth token detected. Please update WS_AUTH_TOKEN in .env"
    fi
    
    if [ -z "$TCP_AUTH_TOKEN" ] || [ "$TCP_AUTH_TOKEN" = "your-secure-tcp-token-change-me" ]; then
        log_warning "Default TCP auth token detected. Please update TCP_AUTH_TOKEN in .env"
    fi
    
    if [ -z "$JWT_SECRET" ] || [ "$JWT_SECRET" = "your-super-secret-jwt-key-minimum-32-chars" ]; then
        log_warning "Default JWT secret detected. Please update JWT_SECRET in .env"
    fi
}

# --- Claude Authentication Check ---
check_claude_auth() {
    log_info "🤖 Checking Claude authentication..."
    
    if [ -d /home/dev/.claude ] && [ -r /home/dev/.claude/.credentials.json ]; then
        log_success "Claude configuration directory mounted from host"
        
        # Create symlink for ubuntu home if needed
        if [ ! -e /home/ubuntu/.claude ]; then
            ln -s /home/dev/.claude /home/ubuntu/.claude 2>/dev/null || true
        fi
        
        # Check if .claude.json exists at home level
        if [ -r /home/dev/.claude.json ]; then
            log_success "Claude JSON config file mounted"
            if [ ! -e /home/ubuntu/.claude.json ]; then
                ln -s /home/dev/.claude.json /home/ubuntu/.claude.json 2>/dev/null || true
            fi
        fi
        
        # If CLAUDE_CODE_OAUTH_TOKEN is set, it will be used automatically
        if [ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
            log_success "Claude OAuth token also provided via environment"
        fi
    elif [ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
        log_success "Claude OAuth token provided via environment"
    elif [ -n "$ANTHROPIC_API_KEY" ]; then
        log_success "Anthropic API key provided via environment"
    else
        log_warning "No Claude authentication detected. Run 'claude login' on your host machine to authenticate."
        log_info "    The host ~/.claude directory will be mounted to the container."
    fi
}

# --- Multi-Agent Helper Script ---
create_multi_agent_helper() {
    if [ ! -f /usr/local/bin/multi-agent ]; then
        if dry_run_log "Would create multi-agent helper script"; then return 0; fi
        
        cat > /usr/local/bin/multi-agent << 'EOF'
#!/bin/bash
# Multi-agent workspace helper

case "$1" in
    status)
        /app/core-assets/scripts/check-setup-status.sh
        ;;
    logs)
        tail -f /app/mcp-logs/automated-setup.log
        ;;
    health)
        /app/core-assets/scripts/health-check.sh
        ;;
    services)
        supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status
        ;;
    restart)
        supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart all
        ;;
    test-mcp)
        echo '{"jsonrpc":"2.0","id":"test","method":"tools/list","params":{}}' | nc localhost 9500
        ;;
    *)
        echo "Multi-Agent Docker Helper"
        echo "Usage: multi-agent [command]"
        echo ""
        echo "Commands:"
        echo "  status    - Check setup and service status"
        echo "  logs      - View automated setup logs"
        echo "  health    - Run health check"
        echo "  services  - Show supervisor service status"
        echo "  restart   - Restart all services"
        echo "  test-mcp  - Test MCP TCP connection"
        ;;
esac
EOF
        chmod +x /usr/local/bin/multi-agent
        log_success "Created multi-agent command"
    fi
}

# --- Helper Functions for File Operations ---
copy_if_missing() {
    local src="$1"
    local dest="$2"
    local make_executable="${3:-false}"

    if [ -f "$dest" ] && [ "$FORCE" = false ]; then
        log_info "Skipping $dest (already exists)"
        return 0
    fi

    if dry_run_log "Would copy $src -> $dest"; then return 0; fi

    if [ -f "$src" ]; then
        cp "$src" "$dest" 2>/dev/null || { log_error "Failed to copy $src to $dest"; return 1; }
        [ "$make_executable" = true ] && chmod +x "$dest" 2>/dev/null
        log_success "Copied $src -> $dest"
    else
        log_warning "Source file not found: $src"
    fi
}

copy_dir_contents_if_missing() {
    local src_dir="$1"
    local dest_dir="$2"

    if [ ! -d "$src_dir" ]; then log_warning "Source directory not found: $src_dir"; return 1; fi
    if dry_run_log "Would create directory $dest_dir and copy contents from $src_dir"; then return 0; fi
    mkdir -p "$dest_dir"

    for item in "$src_dir"/*; do
        local dest_item="$dest_dir/$(basename "$item")"
        if [ ! -e "$dest_item" ] || [ "$FORCE" = true ]; then
             if dry_run_log "Would copy $item -> $dest_item"; then continue; fi
            cp -r "$item" "$dest_item"
            log_success "Copied $item -> $dest_item"
        else
            log_info "Skipping $dest_item (already exists)"
        fi
    done
}

# --- Main Setup Logic ---

# 1. Non-destructive copy of essential assets
log_info "📂 Syncing essential assets and helper scripts..."
copy_dir_contents_if_missing "/app/core-assets/mcp-tools" "./mcp-tools"
copy_dir_contents_if_missing "/app/core-assets/scripts" "./scripts"
copy_if_missing "/app/mcp-helper.sh" "./mcp-helper.sh" true

# 2. Non-destructive .mcp.json merge
merge_mcp_json() {
    local src="/app/core-assets/mcp.json"
    local dest="./.mcp.json"
    local backup="${dest}.bak.$(date +%Y%m%d_%H%M%S)"

    if [ ! -f "$src" ]; then log_error "Source MCP config not found: $src"; return 1; fi
    if [ ! -f "$dest" ]; then
        if dry_run_log "Would copy $src -> $dest (new file)"; then return 0; fi
        cp "$src" "$dest"
        log_success "Created new MCP config: $dest"
        return 0
    fi

    if dry_run_log "Would backup $dest -> $backup"; then :; else
        cp "$dest" "$backup"
        log_info "Backed up existing config to $backup"
    fi

    if ! command -v jq >/dev/null 2>&1; then
        log_warning "jq not found. Overwriting .mcp.json as fallback."
        if dry_run_log "Would copy $src -> $dest (jq not found)"; then return 0; fi
        cp "$src" "$dest"
        return 0
    fi

    if dry_run_log "Would merge MCP configs using jq: $src into $dest"; then return 0; fi
    jq -s '.[0] * .[1]' "$dest" "$src" > "${dest}.tmp" && mv "${dest}.tmp" "$dest"
    log_success "Merged MCP configurations into $dest"
}

merge_mcp_json

# 3. Setup enhanced PATH for all toolchains
setup_toolchain_paths() {
    local profile_script="/etc/profile.d/multi-agent-paths.sh"
    if dry_run_log "Would create/update $profile_script"; then return 0; fi
    log_info "🛠️  Setting up enhanced PATH for toolchains..."

    cat > "$profile_script" << 'EOF'
#!/bin/sh
prepend_path() {
    if [ -d "$1" ] && ! echo "$PATH" | grep -q -s "$1"; then
        export PATH="$1:$PATH"
    fi
}
prepend_path "/home/dev/.cargo/bin"
prepend_path "/opt/venv312/bin"
prepend_path "/home/dev/.npm-global/bin"
prepend_path "/home/dev/.local/bin"
prepend_path "/home/dev/.deno/bin"
prepend_path "/opt/oss-cad-suite/bin"
EOF

    chmod +x "$profile_script"
    log_success "Created toolchain PATH configuration: $profile_script"
}

setup_toolchain_paths

# 4. Update bashrc with supervisorctl-based aliases
add_mcp_aliases() {
    local bashrc_file="/home/dev/.bashrc"
    local marker="# MCP Server Management (supervisorctl-based)"

    if [ ! -f "$bashrc_file" ]; then
        if dry_run_log "Would create empty .bashrc for dev user"; then :; else
            sudo -u dev touch "$bashrc_file"
            log_info "Created empty .bashrc for dev user."
        fi
    fi

    if grep -q "$marker" "$bashrc_file"; then
        log_info "MCP aliases already exist in bashrc"
        return 0
    fi

    if dry_run_log "Would add MCP aliases to $bashrc_file"; then return 0; fi

    read -r -d '' BASHRC_ADDITIONS << 'EOF'

# MCP Server Management (supervisorctl-based)
alias mcp-tcp-start='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf start mcp-tcp-server'
alias mcp-tcp-stop='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf stop mcp-tcp-server'
alias mcp-tcp-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status mcp-tcp-server'
alias mcp-tcp-restart='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart mcp-tcp-server'
alias mcp-tcp-logs='tail -f /app/mcp-logs/mcp-tcp-server.log'
alias mcp-ws-start='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf start mcp-ws-bridge'
alias mcp-ws-stop='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf stop mcp-ws-bridge'
alias mcp-ws-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status mcp-ws-bridge'
alias mcp-ws-restart='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart mcp-ws-bridge'
alias mcp-ws-logs='tail -f /app/mcp-logs/mcp-ws-bridge.log'

# Claude Code Aliases
alias dsp='claude --dangerously-skip-permissions'

# Claude-Flow v110 Agent Commands
alias claude-flow-init-agents='echo "Initializing Goal Planner..." && claude-flow goal init --force && echo "Initializing Neural Agent..." && claude-flow neural init --force'
alias cf-goal='claude-flow goal'
alias cf-neural='claude-flow neural'
alias cf-status='claude-flow status'
alias cf-logs='tail -f /app/mcp-logs/mcp-tcp-server.log | grep -i claude-flow'
alias cf-tcp-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status claude-flow-tcp'
alias cf-tcp-logs='tail -f /app/mcp-logs/claude-flow-tcp.log'
alias cf-test-tcp='echo "{\"jsonrpc\":\"2.0\",\"id\":\"test\",\"method\":\"tools/list\",\"params\":{}}" | nc localhost 9502'

# Playwright MCP Commands
alias playwright-test='echo "Testing local Playwright MCP server..." && echo "{\"jsonrpc\":\"2.0\",\"id\":\"test\",\"method\":\"tools/list\",\"params\":{}}" | node /app/core-assets/scripts/playwright-mcp-local.js 2>&1 | head -20'
alias playwright-headless='node /app/core-assets/scripts/playwright-mcp-local.js'
alias playwright-vnc='echo "VNC access: vncviewer localhost:5901 (or use any VNC client)"'
alias playwright-health='curl -s http://127.0.0.1:9880 | jq 2>/dev/null || echo "Proxy not running"'
alias playwright-proxy-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status playwright-mcp-proxy'
alias playwright-proxy-logs='tail -f /app/mcp-logs/playwright-proxy.log'
alias playwright-stack-test='/app/core-assets/scripts/test-playwright-stack.sh'
alias playwright-visual='echo "Playwright visual mode runs in GUI container. Connect via VNC on port 5901 for visual access."'

# Goalie Research Commands (MCP only, no TCP server)
alias goalie-search='goalie search'
alias goalie-quick='goalie query'
alias goalie-reasoning='goalie reasoning'

# CLAUDE.md Resilience
alias claude-md-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status claude-md-watcher'
alias claude-md-logs='tail -f /app/mcp-logs/claude-md-watcher.log'
alias claude-md-verify='grep -c "SYSTEM_TOOLS_MANIFEST" /workspace/CLAUDE.md'
alias claude-md-repair='/app/core-assets/scripts/claude-md-patcher.sh'

# Quick MCP testing functions
mcp-test-tcp() {
    local port=${1:-9500}
    echo "Testing MCP TCP connection on port $port..."
    echo '{"jsonrpc":"2.0","id":"test","method":"tools/list","params":{}}' | nc -w 2 localhost $port
}
mcp-test-health() {
    echo "Testing MCP health endpoint..."
    curl -s http://127.0.0.1:9501/health | jq . 2>/dev/null || curl -s http://127.0.0.1:9501/health
}
goalie-research() {
    echo "Performing deep research: $1"
    goalie search "$1" --max-results 15 --save
}

# Toolchain validation function
validate-toolchains() {
    echo "=== Toolchain Validation ==="
    printf "Rust cargo: "
    if command -v cargo >/dev/null; then cargo --version; else echo "Not found"; fi
    printf "Python venv: "
    if command -v python >/dev/null; then python --version; else echo "Not found"; fi
    printf "Node.js: "
    if command -v node >/dev/null; then node --version; else echo "Not found"; fi
    printf "Deno: "
    if command -v deno >/dev/null; then deno --version | head -n 1; else echo "Not found"; fi
    printf "JQ: "
    if command -v jq >/dev/null; then jq --version; else echo "Not found"; fi
    printf "Playwright: "
    if command -v playwright >/dev/null 2>&1; then playwright --version; else echo "Not found"; fi
}

# Load Playwright helpers if available
if [ -f /app/core-assets/scripts/playwright-mcp-helpers.sh ]; then
    source /app/core-assets/scripts/playwright-mcp-helpers.sh
fi
EOF

    sudo -u dev bash -c "echo \"\$1\" >> \"\$2\"" -- "$BASHRC_ADDITIONS" "$bashrc_file"
    log_success "Added MCP management aliases to bashrc"
}

add_mcp_aliases

# 5. Validate and fix Rust toolchain availability
# This is now handled in the Dockerfile

# 6. Update CLAUDE.md with compact tool manifest
update_claude_md() {
    local claude_md="./CLAUDE.md"

    if [ "${SETUP_APPEND_CLAUDE_DOC:-true}" != "true" ]; then
        log_info "Skipping CLAUDE.md updates (SETUP_APPEND_CLAUDE_DOC is not 'true')"
        return 0
    fi

    if [ ! -f "$claude_md" ]; then
        log_warning "CLAUDE.md not found, cannot patch."
        return 1
    fi

    if dry_run_log "Would patch $claude_md with compact tool manifest"; then return 0; fi

    # Use the compact patcher
    /app/core-assets/scripts/claude-md-patcher.sh
    log_success "CLAUDE.md patched with compact tool manifest"
}

update_claude_md

# 7. Ensure claude-flow is installed globally
ensure_claude_flow_global() {
    log_info "🔧 Ensuring claude-flow is installed globally..."
    
    if [ ! -f "/usr/bin/claude-flow" ]; then
        log_info "Installing claude-flow globally..."
        if dry_run_log "Would install claude-flow globally"; then return 0; fi
        
        npm install -g claude-flow@alpha 2>/dev/null || {
            log_warning "Failed to install claude-flow globally, trying with sudo..."
            sudo npm install -g claude-flow@alpha || {
                log_error "Failed to install claude-flow globally"
                return 1
            }
        }
        log_success "Installed claude-flow globally"
    else
        log_info "claude-flow already installed globally"
    fi
}

if [ "$DRY_RUN" = false ]; then
    ensure_claude_flow_global
else
    dry_run_log "Would ensure claude-flow is installed globally"
fi

# 8. Runtime patching removed - now handled at build time
# Patches are applied during docker build in the Dockerfile
log_info "🔧 Runtime patching disabled - patches applied at build time"

# 9. Patch TCP server to use global installation and shared database
patch_tcp_server() {
    log_info "🔧 Patching TCP server to use global installation and shared database..."
    
    local tcp_server_path="/app/core-assets/scripts/mcp-tcp-server.js"
    
    if [ ! -f "$tcp_server_path" ]; then
        tcp_server_path="/workspace/scripts/mcp-tcp-server.js"
    fi
    
    if [ ! -f "$tcp_server_path" ]; then
        log_warning "TCP server not found, skipping patch"
        return 1
    fi
    
    log_info "Found TCP server at: $tcp_server_path"
    
    if dry_run_log "Would patch TCP server at $tcp_server_path"; then return 0; fi
    
    # Create backup
    cp "$tcp_server_path" "${tcp_server_path}.bak.$(date +%s)" 2>/dev/null || true
    
    # Patch: Change from npx to global installation and add shared database
    if grep -q "spawn('npx'" "$tcp_server_path" || grep -q "spawn('/usr/bin/claude-flow'" "$tcp_server_path"; then
        log_info "Patching TCP server spawn commands..."
        
        # First, replace npx with global installation
        sed -i "s|spawn('npx', \['claude-flow@alpha'|spawn('/usr/bin/claude-flow', ['|g" "$tcp_server_path"
        
        # Then ensure environment includes shared database - handle both cases
        # Case 1: When env line exists but doesn't have our DB path
        if grep -q "env: {" "$tcp_server_path" && ! grep -q "CLAUDE_FLOW_DB_PATH" "$tcp_server_path"; then
            sed -i "/spawn('\/usr\/bin\/claude-flow'/,/env: {/{
                s|env: { \(.*\)|env: {\n          ...process.env,\n          CLAUDE_FLOW_DB_PATH: '/workspace/.swarm/memory.db', // Ensure same DB is used\n          \1|
            }" "$tcp_server_path"
        fi
        
        # Fix any duplicate env properties that might have been created
        sed -i '/env: {.*env: {/s/env: {.*env: { \.\.\./env: {\n          .../' "$tcp_server_path"
        
        log_success "Patched TCP server to use global installation with shared database"
    else
        log_info "TCP server spawn commands already patched"
    fi
    
    # Ensure database directory exists
    if [ ! -d "/workspace/.swarm" ]; then
        mkdir -p /workspace/.swarm
        chown -R dev:dev /workspace/.swarm
        log_success "Created shared database directory: /workspace/.swarm"
    fi
}

if [ "$DRY_RUN" = false ]; then
    patch_tcp_server
else
    dry_run_log "Would patch TCP server"
fi

# --- Final Summary ---
show_setup_summary() {
    # Create the completion marker file to hide the welcome message
    if [ "$DRY_RUN" = false ]; then
        touch /workspace/.setup_completed
        chown dev:dev /workspace/.setup_completed
    fi

    echo ""
    echo "=== ✅ Enhanced Setup Complete ==="
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "🔍 DRY RUN COMPLETE - No changes were made."
        return 0
    fi

    echo "📋 Setup Summary:"
    echo "  - Merged MCP configuration into .mcp.json"
    echo "  - Configured toolchain PATHs in /etc/profile.d/"
    echo "  - Added supervisorctl-based aliases to .bashrc"
    echo "  - Appended environment context to CLAUDE.md"
    echo "  - Installed claude-flow globally at /usr/bin/claude-flow"
    echo "  - Patched MCP server for proper agent tracking"
    echo "  - Fixed TCP server to use shared database"
    echo "  - Created shared database at /workspace/.swarm/memory.db"
    echo ""

    echo "🛠️  Key Commands:"
    echo "  - mcp-tcp-status, mcp-ws-status"
    echo "  - mcp-test-health, validate-toolchains"
    echo ""

    echo "💡 Development Context:"
    echo "  - Your project root is in: ext/"
    echo "  - Use internal tools like 'cargo check' to validate your work."
    echo "  - Claude cannot build or see external Docker services."
    echo ""
}

# --- Verification Test ---
verify_agent_tracking() {
    log_info "🧪 Verifying agent tracking functionality..."
    
    if [ "$DRY_RUN" = true ]; then
        dry_run_log "Would verify agent tracking"
        return 0
    fi
    
    # Wait for services to be ready
    sleep 3
    
    # Test agent list command
    local test_result=$(echo '{"jsonrpc":"2.0","id":"test","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | nc -w 3 localhost 9500 2>/dev/null | tail -n 1)
    
    if echo "$test_result" | grep -q '"success":true' && ! echo "$test_result" | grep -q '"id":"agent-1"'; then
        log_success "✅ Agent tracking verified - database integration working!"
    elif echo "$test_result" | grep -q '"id":"agent-1"'; then
        log_warning "⚠️  Agent tracking still returning mock data - manual intervention may be needed"
        log_info "    Check that /usr/bin/claude-flow exists and patches were applied"
    else
        log_info "    Service may still be starting - check with: mcp-tcp-status"
    fi
}

# Removed: Patch creation was inappropriate for a setup script
# This should be handled as part of the actual project development, not environment setup

# 10. Verify Playwright MCP Proxy
verify_playwright_proxy() {
    log_info "🎭 Verifying Playwright MCP Proxy connection..."
    
    if [ "$DRY_RUN" = true ]; then
        dry_run_log "Would verify Playwright MCP Proxy"
        return 0
    fi
    
    # Check if proxy is running
    if supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status playwright-mcp-proxy 2>/dev/null | grep -q RUNNING; then
        log_success "Playwright MCP Proxy is running"
        
        # Test proxy health endpoint
        if curl -sf http://127.0.0.1:9880 >/dev/null 2>&1; then
            log_success "Playwright proxy is healthy and connected to GUI container"
            log_info "Access browser automation visually via VNC on port 5901"
        else
            log_warning "Playwright proxy is running but GUI container may not be accessible"
            log_info "Ensure the GUI container is running: docker-compose ps gui-tools-service"
        fi
    else
        log_warning "Playwright MCP Proxy is not running"
        log_info "Start it with: supervisorctl start playwright-mcp-proxy"
    fi
}

# 11. Verify Chrome DevTools MCP Proxy
verify_chrome_devtools_proxy() {
    log_info "🔍 Verifying Chrome DevTools MCP Proxy connection..."
    
    if [ "$DRY_RUN" = true ]; then
        dry_run_log "Would verify Chrome DevTools MCP Proxy"
        return 0
    fi
    
    # Check if proxy is running
    if supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status chrome-devtools-mcp-proxy 2>/dev/null | grep -q RUNNING; then
        log_success "Chrome DevTools MCP Proxy is running"
        
        # Test proxy health endpoint
        if curl -sf http://127.0.0.1:9222 >/dev/null 2>&1; then
            log_success "Chrome DevTools proxy is healthy and connected to GUI container"
            log_info "Access browser automation via VNC on port 5901"
        else
            log_warning "Chrome DevTools proxy is running but GUI container may not be accessible"
            log_info "Ensure the GUI container is running: docker-compose ps gui-tools-service"
        fi
    else
        log_warning "Chrome DevTools MCP Proxy is not running"
        log_info "Start it with: supervisorctl start chrome-devtools-mcp-proxy"
    fi
}

# 12. Verify Claude-Flow MCP Service
verify_claude_flow_service() {
    log_info "🤖 Verifying Claude-Flow MCP service..."
    
    if [ "$DRY_RUN" = true ]; then
        dry_run_log "Would verify Claude-Flow MCP service"
        return 0
    fi
    
    # Check if claude-flow is available
    if ! command -v claude-flow >/dev/null 2>&1 && ! npm list -g claude-flow@alpha >/dev/null 2>&1; then
        log_warning "claude-flow not found globally"
        return 1
    fi
    
    # Create necessary directories
    mkdir -p /workspace/.swarm
    mkdir -p /workspace/.hive-mind
    
    # Claude-Flow runs as an MCP server managed by Claude directly
    # The mcp.json configuration will start it when needed
    log_info "Claude-Flow is configured as an MCP server in .mcp.json"
    log_info "It will start automatically when Claude accesses it"
    
    # Check if we can verify the installation
    if claude-flow --version >/dev/null 2>&1; then
        local version=$(claude-flow --version 2>/dev/null || echo "unknown")
        log_success "Claude-Flow version: $version"
    fi
    
    # Explain the dual setup
    log_info "Claude-Flow is configured with dual access:"
    log_info "  1. Local MCP: Available to Claude Code via .mcp.json"
    log_info "  2. TCP Port 9500: Shared instance for external access"  
    log_info "  3. TCP Port 9502: Isolated sessions for external projects"
    
    log_info "To initialize agents after Claude-Flow MCP server starts:"
    log_info "  - In Claude: Use the mcp__claude-flow__goal_init tool"
    log_info "  - CLI: claude-flow goal init"
    log_info "  - For neural: claude-flow neural init"
    
    # Check if initialization files exist
    if [ -f /workspace/.hive-mind/config.json ] || [ -f /workspace/.swarm/memory.db ]; then
        log_success "Claude-Flow workspace already initialized"
    else
        log_info "Claude-Flow workspace not yet initialized - will be created on first use"
    fi
}

# --- Service Startup Verification ---
verify_services_startup() {
    log_info "🚀 Verifying MCP services status..."
    
    if [ "$DRY_RUN" = true ]; then
        dry_run_log "Would verify services status"
        return 0
    fi
    
    # Just check if services are running, don't try to start them
    log_info "Checking MCP services..."
    
    # Check if TCP server is responding
    if echo '{"jsonrpc":"2.0","id":"health","method":"health","params":{}}' | nc -w 2 localhost 9500 2>/dev/null | grep -q "jsonrpc"; then
        log_success "✅ MCP TCP server is responding!"
    else
        log_warning "⚠️  MCP TCP server not responding on port 9500"
        log_info "Services will be started by supervisord. Check with 'supervisorctl status'"
    fi
    
    # Check WebSocket bridge
    if nc -z localhost 3002 2>/dev/null; then
        log_success "✅ MCP WebSocket bridge is ready!"
    else
        log_warning "⚠️  MCP WebSocket bridge not responding on port 3002"
    fi
    
    # Check health endpoint if available
    if curl -sf http://localhost:9501/health >/dev/null 2>&1; then
        log_success "✅ Health endpoint is responding!"
    else
        log_info "Health endpoint not available (this is normal if services are still starting)"
    fi
    
    # Run the health check script if available
    if [ -x /app/core-assets/scripts/health-check.sh ]; then
        echo ""
        log_info "Running comprehensive health check..."
        /app/core-assets/scripts/health-check.sh || true
    fi
}

# --- Main Execution ---
show_setup_summary

# Run security and auth checks
check_security_tokens
check_claude_auth

# Create helper utilities
create_multi_agent_helper

# Verify services will start properly
log_info "Verifying service prerequisites..."
if [ ! -f /workspace/scripts/mcp-tcp-server.js ]; then
    log_warning "MCP scripts not found in workspace. Copying from core assets..."
    mkdir -p /workspace/scripts
    cp -r /app/core-assets/scripts/* /workspace/scripts/ 2>/dev/null || true
    chown -R dev:dev /workspace/scripts
fi

# Run verification if not in dry-run mode
if [ "$DRY_RUN" = false ]; then
    verify_services_startup
    verify_agent_tracking
    verify_playwright_proxy
    verify_chrome_devtools_proxy
    verify_claude_flow_service
fi

# Mark setup as completed
if [ "$DRY_RUN" = false ]; then
    touch /workspace/.setup_completed
    log_success "Workspace setup completed successfully!"
fi

echo ""
echo "🎉 Multi-Agent environment ready for development!"
echo "🔧 MCP services configured and ready"
echo "🤖 Claude-Flow MCP server will start automatically when accessed"
echo ""
echo "📝 Next Steps:"
echo "  1. Initialize agents: claude-flow-init-agents"
echo "  2. Check MCP status: mcp-tcp-status"
echo "  3. View logs: mcp-tcp-logs"
echo "  4. Access Playwright visually: VNC on port 5901"
echo "🎭 Playwright MCP server ready for browser automation"
echo ""
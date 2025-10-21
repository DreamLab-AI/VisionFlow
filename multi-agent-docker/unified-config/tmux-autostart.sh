#!/bin/bash
# tmux Workspace Auto-Start for Unified DevPod
# Creates 8 windows for comprehensive development workflow

set -e

echo "=== Starting TMux Unified Workspace ==="

# Environment variables
WORKSPACE="${WORKSPACE:-/home/devuser/workspace}"
AGENTS_DIR="${AGENTS_DIR:-/home/devuser/agents}"

# Kill existing session if it exists
tmux kill-session -t workspace 2>/dev/null || true

# Create new session with first window
tmux new-session -d -s workspace -n "Claude-Main" -c "$WORKSPACE"

# Set large scrollback buffer
tmux set-option -g history-limit 50000

# Set status bar
tmux set-option -g status-style "bg=blue,fg=white"
tmux set-option -g status-left "#[bg=green,fg=black] TURBO-FLOW "
tmux set-option -g status-right "#[bg=yellow,fg=black] %Y-%m-%d %H:%M "

# ============================================================================
# Window 0: Claude-Main - Primary Claude Code workspace
# ============================================================================

tmux send-keys -t workspace:0 "clear" C-m
tmux send-keys -t workspace:0 "echo '=== Claude Code Main Workspace ==='" C-m
tmux send-keys -t workspace:0 "echo 'Workspace: $WORKSPACE'" C-m
tmux send-keys -t workspace:0 "echo 'Agents: $AGENTS_DIR'" C-m
tmux send-keys -t workspace:0 "echo ''" C-m
tmux send-keys -t workspace:0 "echo 'Available commands:'" C-m
tmux send-keys -t workspace:0 "echo '  claude             - Start Claude Code'" C-m
tmux send-keys -t workspace:0 "echo '  dsp                - Claude with --dangerously-skip-permissions'" C-m
tmux send-keys -t workspace:0 "echo '  claude-monitor     - Monitor Claude API usage'" C-m
tmux send-keys -t workspace:0 "echo '  as-gemini          - Switch to gemini-user'" C-m
tmux send-keys -t workspace:0 "echo '  as-openai          - Switch to openai-user'" C-m
tmux send-keys -t workspace:0 "echo '  as-zai             - Switch to zai-user'" C-m
tmux send-keys -t workspace:0 "echo ''" C-m
tmux send-keys -t workspace:0 "cd $WORKSPACE" C-m

# ============================================================================
# Window 1: Claude-Agent - Agent execution and testing
# ============================================================================

tmux new-window -t workspace:1 -n "Claude-Agent" -c "$WORKSPACE"
tmux send-keys -t workspace:1 "clear" C-m
tmux send-keys -t workspace:1 "echo '=== Claude Agent Workspace ==='" C-m
tmux send-keys -t workspace:1 "echo 'Use this window for agent execution and testing'" C-m
tmux send-keys -t workspace:1 "echo ''" C-m
tmux send-keys -t workspace:1 "echo 'Mandatory agents:'" C-m
tmux send-keys -t workspace:1 "echo '  cat \$AGENTS_DIR/doc-planner.md'" C-m
tmux send-keys -t workspace:1 "echo '  cat \$AGENTS_DIR/microtask-breakdown.md'" C-m
tmux send-keys -t workspace:1 "echo ''" C-m
tmux send-keys -t workspace:1 "echo 'Total agents: \$(ls -1 \$AGENTS_DIR/*.md | wc -l)'" C-m

# ============================================================================
# Window 2: Services - Supervisord status monitoring
# ============================================================================

tmux new-window -t workspace:2 -n "Services" -c "$WORKSPACE"
tmux send-keys -t workspace:2 "clear" C-m
tmux send-keys -t workspace:2 "echo '=== Service Status Monitor ==='" C-m
tmux send-keys -t workspace:2 "echo ''" C-m
tmux send-keys -t workspace:2 "sudo supervisorctl status" C-m
tmux send-keys -t workspace:2 "echo ''" C-m
tmux send-keys -t workspace:2 "echo 'Commands:'" C-m
tmux send-keys -t workspace:2 "echo '  sudo supervisorctl status       - View all services'" C-m
tmux send-keys -t workspace:2 "echo '  sudo supervisorctl restart <service>'" C-m
tmux send-keys -t workspace:2 "echo '  sudo supervisorctl stop <service>'" C-m
tmux send-keys -t workspace:2 "echo '  sudo supervisorctl start <service>'" C-m

# ============================================================================
# Window 3: Development - Python/Rust/CUDA development environment
# ============================================================================

tmux new-window -t workspace:3 -n "Development" -c "$WORKSPACE"
tmux send-keys -t workspace:3 "clear" C-m
tmux send-keys -t workspace:3 "echo '=== Development Environment ==='" C-m
tmux send-keys -t workspace:3 "echo ''" C-m
tmux send-keys -t workspace:3 "echo 'Python:  python --version'" C-m
tmux send-keys -t workspace:3 "python --version" C-m
tmux send-keys -t workspace:3 "echo 'Rust:    rustc --version'" C-m
tmux send-keys -t workspace:3 "rustc --version" C-m
tmux send-keys -t workspace:3 "echo 'CUDA:    nvcc --version'" C-m
tmux send-keys -t workspace:3 "nvcc --version 2>/dev/null || echo '  (CUDA available if NVIDIA GPU detected)'" C-m
tmux send-keys -t workspace:3 "echo ''" C-m
tmux send-keys -t workspace:3 "echo 'User switching:'" C-m
tmux send-keys -t workspace:3 "echo '  as-gemini  - Google Gemini tools'" C-m
tmux send-keys -t workspace:3 "echo '  as-openai  - OpenAI Codex'" C-m
tmux send-keys -t workspace:3 "echo '  as-zai     - Z.AI service'" C-m

# ============================================================================
# Window 4: Logs - Tail management API and service logs
# ============================================================================

tmux new-window -t workspace:4 -n "Logs" -c "$WORKSPACE"
tmux send-keys -t workspace:4 "clear" C-m
tmux send-keys -t workspace:4 "echo '=== Service Logs ==='" C-m
tmux send-keys -t workspace:4 "echo 'Tailing management API logs...'" C-m
tmux send-keys -t workspace:4 "echo ''" C-m
tmux send-keys -t workspace:4 "tail -f /var/log/management-api.log" C-m

# Split for multiple log viewers
tmux split-window -v -t workspace:4
tmux send-keys -t workspace:4.1 "tail -f /var/log/supervisord.log" C-m

# ============================================================================
# Window 5: System - htop and resource monitoring
# ============================================================================

tmux new-window -t workspace:5 -n "System" -c "$WORKSPACE"
tmux send-keys -t workspace:5 "htop" C-m

# ============================================================================
# Window 6: VNC-Status - VNC server status and connections
# ============================================================================

tmux new-window -t workspace:6 -n "VNC-Status" -c "$WORKSPACE"
tmux send-keys -t workspace:6 "clear" C-m
tmux send-keys -t workspace:6 "echo '=== VNC Server Status ==='" C-m
tmux send-keys -t workspace:6 "echo ''" C-m
tmux send-keys -t workspace:6 "echo 'VNC Display: :1'" C-m
tmux send-keys -t workspace:6 "echo 'VNC Port: 5901'" C-m
tmux send-keys -t workspace:6 "echo 'VNC Password: turboflow'" C-m
tmux send-keys -t workspace:6 "echo ''" C-m
tmux send-keys -t workspace:6 "echo 'VNC Processes:'" C-m
tmux send-keys -t workspace:6 "ps aux | grep -i vnc | grep -v grep" C-m
tmux send-keys -t workspace:6 "echo ''" C-m
tmux send-keys -t workspace:6 "echo 'Active X11 Displays:'" C-m
tmux send-keys -t workspace:6 "ls -la /tmp/.X11-unix/" C-m
tmux send-keys -t workspace:6 "echo ''" C-m
tmux send-keys -t workspace:6 "echo 'To restart VNC: sudo supervisorctl restart xvnc'" C-m

# ============================================================================
# Window 7: SSH-Shell - General purpose shell
# ============================================================================

tmux new-window -t workspace:7 -n "SSH-Shell" -c "$WORKSPACE"
tmux send-keys -t workspace:7 "clear" C-m
tmux send-keys -t workspace:7 "echo '=== General Purpose Shell ==='" C-m
tmux send-keys -t workspace:7 "echo ''" C-m
tmux send-keys -t workspace:7 "echo 'Container Information:'" C-m
tmux send-keys -t workspace:7 "echo '  Hostname: \$(hostname)'" C-m
tmux send-keys -t workspace:7 "echo '  SSH Port: 22'" C-m
tmux send-keys -t workspace:7 "echo '  Management API: http://localhost:9090'" C-m
tmux send-keys -t workspace:7 "echo '  VNC: vnc://localhost:5901'" C-m
tmux send-keys -t workspace:7 "echo ''" C-m
tmux send-keys -t workspace:7 "echo 'Users:'" C-m
tmux send-keys -t workspace:7 "echo '  devuser (1000:1000) - Primary development'" C-m
tmux send-keys -t workspace:7 "echo '  gemini-user (1001:1001) - Google Gemini tools'" C-m
tmux send-keys -t workspace:7 "echo '  openai-user (1002:1002) - OpenAI Codex'" C-m
tmux send-keys -t workspace:7 "echo '  zai-user (1003:1003) - Z.AI service'" C-m

# ============================================================================
# Select the first window (Claude-Main)
# ============================================================================

tmux select-window -t workspace:0

echo "✅ TMux workspace 'workspace' created successfully with 8 windows!"
echo "📝 Windows:"
echo "   0: Claude-Main   - Primary Claude Code workspace"
echo "   1: Claude-Agent  - Agent execution and testing"
echo "   2: Services      - Supervisord monitoring"
echo "   3: Development   - Python/Rust/CUDA development"
echo "   4: Logs          - Service logs (split view)"
echo "   5: System        - htop resource monitoring"
echo "   6: VNC-Status    - VNC server information"
echo "   7: SSH-Shell     - General purpose shell"
echo ""
echo "To attach: tmux attach-session -t workspace"
echo "To navigate: Ctrl+B then window number (0-7)"

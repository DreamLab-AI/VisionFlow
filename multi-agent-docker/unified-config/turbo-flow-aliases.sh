#!/bin/bash
# ============================================================================
# TurboFlow V4 Aliases — Comprehensive Shell UX
# ============================================================================
# VERSION: 4.0.0  |  UPDATED: 2026-03-12
#
# Alias families:
#   rf-*      Ruflo orchestration (replaces cf-*)
#   bd-*      Beads cross-session memory
#   wt-*      Git worktree agent isolation
#   gnx-*     GitNexus codebase knowledge graph
#   aqe-*     Agentic QE quality engineering
#   mem-*     Memory (ruflo native)
#   hooks-*   Intelligence hooks
#   neural-*  Neural patterns
#   ruv-*     RuVector/AgentDB
#   os-*      OpenSpec
#
# Backwards compat: cf-* aliases still work (mapped to rf-*)
# ============================================================================

# --- Agent Teams ---
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

# --- Claude Code ---
alias dsp='claude --dangerously-skip-permissions'

# === RUFLO (Core Orchestration) ===
alias rf='ruflo'
alias rf-init='ruflo init'
alias rf-wizard='ruflo init --wizard'
alias rf-doctor='ruflo doctor --fix'
alias rf-swarm='ruflo swarm init --topology hierarchical --max-agents 8 --strategy specialized'
alias rf-mesh='ruflo swarm init --topology mesh'
alias rf-ring='ruflo swarm init --topology ring'
alias rf-star='ruflo swarm init --topology star'
alias rf-hybrid='ruflo swarm init --topology hierarchical-mesh --max-agents 15 --strategy specialized'
alias rf-daemon='ruflo daemon start'
alias rf-status='ruflo status'
alias rf-migrate='ruflo migrate run --backup'
alias rf-plugins='ruflo plugins list'
alias rf-version='ruflo --version'

rf-spawn() { ruflo agent spawn -t "${1:-coder}" --name "${2:-agent-$RANDOM}"; }
rf-task() { ruflo swarm "$1" --parallel; }

# Backwards compat: cf-* -> rf-*
alias cf='ruflo'
alias cf-init='ruflo init'
alias cf-swarm='rf-swarm'
alias cf-mesh='rf-mesh'
alias cf-doctor='rf-doctor'
alias cf-daemon='rf-daemon'
alias cf-status='rf-status'
alias cf-plugins='rf-plugins'
alias claude-flow='ruflo'

# === BEADS (Cross-Session Project Memory) ===
alias bd-ready='bd ready'
alias bd-add='bd add'
alias bd-list='bd list'
alias bd-status='bd status'
bd-issue() { bd add --type issue "$@"; }
bd-decision() { bd add --type decision "$@"; }
bd-blocker() { bd add --type blocker "$@"; }

# === GIT WORKTREES (Agent Isolation) ===
wt-add() {
    local name="${1:?Usage: wt-add <agent-name>}"
    local branch_name="$name/$(date +%s)"
    git worktree add ".worktrees/$name" -b "$branch_name"
    echo "Worktree created: .worktrees/$name (branch: $branch_name)"
    export DATABASE_SCHEMA="wt_${name}_$(date +%s)"
    echo "Database schema: $DATABASE_SCHEMA"
    # Auto-index with GitNexus if available
    if command -v gitnexus &>/dev/null; then
        (cd ".worktrees/$name" && gitnexus analyze 2>/dev/null &)
        echo "GitNexus indexing in background..."
    fi
}
wt-remove() {
    local name="${1:?Usage: wt-remove <agent-name>}"
    git worktree remove ".worktrees/$name" --force 2>/dev/null
    echo "Worktree removed: $name"
}
wt-list() { git worktree list; }
wt-clean() { git worktree prune && echo "Stale worktrees pruned"; }

# === GITNEXUS (Codebase Knowledge Graph) ===
alias gnx='gitnexus'
alias gnx-analyze='gitnexus analyze'
alias gnx-analyze-force='gitnexus analyze --force'
alias gnx-mcp='gitnexus mcp'
alias gnx-serve='gitnexus serve'
alias gnx-status='gitnexus status'
alias gnx-wiki='gitnexus wiki'
alias gnx-list='gitnexus list'
alias gnx-clean='gitnexus clean'

# === AGENTIC QE (Quality Engineering) ===
alias aqe='ruflo plugins run agentic-qe'
alias aqe-generate='ruflo plugins run agentic-qe generate'
alias aqe-gate='ruflo plugins run agentic-qe gate'
alias aqe-report='ruflo plugins run agentic-qe report'

# === OPENSPEC (Spec-Driven Development) ===
alias os-spec='npx @fission-ai/openspec'
alias os-init='npx @fission-ai/openspec init'

# === RUVECTOR / AGENTDB ===
alias ruv='ruflo agentdb'
alias ruv-stats='ruflo agentdb stats'
alias ruv-init='ruflo agentdb init'
ruv-remember() { ruflo agentdb store --key "$1" --value "$2"; }
ruv-recall() { ruflo agentdb query "$1"; }

# === MEMORY (Ruflo Native) ===
alias mem-search='ruflo memory search'
alias mem-store='ruflo memory store'
alias mem-stats='ruflo memory stats'
alias mem-list='ruflo memory list'

# === HOOKS INTELLIGENCE ===
alias hooks-pre='ruflo hooks pre-edit'
alias hooks-post='ruflo hooks post-edit'
alias hooks-train='ruflo hooks pretrain --depth deep'
alias hooks-route='ruflo hooks route'
alias hooks-metrics='ruflo hooks metrics'

# === NEURAL ===
alias neural-train='ruflo neural train'
alias neural-status='ruflo neural status'
alias neural-patterns='ruflo neural patterns'

# === GEMINI CLI (@google/gemini-cli) ===
# Run Gemini CLI as gemini-user (API key loaded from ~/.gemini/.env)
gemini-ask() { sudo -u gemini-user -i bash -c "gemini -p \"$*\""; }
alias gemini-cli='sudo -u gemini-user -i bash -c "gemini"'
alias gemini-version='sudo -u gemini-user -i bash -c "gemini --version"'

# === GEMINI FLOW ===
alias gf-init='sudo -u gemini-user -i bash -c "cd /home/gemini-user/workspace && npx gemini-flow init --protocols a2a,mcp --topology hierarchical"'
alias gf-swarm='sudo -u gemini-user -i bash -c "cd /home/gemini-user/workspace && npx gemini-flow swarm --agents 66 --intelligent"'
alias gf-architect='sudo -u gemini-user -i bash -c "cd /home/gemini-user/workspace && npx gemini-flow spawn --role architect --count 5"'
alias gf-coder='sudo -u gemini-user -i bash -c "cd /home/gemini-user/workspace && npx gemini-flow spawn --role coder --count 12"'
alias gf-status='sudo -u gemini-user -i bash -c "npx gemini-flow status"'
alias gf-monitor='sudo -u gemini-user -i bash -c "npx gemini-flow monitor"'
alias gf-health='sudo -u gemini-user -i bash -c "npx gemini-flow health"'

# === LOCAL PRIVATE LLM (Nemotron 3 120B via llama.cpp) ===
alias llm-health='curl -s http://${LOCAL_LLM_HOST:-192.168.2.48}:${LOCAL_LLM_PORT:-8080}/health | python3 -m json.tool'
alias llm-models='curl -s http://${LOCAL_LLM_HOST:-192.168.2.48}:${LOCAL_LLM_PORT:-8080}/v1/models | python3 -m json.tool'
llm-ask() {
  curl -s "http://${LOCAL_LLM_HOST:-192.168.2.48}:${LOCAL_LLM_PORT:-8080}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${LOCAL_LLM_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"$*\"}], \"max_tokens\": 2048}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
}

# === LOCAL LLM PROXY (Anthropic → OpenAI translation) ===
alias llm-proxy-start='sudo supervisorctl start local-llm-proxy'
alias llm-proxy-stop='sudo supervisorctl stop local-llm-proxy'
alias llm-proxy-status='sudo supervisorctl status local-llm-proxy'
alias llm-proxy-logs='tail -f /var/log/local-llm-proxy.log'

# === USER SWITCHING ===
alias as-gemini='sudo -u gemini-user -i'
alias as-openai='sudo -u openai-user -i'
alias as-zai='sudo -u zai-user -i'
alias as-deepseek='sudo -u deepseek-user -i'
alias as-local='sudo -u local-private -i'

# === CUDA ===
alias nvcc-version='nvcc --version'
alias cuda-info='nvidia-smi && echo && nvcc --version'
alias ptx-compile='nvcc -ptx'
alias gpu-watch='watch -n1 nvidia-smi'

# === USAGE ===
alias claude-usage='claude usage 2>/dev/null || echo "Run inside claude session"'
alias claude-monitor='~/.cargo/bin/claude-monitor'

# === TURBOFLOW META ===
turbo-status() {
    echo "============================================"
    echo "  TurboFlow 4.0 Status Check"
    echo "============================================"
    echo ""
    echo "Core:"
    claude --version 2>/dev/null | head -1 && echo "  ✓ Claude Code" || echo "  ✗ Claude Code"
    ruflo --version 2>/dev/null && echo "  ✓ Ruflo" || echo "  ✗ Ruflo"
    echo ""
    echo "V4 Systems:"
    command -v bd &>/dev/null && echo "  ✓ Beads (cross-session memory)" || echo "  ✗ Beads (npm i -g beads-cli)"
    command -v gitnexus &>/dev/null && echo "  ✓ GitNexus (codebase graph)" || echo "  ✗ GitNexus (npm i -g gitnexus)"
    echo "  Agent Teams: ${CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS:-off}"
    echo ""
    echo "Memory:"
    [ -d ".beads" ] && echo "  ✓ Beads initialized" || echo "  ○ Beads not initialized (bd init)"
    echo ""
    echo "Plugins:"
    ruflo plugins list 2>/dev/null | head -10 || echo "  Run: rf-plugins"
    echo ""
    echo "Services:"
    sudo supervisorctl status 2>/dev/null | head -15 || echo "  supervisord not running"
    echo ""
    echo "Workspace:"
    [ -f "CLAUDE.md" ] && echo "  ✓ CLAUDE.md" || echo "  ✗ CLAUDE.md missing"
    git worktree list 2>/dev/null | head -5
    echo ""
    echo "Agents: $(ls -1 ~/agents/*.md 2>/dev/null | wc -l) subagent templates"
    echo "Skills: $(ls -d ~/.claude/skills/*/ 2>/dev/null | wc -l) MCP skills"
}

turbo-help() {
    echo "TurboFlow 4.0 — Quick Reference"
    echo ""
    echo "Orchestration (Ruflo):"
    echo "  rf-wizard          Interactive setup"
    echo "  rf-swarm           Hierarchical swarm (8 agents)"
    echo "  rf-hybrid          Hierarchical-mesh swarm (15 agents)"
    echo "  rf-spawn coder     Spawn a coder agent"
    echo "  rf-doctor          Health check + auto-fix"
    echo "  rf-plugins         List plugins"
    echo ""
    echo "Memory (3-tier):"
    echo "  bd-ready           Check project state (session start)"
    echo "  bd-add             Record issue/decision/blocker"
    echo "  mem-search Q       Search ruflo memory"
    echo "  ruv-remember K V   Store in AgentDB"
    echo "  ruv-recall Q       Query AgentDB"
    echo ""
    echo "Isolation:"
    echo "  wt-add agent-1     Create worktree for agent"
    echo "  wt-remove agent-1  Clean up worktree"
    echo "  wt-list            Show all worktrees"
    echo ""
    echo "Quality:"
    echo "  aqe-generate       Generate tests (QE plugin)"
    echo "  aqe-gate           Quality gate check"
    echo "  os-init            Initialize OpenSpec"
    echo ""
    echo "Intelligence:"
    echo "  gnx-analyze        Index repo (GitNexus)"
    echo "  hooks-train        Deep pretrain on codebase"
    echo "  neural-train       Train neural patterns"
    echo ""
    echo "Users: as-gemini | as-openai | as-zai | as-deepseek"
    echo "GPU:   cuda-info | gpu-watch"
    echo "Status: turbo-status"
}

# Agentic Flow CachyOS Workstation ZSH Configuration
# Interactive development environment with all providers

# ===== Oh-My-Zsh Configuration =====
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(
    git
    docker
    docker-compose
    kubectl
    node
    npm
    python
    pip
    vscode
    zsh-syntax-highlighting
)

# Load Oh-My-Zsh
source $ZSH/oh-my-zsh.sh

# ===== Environment Variables =====
export WORKSPACE="$HOME/workspace"
export MODELS_DIR="$HOME/models"
export EDITOR="vim"
export VISUAL="vim"

# Add local node_modules binaries to PATH (project-local dependencies)
export PATH="/tmp/agentic-flow/node_modules/.bin:$PATH"
# Add npm global binaries to PATH
export PATH="$HOME/.npm-global/bin:$PATH"

# ===== Agentic Flow Core Aliases =====
alias af='agentic-flow'
alias afl='agentic-flow --list'
alias afh='agentic-flow --help'
alias afv='agentic-flow --version'

# ===== Quick Agent Execution =====
alias coder='agentic-flow --agent coder --task'
alias reviewer='agentic-flow --agent reviewer --task'
alias researcher='agentic-flow --agent researcher --task'
alias tester='agentic-flow --agent tester --task'
alias planner='agentic-flow --agent planner --task'
alias architect='agentic-flow --agent system-architect --task'
alias backend='agentic-flow --agent backend-dev --task'

# ===== Provider-Specific Shortcuts =====
# Google Gemini (fast, cost-effective)
alias af-gemini='agentic-flow --provider gemini'
alias gemini-coder='agentic-flow --provider gemini --agent coder --task'

# OpenAI (GPT-4o)
alias af-openai='agentic-flow --provider openai'
alias gpt-coder='agentic-flow --provider openai --agent coder --task'

# Anthropic Claude (highest quality)
alias af-claude='agentic-flow --provider anthropic'
alias claude-coder='agentic-flow --provider anthropic --agent coder --task'

# OpenRouter (99% cost savings)
alias af-router='agentic-flow --provider openrouter'
alias router-coder='agentic-flow --provider openrouter --agent coder --task'

# Xinference (free local inference via RAGFlow network)
alias af-local='agentic-flow --provider xinference'
alias local-coder='agentic-flow --provider xinference --agent coder --task'

# ONNX (offline, GPU-accelerated)
alias af-offline='agentic-flow --provider onnx --local-only'
alias offline-coder='agentic-flow --provider onnx --local-only --agent coder --task'

# ===== Intelligent Router Shortcuts =====
alias af-optimize='agentic-flow --optimize'
alias af-perf='agentic-flow --optimize --priority performance'
alias af-cost='agentic-flow --optimize --priority cost'
alias af-quality='agentic-flow --optimize --priority quality'
alias af-balanced='agentic-flow --optimize --priority balanced'

# ===== MCP Server Management =====
alias mcp='agentic-flow mcp'
alias mcp-start='agentic-flow mcp start'
alias mcp-stop='agentic-flow mcp stop'
alias mcp-list='agentic-flow mcp list'
alias mcp-status='agentic-flow mcp status'
alias mcp-add='agentic-flow mcp add'
alias mcp-remove='agentic-flow mcp remove'

# ===== Gemini-Flow Orchestration =====
alias gf='gemini-flow'
alias gf-init='gemini-flow init --protocols a2a,mcp --topology hierarchical'
alias gf-spawn='gemini-flow agents spawn'
alias gf-monitor='gemini-flow monitor --protocols --performance'
alias gf-status='gemini-flow swarm status'
alias gf-health='gemini-flow health-check'

# Gemini-Flow agent swarms (66 specialized agents)
alias gf-swarm='gemini-flow agents spawn --count 66 --coordination intelligent'
alias gf-enterprise='gemini-flow agents spawn --specialization enterprise-ready'
alias gf-architect='gemini-flow agents spawn --specialization system-architect --count 5'
alias gf-coder='gemini-flow agents spawn --specialization master-coder --count 12'
alias gf-research='gemini-flow agents spawn --specialization research-scientist --count 8'
alias gf-analyst='gemini-flow agents spawn --specialization data-analyst --count 10'

# Google AI Services shortcuts
alias gf-veo3='gemini-flow veo3 create'
alias gf-imagen='gemini-flow imagen4 create'
alias gf-lyria='gemini-flow lyria compose'
alias gf-chirp='gemini-flow chirp synthesize'
alias gf-scientist='gemini-flow co-scientist research'
alias gf-mariner='gemini-flow mariner automate'

# ===== Testing & Validation =====
alias test-providers='~/scripts/test-all-providers.sh'
alias test-gpu='nvidia-smi || rocm-smi || echo "No GPU detected"'
alias test-xinference='curl -s http://172.18.0.11:9997/v1/models | jq'
alias test-health='curl -s http://localhost:8080/health | jq'
alias test-gemini-flow='~/scripts/test-gemini-flow.sh'

# ===== Development Helpers =====
alias ws='cd $WORKSPACE'
alias agents='cd $WORKSPACE && find . -name "*.md" -path "*/.claude/agents/*" | head -20'
alias logs='tail -f ~/.claude-flow/logs/*.log 2>/dev/null || echo "No logs found"'
alias memory='du -sh ~/.claude-flow/memory'
alias models='ls -lh $MODELS_DIR'

# ===== Git Shortcuts =====
alias gs='git status'
alias ga='git add'
alias gc='git commit -m'
alias gp='git push'
alias gpl='git pull'
alias gd='git diff'
alias glog='git log --oneline --graph --all --decorate'

# ===== Docker Shortcuts (if needed from inside container) =====
alias dps='docker ps'
alias dpa='docker ps -a'
alias dimages='docker images'

# ===== System Monitoring =====
alias gpu='nvidia-smi'
alias mem='free -h'
alias cpu='htop'
alias disk='df -h'
alias top='btop'

# ===== Useful Functions =====

# Quick agent run with streaming
function afr() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: afr <agent> <task>"
        echo "Example: afr coder 'Build REST API'"
        return 1
    fi
    agentic-flow --agent "$1" --task "$2" --stream
}

# Run optimized agent with custom priority
function afo() {
    if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
        echo "Usage: afo <priority> <agent> <task>"
        echo "Priority: performance | cost | quality | balanced"
        echo "Example: afo performance coder 'Build API'"
        return 1
    fi
    agentic-flow --optimize --priority "$1" --agent "$2" --task "$3"
}

# Test specific provider
function test-provider() {
    if [ -z "$1" ]; then
        echo "Usage: test-provider <provider>"
        echo "Providers: gemini, openai, anthropic, openrouter, xinference, onnx"
        return 1
    fi
    echo "Testing $1..."
    agentic-flow --provider "$1" --agent coder --task "Write a Python hello world" --max-tokens 50
}

# List all available agents
function list-agents() {
    agentic-flow --list | grep -E "^\s+\-" | wc -l | xargs echo "Available agents:"
    agentic-flow --list | grep -E "^\s+\-"
}

# Quick MCP tool search
function mcp-search() {
    if [ -z "$1" ]; then
        echo "Usage: mcp-search <keyword>"
        return 1
    fi
    agentic-flow mcp list | grep -i "$1"
}

# Show API key status
function check-keys() {
    echo "API Key Status:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    [ -n "$ANTHROPIC_API_KEY" ] && echo "✅ ANTHROPIC_API_KEY" || echo "❌ ANTHROPIC_API_KEY"
    [ -n "$OPENAI_API_KEY" ] && echo "✅ OPENAI_API_KEY" || echo "❌ OPENAI_API_KEY"
    [ -n "$GOOGLE_GEMINI_API_KEY" ] && echo "✅ GOOGLE_GEMINI_API_KEY" || echo "❌ GOOGLE_GEMINI_API_KEY"
    [ -n "$GOOGLE_API_KEY" ] && echo "✅ GOOGLE_API_KEY (Gemini-Flow)" || echo "❌ GOOGLE_API_KEY (Gemini-Flow)"
    [ -n "$GOOGLE_CLOUD_PROJECT" ] && echo "✅ GOOGLE_CLOUD_PROJECT" || echo "❌ GOOGLE_CLOUD_PROJECT"
    [ -n "$OPENROUTER_API_KEY" ] && echo "✅ OPENROUTER_API_KEY" || echo "❌ OPENROUTER_API_KEY"
    [ -n "$E2B_API_KEY" ] && echo "✅ E2B_API_KEY" || echo "❌ E2B_API_KEY"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Spawn Gemini-Flow agent swarm
function gf-deploy() {
    if [ -z "$1" ]; then
        echo "Usage: gf-deploy <objective> [agent-count]"
        echo "Example: gf-deploy 'enterprise-transformation' 50"
        return 1
    fi
    local count=${2:-20}
    echo "Deploying Gemini-Flow swarm with $count agents..."
    gemini-flow agents spawn \
        --objective "$1" \
        --count "$count" \
        --protocols a2a,mcp \
        --coordination intelligent
}

# ===== Welcome Message =====
function agentic-welcome() {
    clear
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   🚀 Agentic Flow CachyOS Workstation                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📦 Workspace: $WORKSPACE"
    echo "🤖 Agents: $(agentic-flow --list 2>/dev/null | grep -c "^\s\+-" || echo "Loading...")"
    echo "🔧 MCP Tools: $(agentic-flow mcp list 2>/dev/null | wc -l || echo "Starting...")"
    echo "💾 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Not detected")"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "🎯 Quick Start:"
    echo "  af --agent coder --task 'Build REST API'   # Run agent"
    echo "  af-optimize --agent reviewer --task '...'  # Use router"
    echo "  af-gemini --agent coder --task '...'       # Force Gemini"
    echo "  af-local --agent coder --task '...'        # Use Xinference"
    echo ""
    echo "🔧 Useful Commands:"
    echo "  test-providers    # Test all model providers"
    echo "  test-gemini-flow  # Test Gemini-Flow orchestration"
    echo "  check-keys        # Check API key status"
    echo "  mcp-list          # List all MCP tools"
    echo "  test-gpu          # Check GPU status"
    echo "  list-agents       # Show all agents"
    echo ""
    echo "🐝 Gemini-Flow (66-Agent Swarms):"
    echo "  gf-swarm          # Deploy 66 specialized agents"
    echo "  gf-deploy <obj>   # Deploy custom swarm"
    echo "  gf-monitor        # Monitor A2A protocols"
    echo ""
    echo "📚 Help:"
    echo "  afh               # Agentic Flow help"
    echo "  afl               # List agents"
    echo "  gf --help         # Gemini-Flow help"
    echo "  mcp-status        # MCP server status"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Show welcome message on shell start
agentic-welcome

# ===== Completion Enhancements =====
# Enable command completion
autoload -Uz compinit
compinit

# Case-insensitive completion
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'

# ===== History Configuration =====
HISTFILE=~/.zsh_history
HISTSIZE=10000
SAVEHIST=10000
setopt SHARE_HISTORY
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_FIND_NO_DUPS

# ===== Prompt Customization =====
# Show current git branch in prompt
autoload -Uz vcs_info
precmd_vcs_info() { vcs_info }
precmd_functions+=( precmd_vcs_info )
setopt prompt_subst
RPROMPT=\$vcs_info_msg_0_
zstyle ':vcs_info:git:*' formats '%F{yellow}(%b)%f'
zstyle ':vcs_info:*' enable git

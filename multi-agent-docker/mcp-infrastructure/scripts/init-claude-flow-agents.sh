#!/bin/bash
# Initialize Claude-Flow v110 Goal Planner and SAFLA Neural agents
# This script sets up the advanced AI capabilities in the Docker environment

set -e

echo "🚀 Initializing Claude-Flow v110 Advanced Agents..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure we're in the workspace directory
cd /workspace || exit 1

# Check if claude-flow is available
if ! command_exists claude-flow && ! npm list -g claude-flow@alpha >/dev/null 2>&1; then
    echo "⚠️  claude-flow not found. Installing..."
    npm install -g claude-flow@alpha || {
        echo "❌ Failed to install claude-flow"
        exit 1
    }
fi

# Initialize Goal Planner
echo ""
echo "🎯 Initializing Goal Planner Agent..."
echo "   - Applies GOAP with A* pathfinding"
echo "   - Calculates optimal action sequences"
echo "   - Adapts dynamically to changing conditions"

claude-flow goal init --force || {
    echo "⚠️  Goal Planner may already be initialized"
}

# Initialize SAFLA Neural Agent
echo ""
echo "🧠 Initializing SAFLA Neural Agent..."
echo "   - Four-tier memory architecture"
echo "   - Pattern recognition and learning"
echo "   - Persistent knowledge accumulation"

claude-flow neural init --force || {
    echo "⚠️  SAFLA Neural may already be initialized"
}

# Create example configurations
echo ""
echo "📝 Creating example configurations..."

# Goal Planner example config
cat > /workspace/.claude-flow-goal-example.yaml << 'EOF'
# Example Goal Planner configuration for deployment task
goal: deploy_application
initial_state:
  - code_tested: false
  - docker_built: false
  - deployed: false
  
desired_state:
  - deployed: true
  
actions:
  - name: run_tests
    preconditions:
      - code_tested: false
    effects:
      - code_tested: true
    cost: 1
    
  - name: build_docker
    preconditions:
      - code_tested: true
      - docker_built: false
    effects:
      - docker_built: true
    cost: 2
    
  - name: deploy
    preconditions:
      - docker_built: true
      - deployed: false
    effects:
      - deployed: true
    cost: 3
EOF

# Neural Agent example config
cat > /workspace/.claude-flow-neural-example.json << 'EOF'
{
  "name": "project_assistant",
  "memory_config": {
    "vector_dimension": 768,
    "episodic_retention": 100,
    "semantic_compression": true,
    "working_memory_size": 10
  },
  "learning_config": {
    "pattern_threshold": 0.75,
    "style_adaptation": true,
    "context_window": 2048
  }
}
EOF

# Verify installation
echo ""
echo "✅ Checking agent status..."
claude-flow status || {
    echo "⚠️  Could not verify agent status"
}

# Create helper functions file
cat > /workspace/.claude-flow-helpers.sh << 'EOF'
#!/bin/bash
# Claude-Flow v110 Helper Functions

# Plan a deployment using Goal Planner
cf_plan_deployment() {
    local project="${1:-current}"
    echo "Planning deployment for $project..."
    claude-flow goal plan \
        --config .claude-flow-goal-example.yaml \
        --project "$project"
}

# Start a learning session with Neural agent
cf_learn_project() {
    local context="${1:-.}"
    echo "Starting learning session for $context..."
    claude-flow neural learn \
        --config .claude-flow-neural-example.json \
        --context "$context"
}

# Query Neural agent memory
cf_recall() {
    local query="$1"
    echo "Querying neural memory: $query"
    claude-flow neural recall --query "$query"
}

# Show agent capabilities
cf_capabilities() {
    echo "=== Goal Planner Capabilities ==="
    claude-flow goal capabilities
    echo ""
    echo "=== Neural Agent Capabilities ==="
    claude-flow neural capabilities
}

echo "Claude-Flow v110 helpers loaded. Available commands:"
echo "  - cf_plan_deployment [project]"
echo "  - cf_learn_project [context]"
echo "  - cf_recall <query>"
echo "  - cf_capabilities"
EOF

chmod +x /workspace/.claude-flow-helpers.sh

echo ""
echo "✨ Claude-Flow v110 agents initialized successfully!"
echo ""
echo "📚 Example configurations created:"
echo "   - .claude-flow-goal-example.yaml"
echo "   - .claude-flow-neural-example.json"
echo "   - .claude-flow-helpers.sh"
echo ""
echo "💡 Quick start:"
echo "   source .claude-flow-helpers.sh"
echo "   cf_capabilities"
echo ""
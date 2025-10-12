# CachyOS Workstation - Quick Start Guide

**Get your Agentic Flow development environment running in 5 minutes.**

---

## Prerequisites

- Docker with GPU support (NVIDIA runtime or AMD ROCm)
- At least one API key (Anthropic, OpenAI, Gemini, or OpenRouter)
- Access to `docker_ragflow` network (for Xinference)

---

## Step 1: Setup (2 minutes)

```bash
# Navigate to cachyos directory
cd docker/cachyos

# Copy environment template
cp config/providers.env.template .env

# Edit .env and add at least one API key
nano .env
```

**Required** (add at least one):
```bash
GOOGLE_GEMINI_API_KEY=AIza...         # Recommended: Fast & cheap
OPENAI_API_KEY=sk-proj-...            # Good quality
ANTHROPIC_API_KEY=sk-ant-...          # Highest quality
OPENROUTER_API_KEY=sk-or-v1-...       # 99% cost savings
```

---

## Step 2: Build & Start (2 minutes)

```bash
# Build the image
docker-compose -f docker-compose.workstation.yml build

# Start the container
docker-compose -f docker-compose.workstation.yml up -d

# Check it's running
docker ps | grep agentic-flow-cachyos
```

---

## Step 3: Shell In (10 seconds)

```bash
# Enter the container
docker exec -it agentic-flow-cachyos zsh

# You're now inside! 🎉
```

---

## Step 4: Test It Works (1 minute)

```bash
# Inside container

# Check API keys
check-keys

# Test a simple agent
af-gemini --agent coder --task "Write a Python hello world function"

# Test all providers
test-providers
```

---

## Common Commands (Reference)

### Agent Execution
```bash
# Use intelligent router (auto-selects best model)
af-optimize --agent coder --task "Build REST API"

# Force specific provider
af-gemini --agent coder --task "..."   # Google Gemini
af-openai --agent coder --task "..."   # OpenAI GPT-4o
af-claude --agent coder --task "..."   # Anthropic Claude
af-local --agent coder --task "..."    # Xinference (free)
af-offline --agent coder --task "..."  # ONNX (offline)

# Quick aliases
coder "Build authentication API"
reviewer "Review this code"
researcher "Analyze AI trends 2025"
```

### Management
```bash
# List available agents
afl

# MCP server management
mcp-start
mcp-list
mcp-status

# Check GPU
test-gpu

# Test all providers
test-providers
```

---

## Troubleshooting

### "API key not set"
```bash
# Exit container
exit

# Edit .env file
nano .env

# Restart container
docker-compose -f docker-compose.workstation.yml restart

# Shell back in
docker exec -it agentic-flow-cachyos zsh
```

### "Xinference not reachable"
```bash
# Check network
docker network inspect docker_ragflow | grep agentic-flow-cachyos

# If not connected, recreate container
docker-compose -f docker-compose.workstation.yml down
docker-compose -f docker-compose.workstation.yml up -d
```

### "GPU not detected"
```bash
# Check GPU from host
nvidia-smi

# Check Docker runtime
docker inspect agentic-flow-cachyos | grep -i runtime

# If runtime is not 'nvidia', edit docker-compose.workstation.yml:
#   runtime: nvidia
```

---

## Next Steps

- **Read Full Docs**: [README.workstation.md](README.workstation.md)
- **Configure Router**: Edit `~/.config/agentic-flow/router.config.json`
- **Add Custom Agents**: Create in `~/.claude/agents/custom/`
- **Upstream Merges**: See [UPSTREAM.md](UPSTREAM.md)

---

## Quick Examples

### Example 1: Build a REST API
```bash
af-gemini --agent backend-dev --task "
Create a REST API with:
- JWT authentication
- User CRUD operations
- PostgreSQL database
- Docker deployment
"
```

### Example 2: Code Review
```bash
af-claude --agent reviewer --task "
Review this code for:
- Security vulnerabilities
- Performance issues
- Best practices
- Test coverage
"
```

### Example 3: Generate Tests
```bash
af-cost --agent tester --task "
Generate comprehensive unit tests for:
- User authentication
- API endpoints
- Database operations
- Edge cases
"
```

---

## Stop/Restart Container

```bash
# Stop (keeps data)
docker-compose -f docker-compose.workstation.yml stop

# Start again
docker-compose -f docker-compose.workstation.yml start

# Restart
docker-compose -f docker-compose.workstation.yml restart

# Remove (keeps volumes)
docker-compose -f docker-compose.workstation.yml down

# Remove everything (including data)
docker-compose -f docker-compose.workstation.yml down -v
```

---

## Resources

- **Full Documentation**: [README.workstation.md](README.workstation.md)
- **Upstream Merges**: [UPSTREAM.md](UPSTREAM.md)
- **Main Project**: [../../README.md](../../README.md)
- **GitHub**: https://github.com/ruvnet/agentic-flow

---

**You're ready to go! Start building with AI agents.** 🚀

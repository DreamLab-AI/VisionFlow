# Developer Guide

**Complete developer documentation for extending and deploying Turbo Flow Claude.**

---

## Contents

1. **[Architecture](architecture.md)** - System design and components
2. **[Building Skills](building-skills.md)** - Create custom Claude Code skills
3. **[DevPod Setup](devpod-setup.md)** - Cloud development environments
4. **[Cloud Deployment](cloud-deployment.md)** - Production deployment guides
5. **[Command Reference](command-reference.md)** - Claude Flow aliases and commands

---

## Quick Start for Developers

### Build from Source

```bash
git clone https://github.com/marcuspat/turbo-flow-claude.git
cd turbo-flow-claude

# Build unified container
docker build -f Dockerfile.unified -t turbo-flow-unified:latest .

# Test build
docker run --rm turbo-flow-unified:latest supervisorctl status
```

### Development Workflow

```bash
# Make changes to Dockerfile or configs
vim Dockerfile.unified
vim unified-config/supervisord.unified.conf

# Rebuild
docker build -f Dockerfile.unified -t turbo-flow-unified:test .

# Test
docker run --rm --name test-container \
  -e ANTHROPIC_API_KEY=your-key \
  turbo-flow-unified:test \
  bash -c "claude --version && echo 'Success!'"
```

---

## Architecture Overview

### Multi-User System

Four isolated Linux users with credential separation:

```
┌─────────────────────────────────────┐
│     Turbo Flow Unified Container     │
├─────────────────────────────────────┤
│  devuser (1000)   │ Claude Code     │
│  gemini-user (1001) │ Gemini tools  │
│  openai-user (1002) │ OpenAI tools  │
│  zai-user (1003)   │ Z.AI service   │
└─────────────────────────────────────┘
```

### Service Architecture

```
supervisord (PID 1)
├── dbus (priority 10) - System messaging
├── dbus-user (priority 15) - User session bus
├── sshd (priority 50) - SSH server
├── xvnc (priority 100) - VNC server
├── xfce4 (priority 200) - Desktop environment
├── management-api (priority 300) - HTTP API
├── code-server (priority 400) - Web IDE
├── claude-zai (priority 500) - Z.AI service
├── gemini-flow (priority 600) - Gemini orchestration
└── tmux-autostart (priority 900) - Workspace session
```

See [Architecture](architecture.md) for complete details.

---

## Creating Custom Skills

Skills are stdio-based tools that integrate with Claude Code.

### Skill Structure

```
~/.claude/skills/my-skill/
├── SKILL.md              # Skill definition
└── tools/
    └── my_tool.py        # Tool implementation
```

### Example Skill

**SKILL.md**:
```markdown
---
name: My Custom Skill
description: Does amazing things
---

# My Custom Skill

This skill provides...

## When to Use

Use this skill when...

## Instructions

To use this skill:
1. Describe what you want
2. Skill will execute
3. Results returned
```

**my_tool.py**:
```python
#!/usr/bin/env python3
import sys
import json

def main():
    # Read input from stdin
    input_data = sys.stdin.read()

    # Process
    result = process(input_data)

    # Write output to stdout
    print(json.dumps(result))

if __name__ == "__main__":
    main()
```

See [Building Skills](building-skills.md) for complete guide.

---

## Deployment Options

### 1. Standalone Docker

```bash
docker build -f Dockerfile.unified -t turbo-flow-unified .
docker run -d --name turbo-flow \
  -p 2222:22 -p 5901:5901 -p 9090:9090 \
  --env-file .env \
  --runtime=nvidia \
  turbo-flow-unified:latest
```

### 2. DevPod (Cloud Development)

```bash
devpod up https://github.com/marcuspat/turbo-flow-claude --ide vscode
```

Supports:
- GitHub Codespaces
- Google Cloud Shell
- AWS Cloud9
- DigitalOcean
- Civo

See [DevPod Setup](devpod-setup.md) for provider configuration.

### 3. Cloud Deployment

**Spot/Rackspace**:
```bash
# See cloud-deployment.md for complete guide
spot-deploy --config turbo-flow-config.yaml
```

**Kubernetes**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: turbo-flow
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: turbo-flow
        image: turbo-flow-unified:latest
        ports:
        - containerPort: 9090
```

See [Cloud Deployment](cloud-deployment.md) for production setups.

---

## API Development

### Management API Structure

```
multi-agent-docker/management-api/
├── server.js                # Fastify server
├── routes/
│   ├── tasks.js            # Task endpoints
│   └── status.js           # Status endpoints
├── middleware/
│   └── auth.js             # Bearer token auth
└── utils/
    ├── process-manager.js  # Task execution
    ├── system-monitor.js   # System metrics
    └── metrics.js          # Prometheus metrics
```

### Adding New Endpoints

**routes/custom.js**:
```javascript
async function customRoutes(fastify, options) {
  fastify.get('/v1/custom', async (request, reply) => {
    return { message: 'Custom endpoint' };
  });
}

module.exports = customRoutes;
```

**server.js**:
```javascript
app.register(require('./routes/custom'), {
  prefix: ''
});
```

---

## Testing

### Unit Tests

```bash
cd multi-agent-docker/tests
npm test
```

### Integration Tests

```bash
./run-tests.sh --integration
```

### E2E Tests

```bash
./run-tests.sh --e2e
```

### Manual Testing

```bash
# Test VNC
docker exec turbo-flow-unified supervisorctl status xvnc

# Test API
curl http://localhost:9090/health

# Test Claude CLI
docker exec -u devuser turbo-flow-unified claude --version
```

---

## Build Configuration

### Dockerfile Phases

The Dockerfile.unified has 17 build phases:

1. Base packages (ArchLinux/CachyOS)
2. CUDA toolkit
3. Rust toolchain
4. Multi-user setup
5. Node.js globals
6. Python virtualenv
7. devuser configuration
8. gemini-user setup
9. openai-user setup
10. zai-user setup
11. VNC/desktop
12. SSH server
13. Application files
14. Supervisord
15. Environment variables
16. Ports and volumes
17. Entrypoint

### Customization

```dockerfile
# Add custom packages
RUN pacman -S --noconfirm custom-package

# Install Python packages
RUN /opt/venv/bin/pip install custom-library

# Add custom skills
COPY my-skills/ /home/devuser/.claude/skills/
```

---

## Environment Variables

### Required

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### Optional

```bash
GOOGLE_GEMINI_API_KEY=...
OPENAI_API_KEY=...
GITHUB_TOKEN=...
ZAI_API_KEY=...
```

### Service Configuration

```bash
MANAGEMENT_API_KEY=your-secure-token
CLAUDE_WORKER_POOL_SIZE=4
CLAUDE_MAX_QUEUE_SIZE=50
GPU_ACCELERATION=true
```

See [../SETUP.md](../SETUP.md) for complete environment reference.

---

## Performance Tuning

### Resource Allocation

```yaml
# docker-compose.unified.yml
deploy:
  resources:
    limits:
      memory: 64G
      cpus: '32'
    reservations:
      memory: 16G
      cpus: '8'
```

### Z.AI Worker Pool

```bash
# Increase concurrent workers
CLAUDE_WORKER_POOL_SIZE=8
CLAUDE_MAX_QUEUE_SIZE=100
```

### Database Optimization

```bash
# Enable SQLite WAL mode for better concurrency
docker exec turbo-flow-unified sqlite3 /path/to/db.sqlite \
  "PRAGMA journal_mode=WAL;"
```

---

## Security Considerations

### Default Credentials

**⚠️ CRITICAL**: Change all default credentials in production:

```bash
# Generate secure API key
MANAGEMENT_API_KEY=$(openssl rand -hex 32)

# Set strong SSH password
docker exec turbo-flow-unified passwd devuser

# Set VNC password
docker exec turbo-flow-unified vncpasswd
```

### Network Isolation

```bash
# Create isolated network
docker network create --driver bridge \
  --subnet=172.20.0.0/16 \
  turbo-flow-net

# Run container in isolated network
docker run --network turbo-flow-net ...
```

### API Security

- Use HTTPS in production (reverse proxy with TLS)
- Rotate Bearer tokens regularly
- Implement rate limiting (already included)
- Monitor access logs

See [../SECURITY.md](../SECURITY.md) for complete security guide.

---

## Contributing

### Development Setup

```bash
git clone https://github.com/marcuspat/turbo-flow-claude.git
cd turbo-flow-claude
git checkout -b feature/my-feature

# Make changes
vim Dockerfile.unified

# Test
docker build -f Dockerfile.unified -t turbo-flow-test .
./multi-agent-docker/tests/run-tests.sh

# Commit
git add .
git commit -m "Add feature: description"
git push origin feature/my-feature
```

### Pull Request Guidelines

1. Test all changes thoroughly
2. Update documentation
3. Follow existing code style
4. Include tests for new features
5. Update CHANGES.md

---

## Troubleshooting

### Build Failures

```bash
# Clear Docker build cache
docker builder prune -a

# Rebuild with no cache
docker build --no-cache -f Dockerfile.unified -t turbo-flow-unified .
```

### Service Failures

```bash
# Check supervisord logs
docker exec turbo-flow-unified tail -f /var/log/supervisord.log

# Restart specific service
docker exec turbo-flow-unified supervisorctl restart {service}
```

### Permission Issues

```bash
# Fix ownership
docker exec turbo-flow-unified chown -R devuser:devuser /home/devuser
```

---

## Resources

- **Architecture**: [architecture.md](architecture.md)
- **Skills Guide**: [building-skills.md](building-skills.md)
- **DevPod Setup**: [devpod-setup.md](devpod-setup.md)
- **Cloud Deploy**: [cloud-deployment.md](cloud-deployment.md)
- **Commands**: [command-reference.md](command-reference.md)

---

**Ready to build and extend Turbo Flow!** 🛠️

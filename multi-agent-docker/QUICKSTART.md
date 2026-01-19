# Quick Start Guide - Turbo Flow Unified Container

## Connection Information

### VNC Desktop
```
URL: vnc://localhost:5901
Password: None (no authentication)
Resolution: 2048x2048
Desktop: Openbox + tint2 panel
VNC Server: x11vnc + Xvfb
```

**VNC Client Recommendations:**
- **Linux (CachyOS)**: `sudo pacman -S tigervnc` then `vncviewer localhost:5901`
- **Android**: bVNC - Remote Desktop (Google Play Store)
- **iOS**: VNC Viewer by RealVNC (App Store)

**What You'll See:**
- 9 colorful terminal windows in 3x3 grid with custom banners identifying each purpose
- tint2 taskbar at bottom showing all open windows
- Chromium browser with DevTools enabled
- Right-click desktop for application menu

### SSH Access
```bash
ssh devuser@localhost -p 2222
Password: turboflow
```

### Web Interfaces
- **code-server**: http://localhost:8080 (no auth)
- **Management API**: http://localhost:9090/health
- **Swagger Docs**: http://localhost:9090/documentation

## Starting Claude Code

### Option 1: Via VNC Desktop
1. Connect to VNC at `vnc://localhost:5901`
2. Open terminal
3. Run: `tmux attach -t workspace`
4. In window 0, run: `dsp` (alias for claude --dangerously-skip-permissions)

### Option 2: Via SSH
```bash
ssh devuser@localhost -p 2222
tmux attach -t workspace
# Navigate to window 0 (Ctrl+B, 0)
dsp
```

### Option 3: Docker Exec
```bash
docker exec -it -u devuser agentic-workstation tmux attach -t workspace
# In window 0:
dsp
```

## tmux Workspace Layout

| Window | Name | Purpose |
|--------|------|---------|
| 0 | Claude-Main | Primary Claude Code workspace |
| 1 | Claude-Agent | Agent execution and testing |
| 2 | Services | `sudo supervisorctl status` |
| 3 | Development | Python/Rust/CUDA development |
| 4 | Logs | Service logs (split panes) |
| 5 | System | htop resource monitoring |
| 6 | VNC-Status | VNC server information |
| 7 | SSH-Shell | General purpose shell |
| 8 | Gemini-Shell | gemini-user (UID 1001) |
| 9 | OpenAI-Shell | openai-user (UID 1002) |
| 10 | ZAI-Shell | zai-user (UID 1003) |

### tmux Navigation
- **Ctrl+B, 0-9**: Switch to window 0-9
- **Ctrl+B, :select-window -t 10**: Switch to window 10
- **Ctrl+B, d**: Detach from session
- **Ctrl+B, [**: Scroll mode (q to exit)
- **Ctrl+B, "**: Split horizontal
- **Ctrl+B, %**: Split vertical

## Available Skills (18+ Total)

### Development & Research ⭐ NEW
- **jupyter-notebooks** - Interactive notebook execution with full MCP support
- **latex-documents** - Academic paper compilation (TeX Live, BibTeX, Beamer)
- **rust-development** - Complete Rust toolchain (cargo, rustfmt, clippy, WASM)
- **pytorch-ml** - Deep learning with PyTorch and CUDA acceleration
- **ffmpeg-processing** - Professional video/audio transcoding and editing

### Research & Web
- **perplexity** - Real-time AI research with UK focus
- **web-summary** - YouTube transcripts + web summaries (Z.AI powered)
- **playwright** - Browser automation
- **chrome-devtools** - Chrome debugging (remote port 9222)

### Graphics & Design
- **blender** - 3D modeling (socket port 2800)
- **qgis** - GIS operations (socket port 2801)
- **imagemagick** - Image processing
- **pbr-rendering** - PBR material generation

### Engineering
- **kicad** - PCB design and schematics
- **ngspice** - Circuit simulation

### Infrastructure
- **docker-manager** - Container management

### Specialized
- **wardley-maps** - Strategic mapping
- **logseq-formatted** - Knowledge management
- **import-to-ontology** - Ontology construction

## Using Perplexity Skill

### Quick Search
```javascript
// In Claude Code
Use perplexity to search for current UK mortgage rates from major banks
```

### Deep Research
```javascript
Research UK fintech landscape using perplexity skill:
- Focus on payment processing companies
- Market share analysis
- UK regulatory compliance
- Format as table with top 5 players
```

### Generate Optimized Prompt
```javascript
Use perplexity to generate an optimized prompt for researching:
Goal: Find best SaaS analytics tools for UK startups
Context: B2B product, £100K ARR, Series A
```

## User Switching

```bash
# From devuser, switch to other users
as-gemini   # Switch to gemini-user
as-openai   # Switch to openai-user
as-zai      # Switch to zai-user

# Or use sudo directly
sudo -u gemini-user -i
```

## Service Management

### Check All Services
```bash
sudo supervisorctl status
```

### Restart a Service
```bash
sudo supervisorctl restart management-api
sudo supervisorctl restart perplexity-mcp
```

### View Service Logs
```bash
sudo supervisorctl tail -f management-api
tail -f /var/log/perplexity-mcp.log
```

### Start/Stop Services
```bash
sudo supervisorctl stop perplexity-mcp
sudo supervisorctl start perplexity-mcp
```

## Management API

### Health Check
```bash
curl http://localhost:9090/health
```

### System Status (requires API key)
```bash
curl -H "X-API-Key: change-this-secret-key" \
  http://localhost:9090/api/status
```

### Metrics (Prometheus format)
```bash
curl http://localhost:9090/metrics
```

## Claude Flow Integration

### Initialize Project
```bash
cd ~/workspace/your-project
claude-flow init --force
```

### Swarm with Perplexity
```bash
cf-swarm "research UK AI regulations using perplexity and create compliance checklist"
```

### Hive Mind Deployment
```bash
claude-flow hive-mind spawn \
  "Market analysis: UK renewable energy sector" \
  --agents 10 \
  --tools perplexity,web-summary \
  --verify --pair
```

## API Keys Configuration

Edit `.env` file in project root:
```bash
ANTHROPIC_API_KEY=sk-ant-xxxxx
PERPLEXITY_API_KEY=pplx-xxxxx
GOOGLE_GEMINI_API_KEY=xxxxx
OPENAI_API_KEY=sk-xxxxx
GITHUB_TOKEN=ghp_xxxxx
```

Restart container after changing:
```bash
docker compose -f docker-compose.unified.yml restart
```

## Troubleshooting

### VNC Won't Connect
```bash
# Check if xvnc is running
docker exec agentic-workstation ps aux | grep Xvnc

# Restart VNC
docker exec agentic-workstation supervisorctl restart xvnc xfce4
```

### tmux Session Not Found
```bash
# Create manually
docker exec -u devuser agentic-workstation /home/devuser/.config/tmux-autostart.sh

# Or restart tmux-autostart service
docker exec agentic-workstation supervisorctl restart tmux-autostart
```

### MCP Server Failing
```bash
# Check logs
docker exec agentic-workstation cat /var/log/perplexity-mcp.error.log

# Check if node path is correct
docker exec agentic-workstation which node
# Should output: /usr/local/bin/node

# Restart MCP server
docker exec agentic-workstation supervisorctl restart perplexity-mcp
```

### Claude Code Can't Find Credentials
```bash
# Check mount
docker exec -u devuser agentic-workstation ls -la ~/.claude

# Check config
docker exec -u devuser agentic-workstation cat ~/.claude/config.json
```

## Performance Tips

### Monitor Resource Usage
```bash
# Inside container (tmux window 5)
htop

# From host
docker stats agentic-workstation
```

### View Service Performance
```bash
# Prometheus metrics
curl http://localhost:9090/metrics | grep -E "(cpu|memory|request)"
```

## Container Management

### Restart Container
```bash
docker compose -f docker-compose.unified.yml restart
```

### Stop Container
```bash
docker compose -f docker-compose.unified.yml down
```

### Start Container
```bash
docker compose -f docker-compose.unified.yml up -d
```

### Rebuild Image
```bash
./REBUILD.sh
# Or manually:
docker build -f Dockerfile.unified -t turbo-flow-unified:latest .
docker compose -f docker-compose.unified.yml up -d --force-recreate
```

### View Container Logs
```bash
docker compose -f docker-compose.unified.yml logs -f
```

## Default Credentials (⚠️ CHANGE IN PRODUCTION)

- **SSH**: devuser / turboflow
- **VNC**: No password (x11vnc runs with -nopw flag)
- **Management API**: X-API-Key: change-this-secret-key
- **code-server**: No authentication

## Quick Verification Checklist

After container starts:

- [ ] VNC accessible at vnc://localhost:5901
- [ ] SSH accessible at localhost:2222
- [ ] Management API responds: `curl http://localhost:9090/health`
- [ ] tmux workspace exists: `docker exec agentic-workstation tmux ls`
- [ ] All services running: `docker exec agentic-workstation supervisorctl status`
- [ ] Claude Code can start: `docker exec -it -u devuser agentic-workstation claude --version`
- [ ] Perplexity MCP running: `docker exec agentic-workstation supervisorctl status perplexity-mcp`

## Support & Documentation

- **Full Status**: `/FINAL_STATUS.md`
- **Upstream Analysis**: `/docs/upstream-analysis.md`
- **Perplexity Docs**: `/skills/perplexity/SKILL.md`
- **Research Templates**: `/skills/perplexity/docs/templates.md`

## Advanced Usage

### Execute Command in Specific User Context
```bash
# As gemini-user
docker exec -u gemini-user agentic-workstation gemini-flow --version

# As zai-user
docker exec -u zai-user agentic-workstation curl http://localhost:9600/health
```

### Mount Additional Volumes
Edit `docker-compose.unified.yml` and add:
```yaml
volumes:
  - /host/path:/container/path:rw
```

### Expose Additional Ports
Edit `docker-compose.unified.yml` and add:
```yaml
ports:
  - "1234:1234"
```

---

**Ready to Use!** Connect via VNC or SSH and start coding with Claude Code + 14 skills.

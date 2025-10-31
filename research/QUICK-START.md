# VisionFlow Quick Start Guide

**Last Updated:** 2025-10-23

## 🚀 For Developers

### View Complete Analysis
```bash
# Main architecture document (24,000 words)
cat /home/devuser/docs/visionflow-architecture-analysis.md

# Visual diagrams (11 diagrams)
cat /home/devuser/docs/visionflow-architecture-diagrams.md

# Integration recommendations (12,000 words)
cat /home/devuser/docs/visionflow-integration-recommendations.md

# Executive summary (this location)
cat /home/devuser/docs/VISIONFLOW-ANALYSIS-SUMMARY.md
```

### Quick Container Commands
```bash
# Service status
docker exec visionflow_container supervisorctl status

# View logs (live tail)
docker exec visionflow_container tail -f /app/logs/rust.log
docker exec visionflow_container tail -f /app/logs/vite.log
docker exec visionflow_container tail -f /app/logs/nginx.log

# Restart services
docker exec visionflow_container supervisorctl restart rust-backend
docker exec visionflow_container supervisorctl restart vite-dev
docker exec visionflow_container supervisorctl restart nginx
docker exec visionflow_container supervisorctl restart all

# Rebuild Rust backend
docker exec visionflow_container bash -c "cd /app && cargo build --features gpu"

# Check running processes
docker exec visionflow_container ps aux | grep -E "nginx|rust|vite|supervisord"
```

### Test Endpoints
```bash
# Main application
curl http://localhost:3001/

# Health check (TO BE IMPLEMENTED)
curl http://localhost:3001/health

# API test
curl http://localhost:3001/api/health

# File browser test
curl http://localhost:3001/browser-test

# Test WebSocket (requires wscat: npm install -g wscat)
wscat -c ws://localhost:3001/wss
```

### Debugging
```bash
# Check Nginx configuration
docker exec visionflow_container nginx -t

# View environment variables
docker exec visionflow_container env | grep -E "VITE_|RUST_|MCP_|PORT" | sort

# Check port bindings
docker port visionflow_container

# View container networking
docker inspect visionflow_container --format '{{json .NetworkSettings.Networks}}' | python3 -m json.tool

# Check supervisord config
docker exec visionflow_container cat /app/supervisord.dev.conf
```

## 📊 System Overview

### Architecture
```
Browser → Nginx :3001 → Backend :4000
                     → Frontend :5173
```

### Container Info
- **Name:** visionflow_container
- **IP:** 172.18.0.11 (docker_ragflow network)
- **Exposed Port:** 3001 (only)
- **Base Image:** nvidia/cuda:ubuntu22.04

### Key Services
| Service | Port | Purpose |
|---------|------|---------|
| Nginx | 3001 | Reverse proxy (exposed) |
| Rust Backend | 4000 | API + WebSocket (internal) |
| Vite Frontend | 5173 | React dev server (internal) |

### Important Paths
```
Source:        /app/src/ (Rust), /app/client/src/ (React)
Logs:          /app/logs/
Config:        /app/settings.yaml, /etc/nginx/nginx.conf
Binary:        /app/target/debug/webxr (dev)
Supervisord:   /app/supervisord.dev.conf
```

## ⚠️ Critical Issues

### 1. Health Checks Missing 🔴 CRITICAL
**Problem:** No `/health` endpoints implemented
**Action:** Implement health check routes in Rust backend
**Priority:** HIGH

### 2. Startup Logging Missing 🔴 CRITICAL
**Problem:** Backend logs end after "Starting Rust backend..."
**Action:** Add structured logging to main.rs
**Priority:** HIGH

### 3. Redundant Vite Proxy 🟡 MEDIUM
**Problem:** Vite proxies API calls (Nginx already does this)
**Action:** Remove proxy config from vite.config.ts
**Priority:** MEDIUM

### 4. Docker Compose Missing 🟡 MEDIUM
**Problem:** No docker-compose.yml in repository
**Action:** Create compose files for dev and production
**Priority:** MEDIUM

## 🎯 Next Steps

### Phase 1 (Week 1) - CRITICAL
1. Add health check endpoints
2. Implement startup logging
3. Remove Vite proxy config
4. Create Docker Compose files

### Phase 2 (Week 2-3) - IMPORTANT
5. Add Prometheus metrics
6. Set up Grafana dashboards
7. Implement rate limiting
8. Optimize for production

### Phase 3 (Week 4+) - ENHANCEMENT
9. Add HTTPS/SSL
10. Set up monitoring alerts
11. Create CI/CD pipeline
12. Load testing

## 📚 Documentation Structure

```
/home/devuser/docs/
├── visionflow-architecture-analysis.md      # Complete technical analysis
├── visionflow-architecture-diagrams.md      # 11 visual diagrams
├── visionflow-integration-recommendations.md # Implementation guide
├── VISIONFLOW-ANALYSIS-SUMMARY.md           # Executive summary
└── QUICK-START.md                           # This file
```

## 🔗 Quick Links

### URLs
- Application: http://localhost:3001/
- File Browser: http://localhost:3001/browser/
- API: http://localhost:3001/api/*
- WebSocket: ws://localhost:3001/wss

### External Services
- MCP Server: agentic-workstation:9500 (172.18.0.7)
- Copyparty: ragflow-es-01:3923 (172.18.0.4)
- Whisper: whisper-webui (172.18.0.6)
- Kokoro TTS: kokoro-tts-container (172.18.0.10)

## 💡 Tips

### Hot Reload
- **Frontend:** Edit `/app/client/src/` files, browser auto-reloads
- **Backend:** Restart with `supervisorctl restart rust-backend`
- **Nginx:** Reload with `supervisorctl restart nginx`

### GPU Status
Check CUDA availability:
```bash
docker exec visionflow_container nvidia-smi
```

### Network Debugging
```bash
# Ping MCP server
docker exec visionflow_container ping -c 3 agentic-workstation

# Test internal ports
docker exec visionflow_container curl -s http://localhost:4000/health
docker exec visionflow_container curl -s http://localhost:5173/
```

## 📞 Support

**Documentation:**
- Read `/home/devuser/docs/visionflow-architecture-analysis.md` for complete details
- Review `/home/devuser/docs/visionflow-integration-recommendations.md` for fixes

**Common Issues:**
- Service won't start → Check logs in `/app/logs/`
- Port conflicts → Check `docker port visionflow_container`
- GPU errors → Verify `nvidia-smi` works in container
- MCP connection fails → Check `agentic-workstation` is running

---

**Quick Start Complete!** 🎉

For full details, see the comprehensive documentation in `/home/devuser/docs/`.

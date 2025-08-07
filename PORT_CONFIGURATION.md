# Port Configuration - Critical Information

## Production/Dev Port Architecture

### External Access (What Users Connect To)
- **Port 3001** - Nginx reverse proxy (exposed to host)
- All client connections go through this port

### Internal Services (Docker Network Only)
- **Port 4000** - Rust backend API server
- **Port 5173** - Vite development server  
- **Port 24678** - Vite HMR (Hot Module Replacement)

## How It Works

```
User Browser
     ↓
Port 3001 (Nginx)
     ↓
  Proxies to:
     ├── /api/* → Rust Backend (4000)
     ├── /wss → WebSocket (4000)
     └── /* → Vite Frontend (5173)
```

## Environment Variables Override

In `docker-compose.dev.yml`:
- `SYSTEM_NETWORK_PORT=4000` - Internal Rust port
- `VITE_API_PORT=4000` - Tells Vite where backend is

In `data/settings.yaml`:
- `port: 3001` - Default, but overridden by env var to 4000 internally

## Important Notes

1. **Never use port 3090** - This was a typo in the refactor
2. **Settings.yaml should have `port: 3001`** as default
3. **Docker compose overrides it to 4000** for internal use
4. **Nginx always listens on 3001** for external access

## Common Issues

### 502 Bad Gateway
- Means the Rust backend on port 4000 isn't running
- Check if Docker container is up
- Verify `SYSTEM_NETWORK_PORT=4000` is set

### WebSocket Connection Failed  
- WebSocket proxied through Nginx from /wss to ws://backend:4000
- If failing, backend isn't running

## Correct Startup

```bash
# From /workspace/ext directory
docker-compose -f docker-compose.dev.yml up --build
```

Then access at: `http://localhost:3001`

## Never Do This

```bash
# WRONG - Don't expose internal ports directly
docker run -p 3001:3090  # 3090 doesn't exist!
docker run -p 4000:4000  # Bypasses Nginx!
```

## The Refactor

The new unified settings correctly uses:
- `port: 3001` in settings-unified.yaml
- Matches the original configuration
- Ready for deployment when needed
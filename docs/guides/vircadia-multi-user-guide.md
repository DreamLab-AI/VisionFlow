# Vircadia Multi-User XR Integration - User Guide

## Overview

VisionFlow now supports **multi-user collaborative visualization** through Vircadia World Server integration. Multiple users can simultaneously view and interact with the same agent swarm and knowledge graph in real-time, with full XR support for Meta Quest 3.

## Features

✨ **Real-Time Collaboration**
- See other users' avatar positions
- View teammate selections and annotations
- Synchronized graph exploration
- Shared filter states

🎧 **Spatial Audio**
- 3D positional voice communication
- Distance-based audio attenuation
- Proximity-based audio zones
- Works in both desktop and VR modes

🤖 **Agent Swarm Synchronization**
- All users see the same agent positions
- Multi-user agent monitoring
- Collaborative swarm analysis
- Real-time position updates (100ms)

📊 **Graph Collaboration**
- Shared node selections with highlighting
- Multi-user annotations
- Collaborative filtering
- User presence indicators

---

## Quick Start

### 1. Start Vircadia Server

```bash
cd /workspace/ext

# Start VisionFlow with Vircadia enabled
docker-compose -f docker-compose.yml -f docker-compose.vircadia.yml --profile dev up -d

# Verify Vircadia server is running
docker ps | grep vircadia-world-server

# Check server logs
docker logs vircadia-world-server
```

**Expected Output:**
```
Vircadia World Server v1.0.0
Listening on ws://0.0.0.0:3020/world/ws
Connected to PostgreSQL: vircadia_world
Default world initialized: VisionFlow World
```

### 2. Enable Multi-User Mode in VisionFlow

1. Open VisionFlow: `http://localhost:3030`
2. Navigate to **Settings** → **Multi-User XR**
3. Toggle **"Enable Multi-User Mode"** to ON
4. Server URL should auto-populate: `ws://vircadia-world-server:3020/world/ws`
5. Click **"Connect"**

**Connection Status:**
- 🟢 **Connected** - You're in the shared world!
- 🟡 **Connecting...** - Wait a moment
- 🔴 **Error** - Check Docker logs or server URL

### 3. Invite Teammates

Share your VisionFlow URL with teammates:
```
http://your-server-ip:3030
```

Each user should:
1. Open VisionFlow
2. Enable Multi-User Mode in Settings
3. Connect to the same Vircadia server

**You'll see:**
- 👤 Active users list in Settings panel
- 🎨 Colored avatars representing each user
- 📍 Real-time position indicators

---

## Using Multi-User Features

### Agent Swarm Collaboration

**Automatic Synchronization:**
- Agent positions sync automatically across all users
- No manual action required
- 100ms update interval for smooth movement

**What You See:**
- Your view of agents matches all other users
- Agent health, status, and metadata visible to everyone
- Communication links (edges) synchronized

**Use Cases:**
- **Team Monitoring:** Multiple team members watch agent swarm together
- **Debugging Sessions:** Point out problematic agents to teammates
- **Live Demos:** Show agent behavior to stakeholders in real-time

### Graph Collaboration

**Selecting Nodes:**
1. Click any graph node
2. Your selection is highlighted in your color
3. Other users see your highlight instantly
4. Each user has a unique color

**Adding Annotations:**
1. Right-click a node → "Add Annotation"
2. Type your note (e.g., "Check this connection")
3. Annotation appears for all users
4. Includes your username and timestamp

**Shared Filtering:**
1. Apply filters (category, time range, search)
2. Other users see your active filters
3. Optionally adopt teammates' filter states

### Spatial Audio Communication

**In VR Mode (Quest 3):**
- Walk close to teammates to hear them clearly
- Voice fades with distance
- Natural 3D audio positioning
- Audio zones prevent crosstalk

**In Desktop Mode:**
- Audio still works with virtual positioning
- Volume based on camera proximity
- Less immersive but still functional

**Best Practices:**
- Stay within 10m for clear communication
- Use proximity audio for private conversations
- Move to different areas for team sub-groups

---

## Settings Reference

### Multi-User XR Settings Panel

**Enable Multi-User Mode**
- Toggle ON to connect to Vircadia
- Toggle OFF to work solo

**Vircadia Server URL**
- Default: `ws://vircadia-world-server:3020/world/ws` (Docker network)
- External: `ws://your-server-ip:3020/world/ws` (outside Docker)
- Must be accessible from all users' machines

**Auto-Connect**
- When enabled, automatically connects on app load
- Useful for dedicated collaboration sessions

### Connection Status

**Connected For:** Shows connection duration
**Agent ID:** Your unique ID in the Vircadia world
**Session ID:** Current session identifier
**Pending Requests:** WebSocket query queue size

### Active Users Panel

Shows all connected users:
- 🟢 Green dot = online
- Username display
- Number of nodes selected
- Click to focus camera on their position (coming soon)

### Synchronization Status

**Bots Bridge:** ✓ Active when agent sync is working
**Graph Bridge:** ✓ Active when graph sync is working

---

## Troubleshooting

### "Connection Failed" Error

**Check 1: Is Vircadia server running?**
```bash
docker ps | grep vircadia-world-server
```
If not running:
```bash
docker-compose -f docker-compose.yml -f docker-compose.vircadia.yml up -d vircadia-world-server
```

**Check 2: Network connectivity**
```bash
# From VisionFlow container
docker exec -it visionflow_container curl -v ws://vircadia-world-server:3020/world/ws
```

**Check 3: Server logs**
```bash
docker logs vircadia-world-server --tail 50
```

Common issues:
- PostgreSQL not running → Start postgres service
- Port 3020 not exposed → Check docker-compose.vircadia.yml
- Network misconfiguration → Verify `docker_ragflow` network

### "Bridge Initialization Failed"

**Symptoms:** Connection succeeds but bridges show "Initializing..."

**Solution:**
1. Check browser console for errors (F12 → Console tab)
2. Verify Babylon.js scene is created (for Graph Bridge)
3. Restart VisionFlow client
4. Check for TypeScript errors in bridge services

### Agents Not Syncing

**Check BotsDataContext:**
1. Open DevTools → Console
2. Search for "BotsVircadiaBridge" logs
3. Verify agents array is populated

**Force Resync:**
```javascript
// In browser console
window.__vircadiaBridges?.syncAgentsToVircadia(agents, edges);
```

### No Spatial Audio

**Requirements:**
- Microphone permission granted
- Audio context started (browsers require user interaction)
- WebRTC supported by browser

**Enable Audio:**
1. Click anywhere in the app (activates audio context)
2. Grant microphone permission when prompted
3. Check browser audio settings

---

## Advanced Usage

### Programmatic Control

**Access Bridges in Code:**
```typescript
import { useVircadiaBridges } from './contexts/VircadiaBridgesContext';

const MyComponent = () => {
  const {
    syncAgentsToVircadia,
    broadcastSelection,
    addAnnotation,
    activeUsers
  } = useVircadiaBridges();

  // Manually sync agents
  const handleSync = () => {
    syncAgentsToVircadia(agents, edges);
  };

  // Broadcast selection
  const handleSelect = (nodeIds: string[]) => {
    broadcastSelection(nodeIds);
  };

  // Add annotation
  const handleAnnotate = async () => {
    const id = await addAnnotation(
      'node-123',
      'Interesting connection',
      { x: 0, y: 1, z: 0 }
    );
  };

  return <div>...</div>;
};
```

### Custom Configuration

**Adjust Sync Intervals:**
```typescript
const bridge = new BotsVircadiaBridge(client, entitySync, avatars, {
  syncPositions: true,
  syncMetadata: true,
  syncEdges: true,
  updateInterval: 50, // 50ms for ultra-smooth (high CPU)
  enableAvatars: true
});
```

### External Vircadia Server

**Connect to External Server:**
1. Deploy Vircadia World Server on separate machine
2. Expose port 3020 to network
3. Update Server URL in Settings:
   ```
   ws://your-vircadia-server.com:3020/world/ws
   ```
4. Ensure firewall allows WebSocket connections

---

## Performance Optimization

### Network Bandwidth

**Reduce Data Usage:**
- Disable edge sync if not needed: `syncEdges: false`
- Increase update interval: `updateInterval: 200` (from 100ms)
- Disable metadata sync: `syncMetadata: false`

### Client Performance

**Lower Visual Quality:**
1. Settings → Graphics → Quality: Medium
2. Reduce agent count display limit
3. Disable particle effects

**Optimize Updates:**
- Smart polling reduces REST requests during idle
- Binary position updates (34 bytes) minimize bandwidth
- Delta compression skips unchanged data

### Server Scaling

**Handle More Users:**
```yaml
# docker-compose.vircadia.yml
environment:
  - MAX_USERS_PER_WORLD=100  # Increase from 50
  - ENTITY_SYNC_INTERVAL=100  # Lower for smoother sync
```

**Add More Servers:**
- Deploy multiple Vircadia servers
- Load balance with Nginx
- Shard users by team/project

---

## Security Considerations

### Authentication

**Nostr Integration:**
```typescript
// VircadiaContext automatically uses Nostr auth
const { client } = useVircadia();
// authToken comes from Nostr extension
```

**Manual Token:**
```bash
# Set custom auth token
export VIRCADIA_AUTH_TOKEN=your_secure_token_here
```

### Data Privacy

**What's Shared:**
- Agent positions and metadata
- Node selections and annotations
- User presence and activity

**What's NOT Shared:**
- API keys or credentials
- Local file system data
- Private settings or preferences

### Network Security

**Recommended Setup:**
- Use TLS/WSS in production: `wss://your-server.com:3020/world/ws`
- Enable JWT authentication
- Firewall Vircadia port to trusted IPs only
- Use VPN for team collaboration

---

## FAQ

**Q: How many users can connect simultaneously?**
A: Default limit is 50 users per world. Configurable in `docker-compose.vircadia.yml`.

**Q: Does it work without VR?**
A: Yes! Desktop mode fully supported with mouse/keyboard.

**Q: Can I use my own Vircadia server?**
A: Absolutely. Just point the Server URL to your deployment.

**Q: Is audio required?**
A: No, audio is optional. Visual collaboration works without it.

**Q: What if users have different agent counts?**
A: Sync handles mismatches gracefully. All users see the union of visible agents.

**Q: Can I disable multi-user temporarily?**
A: Yes, toggle OFF in Settings to work solo without disconnecting.

**Q: Does it work with existing Vircadia accounts?**
A: Not yet. Currently uses anonymous sessions with auto-generated IDs.

---

## Next Steps

- **[Vircadia Integration Analysis](../architecture/vircadia-integration-analysis.md)** - Technical architecture
- **[WebRTC Migration Plan](../architecture/voice-webrtc-migration-plan.md)** - Voice system upgrade path
- **[Multi-Agent Docker Environment](../reference/architecture/README.md)** - Overall system architecture

---

**Need Help?**
- Check Docker logs: `docker logs vircadia-world-server`
- Review browser console: F12 → Console
- Open GitHub issue with logs and screenshots

**Happy Collaborating! 🚀**

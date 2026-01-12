# VisionFlow Database Locations - FOUND!

## ✅ Database Files Located

### Primary Data Databases
```
/app/data/settings.db          - Application settings
/app/data/ontology.db          - Ontology classes, properties, individuals
/app/data/knowledge_graph.db   - Graph nodes, edges, relationships
```

### Swarm Memory Databases (Claude Flow)
```
/app/client/.swarm/memory.db
/app/src/utils/.swarm/memory.db
/app/src/handlers/.swarm/memory.db
```

## Backend Architecture Discovery

### Running Process
**Vite Dev Server** (Node.js):
```
PID: 108
Command: node /app/client/node_modules/.bin/vite
CPU: 0.5%
Memory: 232MB
```

**ESBuild Worker**:
```
PID: 119
Command: /app/client/node_modules/@esbuild/linux-x64/bin/esbuild
```

### Critical Finding
**There is NO separate backend server process!** The Vite dev server is handling the API endpoints.

This means:
1. Backend is integrated into Vite dev server
2. API routes defined in Vite config or plugins
3. Database access happens via Vite server-side code
4. `/app/backend/` directory might not contain a running server

## Implications for Bug Investigation

### Why Some Endpoints Work
- `/api/health` - Simple static response
- `/api/config` - Reads config files or basic DB queries

### Why Some Endpoints Crash (Exit 52)
- `/api/settings` - Complex DB query crashes Vite server
- `/api/ontology/classes` - Complex query crashes
- `/api/ontology/properties` - Complex query crashes

**Crash Pattern**: When Vite's API handler throws unhandled error → connection closes immediately

### Why One Endpoint Hangs (Exit 28)
- `/api/graph/data` - Expensive query blocks event loop
- Vite dev server is single-threaded
- Long query = entire server hangs

## Next Steps for Debugging Agent

1. **Find Vite API Configuration**:
   ```bash
   cat /app/client/vite.config.ts
   # or
   cat /app/client/vite.config.js
   ```

2. **Find API Handler Code**:
   ```bash
   find /app -name "*api*" -type f | grep -E "\.(ts|js)$"
   find /app/client -name "*server*" -type f
   ```

3. **Check Database Query Code**:
   - Look for better-sqlite3 or sqlite3 imports
   - Find query execution for `/settings`
   - Find query execution for `/ontology/classes`
   - Compare with working `/config` queries

4. **Install SQLite Tools** (optional):
   ```bash
   docker exec visionflow_container apk add sqlite
   # or
   docker exec visionflow_container apt-get install -y sqlite3
   ```

5. **Test Direct Database Access**:
   ```bash
   docker exec visionflow_container sqlite3 /app/data/settings.db "SELECT * FROM sqlite_master WHERE type='table';"
   ```

## Database Schema Validation Needed

For each crashing endpoint, verify:
1. Does the table exist?
2. Does the schema match code expectations?
3. Are there foreign key constraints?
4. Is there data in the table?

### Example Checks
```bash
# Settings table
docker exec visionflow_container sqlite3 /app/data/settings.db ".schema"

# Ontology classes
docker exec visionflow_container sqlite3 /app/data/ontology.db ".tables"
docker exec visionflow_container sqlite3 /app/data/ontology.db "SELECT COUNT(*) FROM classes;"

# Check for errors
docker exec visionflow_container sqlite3 /app/data/ontology.db "PRAGMA integrity_check;"
```

## Hypothesis: Unhandled Promise Rejection

Since Vite crashes on specific endpoints, likely pattern:
```javascript
// In Vite config or plugin
server: {
  middlewares: [
    (req, res, next) => {
      if (req.url === '/api/settings') {
        // This probably throws an error
        const db = new Database('/app/data/settings.db');
        const result = db.prepare('SELECT * FROM settings').all();
        // ^ If table doesn't exist or query is malformed → unhandled error
        res.json(result);
      }
    }
  ]
}
```

**Why it crashes**:
- Vite dev server doesn't have global error handler for DB queries
- Unhandled promise rejection → process.exit()
- Connection closes before HTTP response sent

**Why `/api/config` works**:
- Simpler query that doesn't throw
- Or has try/catch error handling
- Or reads from file instead of DB

---

**Status**: Database locations confirmed ✅
**Backend type**: Vite dev server (Node.js) ✅
**Next agent**: Debugging agent should examine Vite config and API handlers

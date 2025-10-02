# Vircadia World Integration

Vircadia spatial computing platform integrated with AR/AI Knowledge Graph for immersive data visualization and collaboration.

## Quick Start

### Prerequisites
- Docker installed and running
- Ports 5432 (PostgreSQL) and 5437 (PGWeb) available

### Start Services

```bash
./start-vircadia.sh
```

This will:
- Start PostgreSQL database on port 5432
- Start PGWeb UI on port 5437
- Connect services to ragflow network for integration

### Access Points

- **PGWeb UI**: http://localhost:5437 (Database management interface)
- **PostgreSQL**: localhost:5432
  - Database: `vircadia_world`
  - User: `postgres`
  - Password: `vircadia_password`

## Architecture

```
┌─────────────────────────────────────────────────┐
│              VisionFlow Client                  │
│  (React + Babylon.js + Quest 3 WebXR)          │
└──────────────────┬──────────────────────────────┘
                   │
                   ├─────► API Manager (port 3020)
                   │       - REST API endpoints
                   │       - Entity management
                   │
                   ├─────► State Manager (port 3021)
                   │       - Real-time state sync
                   │       - WebSocket connections
                   │
                   └─────► PostgreSQL (port 5432)
                           - Spatial data storage
                           - Knowledge graph integration

                   Network: docker_ragflow
                   - Shared with RAGFlow services
                   - Service discovery enabled
```

## Services

### PostgreSQL Database
- **Container**: `vircadia_world_postgres`
- **Image**: postgres:17.5-alpine3.21
- **Port**: 5432
- **Features**:
  - Logical replication enabled (wal_level=logical)
  - Optimized WAL settings for spatial data
  - Extensions: uuid-ossp, hstore, pgcrypto

### PGWeb UI
- **Container**: `vircadia_world_pgweb`
- **Image**: sosedoff/pgweb:0.16.2
- **Port**: 5437
- **Purpose**: Web-based database management and query interface

### API Manager (Planned)
- **Port**: 3020
- **Purpose**: RESTful API for entity management and spatial queries

### State Manager (Planned)
- **Port**: 3021
- **Purpose**: Real-time state synchronization via WebSocket

## Configuration

Environment variables are configured in `.env`:

```bash
# Database
VRCA_SERVER_SERVICE_POSTGRES_DATABASE=vircadia_world
VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_USERNAME=postgres
VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD=vircadia_password

# PGWeb UI
VRCA_SERVER_SERVICE_PGWEB_PORT_CONTAINER_BIND_EXTERNAL=5437

# API Manager
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_PORT_PUBLIC_AVAILABLE_AT=3020

# State Manager
VRCA_SERVER_SERVICE_WORLD_STATE_MANAGER_PORT_CONTAINER_BIND_EXTERNAL=3021
```

## Development

### Project Structure

```
vircadia/
├── server/vircadia-world/          # Vircadia world server
│   ├── server/service/             # Service configurations
│   │   └── server.docker.compose.yml
│   ├── sdk/                        # TypeScript SDK
│   └── cli/                        # CLI tools
├── client/vircadia-web/            # Web client (Babylon.js)
├── .env                            # Environment configuration
├── start-vircadia.sh               # Quick start script
└── README.md                       # This file
```

### Database Management

Access PGWeb UI at http://localhost:5437 to:
- Execute SQL queries
- Browse table structure
- View spatial data
- Export/import data

### Stopping Services

```bash
cd server/vircadia-world/server/service
sudo docker-compose -f server.docker.compose.yml down
```

## Integration with RAGFlow

Vircadia services are connected to the `docker_ragflow` network, enabling:
- Direct communication with RAGFlow knowledge base
- Shared service discovery
- Unified data pipeline for AR visualization

## Next Steps

1. ✅ PostgreSQL configured on ragflow network
2. ✅ PGWeb UI accessible
3. 🔄 Install @vircadia/web-sdk in VisionFlow client
4. 🔄 Create VircadiaContext.tsx React provider
5. 🔄 Implement GraphEntityMapper service
6. 🔄 Test Quest 3 WebXR connection

## Resources

- [Vircadia Documentation](https://docs.vircadia.com)
- [Babylon.js WebXR](https://doc.babylonjs.com/features/featuresDeepDive/webXR)
- [Meta Quest 3 WebXR](https://developer.oculus.com/webxr/)

## License

See LICENSE file in project root.

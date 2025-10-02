#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Vircadia World Server...${NC}"

# Check if Docker is running
if ! sudo docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✓ Environment variables loaded${NC}"
else
    echo -e "${YELLOW}Warning: .env file not found, using defaults${NC}"
fi

# Check if containers are already running
if sudo docker ps | grep -q vircadia_world_postgres; then
    echo -e "${GREEN}✓ PostgreSQL is already running${NC}"
else
    echo -e "${BLUE}Starting PostgreSQL...${NC}"
    cd server/vircadia-world/server/service
    sudo docker-compose -f server.docker.compose.yml up -d vircadia_world_postgres
    cd - > /dev/null
    echo -e "${GREEN}✓ PostgreSQL started${NC}"
fi

if sudo docker ps | grep -q vircadia_world_pgweb; then
    echo -e "${GREEN}✓ PGWeb is already running${NC}"
else
    echo -e "${BLUE}Starting PGWeb UI...${NC}"
    cd server/vircadia-world/server/service
    sudo docker-compose -f server.docker.compose.yml up -d vircadia_world_pgweb
    cd - > /dev/null
    echo -e "${GREEN}✓ PGWeb started${NC}"
fi

# Ensure both containers are on ragflow network
echo -e "${BLUE}Connecting to ragflow network...${NC}"
sudo docker network connect docker_ragflow vircadia_world_postgres 2>/dev/null || echo -e "${GREEN}✓ PostgreSQL already on ragflow network${NC}"
sudo docker network connect docker_ragflow vircadia_world_pgweb 2>/dev/null || echo -e "${GREEN}✓ PGWeb already on ragflow network${NC}"

echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Vircadia World Server is ready!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""
echo -e "📊 PGWeb UI:      ${BLUE}http://localhost:5437${NC}"
echo -e "🗄️  PostgreSQL:    ${BLUE}localhost:5432${NC}"
echo -e "   Database:      ${BLUE}vircadia_world${NC}"
echo -e "   User:          ${BLUE}postgres${NC}"
echo ""
echo -e "${YELLOW}To stop services: sudo docker-compose -f server/vircadia-world/server/service/server.docker.compose.yml down${NC}"

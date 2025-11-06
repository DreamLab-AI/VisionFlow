#!/bin/bash
# Check if Neo4j is available in the docker_ragflow network

echo "=== Checking for Neo4j in docker_ragflow network ==="
echo ""

# Check if docker_ragflow network exists
if docker network inspect docker_ragflow &>/dev/null; then
    echo "✓ docker_ragflow network exists"
    echo ""

    # List all containers in the network
    echo "Containers in docker_ragflow network:"
    docker network inspect docker_ragflow -f '{{range .Containers}}{{.Name}} ({{.IPv4Address}}){{"\n"}}{{end}}'
    echo ""

    # Check for Neo4j specifically
    NEO4J_CONTAINER=$(docker ps --filter "network=docker_ragflow" --format "{{.Names}}" | grep -i neo4j)

    if [ -n "$NEO4J_CONTAINER" ]; then
        echo "✓ Neo4j container found: $NEO4J_CONTAINER"
        echo ""

        # Get container details
        echo "Neo4j container details:"
        docker inspect "$NEO4J_CONTAINER" --format '
Container: {{.Name}}
Image: {{.Config.Image}}
Status: {{.State.Status}}
Ports: {{range $p, $conf := .NetworkSettings.Ports}}{{$p}} {{end}}
Network IP: {{(index .NetworkSettings.Networks "docker_ragflow").IPAddress}}'
        echo ""

        # Check if Neo4j is responding
        NEO4J_IP=$(docker inspect "$NEO4J_CONTAINER" --format '{{(index .NetworkSettings.Networks "docker_ragflow").IPAddress}}')
        echo "Testing Neo4j connectivity at bolt://$NEO4J_IP:7687..."

        # Test connection (requires netcat)
        if timeout 2 bash -c "echo > /dev/tcp/$NEO4J_IP/7687" 2>/dev/null; then
            echo "✓ Neo4j bolt port (7687) is accessible"
        else
            echo "✗ Neo4j bolt port (7687) is NOT accessible"
        fi

    else
        echo "✗ No Neo4j container found in docker_ragflow network"
        echo ""
        echo "You need to add Neo4j to the network."
    fi

else
    echo "✗ docker_ragflow network does NOT exist"
    echo "Create it with: docker network create docker_ragflow"
fi

echo ""
echo "=== Check complete ==="

#!/bin/bash

echo "Testing swarm creation via MCP..."

# Create a swarm
request='{"jsonrpc":"2.0","id":"test-swarm","method":"tools/call","params":{"name":"swarm_init","arguments":{"topology":"mesh","maxAgents":4,"strategy":"balanced"}}}'

echo "$request" | nc -w 5 localhost 9500 > /tmp/swarm_response.txt 2>&1

echo "Response:"
cat /tmp/swarm_response.txt | head -5

# Now list agents to see if the swarm was created
echo ""
echo "Listing agents..."
request2='{"jsonrpc":"2.0","id":"test-list","method":"tools/call","params":{"name":"agent_list","arguments":{}}}'

echo "$request2" | nc -w 5 localhost 9500 > /tmp/agent_list.txt 2>&1

echo "Agent list:"
cat /tmp/agent_list.txt | head -5

echo "Done."
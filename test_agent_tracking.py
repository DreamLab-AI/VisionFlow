#!/usr/bin/env python3
"""
Test MCP Agent Tracking System
Tests the complete flow from swarm creation to agent listing
"""

import socket
import json
import time
import sys

def send_mcp_request(host, port, request):
    """Send JSON-RPC request to MCP server and return response"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall((json.dumps(request) + '\n').encode())
    
    response = b''
    s.settimeout(3)
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            response += chunk
    except socket.timeout:
        pass
    
    s.close()
    return response.decode()

def main():
    print("="*60)
    print("MCP AGENT TRACKING SYSTEM TEST")
    print("="*60)
    print()

    # 1. Initialize connection
    print("Step 1: Initializing MCP connection...")
    init_req = {
        "jsonrpc": "2.0",
        "id": "init",
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0.0",
            "capabilities": {"roots": True, "sampling": True, "tools": True},
            "clientInfo": {"name": "test", "version": "1.0.0"}
        }
    }
    resp = send_mcp_request('localhost', 9500, init_req)
    if '"result"' in resp:
        print("✅ MCP connection initialized")
    else:
        print("❌ Failed to initialize")
        return 1
    print()

    # 2. Create swarm
    print("Step 2: Creating agent swarm...")
    swarm_req = {
        "jsonrpc": "2.0",
        "id": "swarm1",
        "method": "tools/call",
        "params": {
            "name": "swarm_init",
            "arguments": {
                "topology": "mesh",
                "maxAgents": 8,
                "strategy": "balanced"
            }
        }
    }
    resp = send_mcp_request('localhost', 9500, swarm_req)
    
    swarm_id = None
    for line in resp.split('\n'):
        if '"id":"swarm1"' in line and 'result' in line:
            try:
                j = json.loads(line)
                if 'result' in j and 'content' in j['result']:
                    text = json.loads(j['result']['content'][0]['text'])
                    swarm_id = text['swarmId']
                    print(f"✅ Swarm created: {swarm_id}")
                    break
            except Exception as e:
                print(f"Parse error: {e}")
    
    if not swarm_id:
        print("❌ Failed to create swarm")
        return 1
    print()

    # 3. Spawn multiple agents
    print("Step 3: Spawning agents...")
    agent_types = ["coordinator", "researcher", "coder", "tester", "reviewer"]
    spawned_agents = []
    
    for i, agent_type in enumerate(agent_types):
        spawn_req = {
            "jsonrpc": "2.0",
            "id": f"spawn{i}",
            "method": "tools/call",
            "params": {
                "name": "agent_spawn",
                "arguments": {
                    "type": agent_type,
                    "swarmId": swarm_id
                }
            }
        }
        resp = send_mcp_request('localhost', 9500, spawn_req)
        
        for line in resp.split('\n'):
            if f'"id":"spawn{i}"' in line and 'result' in line:
                try:
                    j = json.loads(line)
                    if 'result' in j and 'content' in j['result']:
                        text = json.loads(j['result']['content'][0]['text'])
                        agent_id = text['agentId']
                        spawned_agents.append({
                            'id': agent_id,
                            'type': agent_type
                        })
                        print(f"  ✅ Spawned {agent_type}: {agent_id}")
                except Exception as e:
                    print(f"  ❌ Failed to spawn {agent_type}: {e}")
    
    print(f"\nTotal spawned: {len(spawned_agents)} agents")
    print()

    # 4. List agents
    print("Step 4: Retrieving agent list...")
    time.sleep(1)  # Give the system a moment to process
    
    list_req = {
        "jsonrpc": "2.0",
        "id": "list",
        "method": "tools/call",
        "params": {
            "name": "agent_list",
            "arguments": {
                "swarmId": swarm_id,
                "filter": "all"
            }
        }
    }
    resp = send_mcp_request('localhost', 9500, list_req)
    
    for line in resp.split('\n'):
        if '"id":"list"' in line and 'result' in line:
            try:
                j = json.loads(line)
                if 'result' in j and 'content' in j['result']:
                    text = json.loads(j['result']['content'][0]['text'])
                    returned_agents = text.get('agents', [])
                    
                    print("\n" + "="*60)
                    print("RESULTS")
                    print("="*60)
                    print(f"Swarm ID: {text.get('swarmId', 'unknown')}")
                    print(f"Agent Count: {text.get('count', 0)}")
                    print()
                    
                    if returned_agents:
                        print("Returned agents:")
                        is_mock = False
                        for agent in returned_agents:
                            print(f"  • {agent['type']}: {agent['id']} ({agent['status']})")
                            if agent['id'].startswith('agent-'):
                                is_mock = True
                        
                        print("\n" + "-"*60)
                        print("VERDICT:")
                        
                        if is_mock:
                            print("❌ FAILURE: Still returning mock data!")
                            print("   Mock agent IDs detected (agent-1, agent-2, etc.)")
                            return 1
                        else:
                            # Compare spawned vs returned
                            spawned_ids = set(a['id'] for a in spawned_agents)
                            returned_ids = set(a['id'] for a in returned_agents)
                            
                            if spawned_ids == returned_ids:
                                print("✅ SUCCESS! Agent tracking is working perfectly!")
                                print(f"   All {len(spawned_agents)} spawned agents are tracked correctly")
                                return 0
                            elif returned_ids.issubset(spawned_ids):
                                print("⚠️  PARTIAL SUCCESS: Some agents are tracked")
                                print(f"   Spawned: {len(spawned_ids)} agents")
                                print(f"   Tracked: {len(returned_ids)} agents")
                                missing = spawned_ids - returned_ids
                                print(f"   Missing: {missing}")
                                return 1
                            else:
                                print("❌ UNEXPECTED: Unknown agents in response")
                                extra = returned_ids - spawned_ids
                                print(f"   Extra agents: {extra}")
                                return 1
                    else:
                        print("❌ No agents returned (empty list)")
                        print("   Agent tracker may not be storing agents properly")
                        return 1
                        
            except Exception as e:
                print(f"Error parsing response: {e}")
                return 1
    
    print("\n" + "="*60)
    print("Test failed to get agent list response")
    return 1

if __name__ == "__main__":
    sys.exit(main())
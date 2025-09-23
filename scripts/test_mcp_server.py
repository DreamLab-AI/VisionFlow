#!/usr/bin/env python3
"""
Simple MCP Test Server

This creates a minimal MCP-compatible TCP server for testing the Rust client.
It responds to basic JSON-RPC requests with mock agent data.
"""

import json
import socket
import threading
import time
import sys
from typing import Dict, Any, List

class McpTestServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9500):
        self.host = host
        self.port = port
        self.running = False

        # Mock agent data
        self.agents = [
            {
                "id": "agent_001",
                "name": "Coordinator Agent",
                "type": "coordinator",
                "status": "active",
                "swarm_id": "test_swarm",
                "capabilities": ["coordination", "task_management"],
                "performance": {
                    "cpu_usage": 25.5,
                    "memory_usage": 42.3,
                    "health_score": 95.0,
                    "activity_level": 75.0,
                    "tasks_active": 3,
                    "tasks_completed": 127,
                    "tasks_failed": 2,
                    "success_rate": 98.4,
                    "token_usage": 15420,
                    "token_rate": 12.5,
                    "response_time_ms": 145.2,
                    "throughput": 8.3
                },
                "metadata": {
                    "session_id": "session_001",
                    "task_queue_size": 2,
                    "error_count": 0,
                    "warning_count": 1,
                    "tags": ["production", "critical"]
                },
                "neural": {
                    "model_type": "transformer",
                    "model_size": "large",
                    "training_status": "active",
                    "cognitive_pattern": "analytical",
                    "learning_rate": 0.001,
                    "adaptation_score": 0.82,
                    "memory_capacity": 2048000,
                    "knowledge_domains": ["coordination", "planning"]
                }
            },
            {
                "id": "agent_002",
                "name": "Research Agent",
                "type": "researcher",
                "status": "active",
                "swarm_id": "test_swarm",
                "capabilities": ["research", "analysis", "data_gathering"],
                "performance": {
                    "cpu_usage": 15.2,
                    "memory_usage": 35.8,
                    "health_score": 89.5,
                    "activity_level": 60.0,
                    "tasks_active": 1,
                    "tasks_completed": 89,
                    "tasks_failed": 1,
                    "success_rate": 98.9,
                    "token_usage": 8940,
                    "token_rate": 7.2,
                    "response_time_ms": 289.1,
                    "throughput": 5.1
                },
                "metadata": {
                    "session_id": "session_002",
                    "task_queue_size": 1,
                    "error_count": 0,
                    "warning_count": 0,
                    "tags": ["research", "analysis"]
                }
            },
            {
                "id": "agent_003",
                "name": "Coder Agent",
                "type": "coder",
                "status": "active",
                "swarm_id": "test_swarm",
                "capabilities": ["coding", "debugging", "optimization"],
                "performance": {
                    "cpu_usage": 45.7,
                    "memory_usage": 67.2,
                    "health_score": 92.1,
                    "activity_level": 85.0,
                    "tasks_active": 5,
                    "tasks_completed": 203,
                    "tasks_failed": 8,
                    "success_rate": 96.2,
                    "token_usage": 32100,
                    "token_rate": 18.4,
                    "response_time_ms": 156.8,
                    "throughput": 12.7
                },
                "metadata": {
                    "session_id": "session_003",
                    "task_queue_size": 3,
                    "error_count": 1,
                    "warning_count": 2,
                    "tags": ["development", "coding"]
                }
            }
        ]

        self.server_info = {
            "server_id": "test_mcp_server",
            "server_type": "claude-flow",
            "supported_tools": [
                "agent_list",
                "swarm_status",
                "server_info",
                "initialize"
            ],
            "agent_count": len(self.agents)
        }

        self.swarm_topology = {
            "topology_type": "hierarchical",
            "total_agents": len(self.agents),
            "coordination_layers": 2,
            "efficiency_score": 0.87
        }

    def handle_request(self, request_data: str) -> str:
        """Handle incoming JSON-RPC request"""
        try:
            request = json.loads(request_data.strip())
            method = request.get("method", "")
            params = request.get("params", {})
            request_id = request.get("id", 1)

            print(f"Received request: {method} with params: {params}")

            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "logging": {},
                        "tools": {
                            "listChanged": True
                        }
                    },
                    "serverInfo": {
                        "name": "test-mcp-server",
                        "version": "1.0.0"
                    }
                }
            elif method == "agent_list":
                result = {
                    "agents": self.agents
                }
            elif method == "server_info":
                result = self.server_info
            elif method == "swarm_status":
                result = self.swarm_topology
            else:
                # Return error for unknown methods
                return json.dumps({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    },
                    "id": request_id
                }) + "\n"

            response = {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }

            return json.dumps(response) + "\n"

        except Exception as e:
            print(f"Error handling request: {e}")
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                },
                "id": 1
            }) + "\n"

    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle individual client connection"""
        print(f"Client connected from {address}")

        try:
            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    break

                request_data = data.decode('utf-8')
                print(f"Received: {request_data.strip()}")

                response = self.handle_request(request_data)
                client_socket.send(response.encode('utf-8'))
                print(f"Sent: {response.strip()}")

        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"Client {address} disconnected")

    def start(self):
        """Start the MCP test server"""
        self.running = True

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            print(f"MCP Test Server listening on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, address = server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")

        except Exception as e:
            print(f"Server error: {e}")
        finally:
            server_socket.close()
            print("MCP Test Server stopped")

    def stop(self):
        """Stop the server"""
        self.running = False

if __name__ == "__main__":
    port = 9500
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    server = McpTestServer(port=port)

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()
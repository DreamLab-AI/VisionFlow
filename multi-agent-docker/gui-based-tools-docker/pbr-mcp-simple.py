#!/usr/bin/env python3
"""
Simple PBR Generator MCP Server - Provides basic PBR texture generation service.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

class SimplePBRServer:
    def __init__(self, host='0.0.0.0', port=9878):
        self.host = host
        self.port = port

    async def handle_client(self, reader, writer):
        """Handle incoming client connections."""
        client_addr = writer.get_extra_info('peername')
        print(f"Client connected from {client_addr}")
        
        try:
            while True:
                # Read the JSON request line by line
                data = await reader.readline()
                if not data:
                    break
                
                try:
                    request = json.loads(data.decode('utf-8').strip())
                    print(f"Received request: {request.get('method', 'unknown')}")
                    
                    response = self.handle_request(request)
                    writer.write(json.dumps(response).encode() + b'\n')
                    await writer.drain()
                    
                except json.JSONDecodeError:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    writer.write(json.dumps(error_response).encode() + b'\n')
                    await writer.drain()
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"Client disconnected from {client_addr}")

    def handle_request(self, request):
        """Handle MCP protocol requests."""
        method = request.get('method')
        request_id = request.get('id')
        
        if method == 'initialize':
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "capabilities": {
                        "tools": {
                            "listChanged": True
                        }
                    }
                }
            }
        elif method == 'tools/list':
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "pbr_create_material",
                            "description": "Create a PBR material texture set",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "material_type": {
                                        "type": "string",
                                        "enum": ["wood", "metal", "stone", "fabric", "concrete"],
                                        "description": "Type of material to generate"
                                    },
                                    "resolution": {
                                        "type": "integer",
                                        "enum": [512, 1024, 2048, 4096],
                                        "description": "Texture resolution"
                                    },
                                    "output_dir": {
                                        "type": "string",
                                        "description": "Output directory for textures"
                                    }
                                },
                                "required": ["material_type"]
                            }
                        },
                        {
                            "name": "pbr_list_materials",
                            "description": "List available material presets",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    ]
                }
            }
        elif method == 'tools/call':
            params = request.get('params', {})
            tool_name = params.get('name')
            
            if tool_name == 'pbr_list_materials':
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "materials": ["wood", "metal", "stone", "fabric", "concrete"],
                        "description": "Available material types for PBR generation"
                    }
                }
            elif tool_name == 'pbr_create_material':
                args = params.get('arguments', {})
                material_type = args.get('material_type', 'stone')
                resolution = args.get('resolution', 1024)
                output_dir = args.get('output_dir', '/workspace/pbr_outputs')
                
                # Create output directory
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Simulate material creation
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "status": "success",
                        "material_type": material_type,
                        "resolution": resolution,
                        "output_files": {
                            "diffuse": f"{output_dir}/{material_type}_diffuse.png",
                            "normal": f"{output_dir}/{material_type}_normal.png",
                            "roughness": f"{output_dir}/{material_type}_roughness.png",
                            "metallic": f"{output_dir}/{material_type}_metallic.png",
                            "height": f"{output_dir}/{material_type}_height.png"
                        },
                        "message": f"PBR material '{material_type}' created successfully (simulated)"
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": "Method not found"
                }
            }

    async def run(self):
        """Start the server."""
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port)
        
        addr = server.sockets[0].getsockname()
        print(f'PBR Generator MCP server running on {addr[0]}:{addr[1]}')
        
        async with server:
            await server.serve_forever()

async def main():
    server = SimplePBRServer()
    await server.run()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down PBR server...")
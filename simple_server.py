#!/usr/bin/env python3
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

class GraphAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Handle CORS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        if parsed_path.path == '/api/graph/data':
            # Load the actual graph data from file
            try:
                with open('/workspace/ext/data/metadata/graph.json', 'r') as f:
                    data = json.load(f)
                
                print(f"Loaded graph data: {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges")
                
                self.wfile.write(json.dumps(data).encode())
            except Exception as e:
                print(f"Error loading graph data: {e}")
                # Return empty graph as fallback
                empty_data = {"nodes": [], "edges": [], "metadata": {}}
                self.wfile.write(json.dumps(empty_data).encode())
        else:
            # 404 for other paths
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == '__main__':
    port = 8000
    server = HTTPServer(('0.0.0.0', port), GraphAPIHandler)
    print(f"Simple graph API server running on port {port}")
    print(f"API endpoint: http://localhost:{port}/api/graph/data")
    server.serve_forever()
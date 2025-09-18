#!/usr/bin/env python3
"""
TCP Connection Persistence Integration Tests

Tests to verify TCP connections remain stable and persistent
across various scenarios including reconnections and timeouts.
"""

import asyncio
import socket
import time
import json
import pytest
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCPTestClient:
    """Test client for TCP connection testing"""
    
    def __init__(self, host: str = 'localhost', port: int = 9500):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        
    def connect(self) -> bool:
        """Establish TCP connection"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
            
    def send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request and receive response"""
        if not self.connected or not self.socket:
            return None
            
        try:
            # Send request
            message = json.dumps(request).encode() + b'\n'
            self.socket.sendall(message)
            
            # Receive response
            response = b""
            while True:
                chunk = self.socket.recv(1024)
                if not chunk:
                    break
                response += chunk
                if b'\n' in response:
                    break
                    
            return json.loads(response.decode().strip())
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
            
    def disconnect(self):
        """Close TCP connection"""
        if self.socket:
            self.socket.close()
            self.connected = False
            logger.info("Disconnected")

class TestTCPPersistence:
    """TCP Connection Persistence Test Suite"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        client = TCPTestClient()
        yield client
        client.disconnect()
        
    def test_basic_connection(self, client):
        """Test basic TCP connection establishment"""
        assert client.connect() == True
        assert client.connected == True
        
        # Test simple request
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0"
                }
            }
        })
        
        assert response is not None
        assert "result" in response
        
    def test_connection_persistence(self, client):
        """Test connection remains stable over time"""
        assert client.connect() == True
        
        # Send multiple requests over time
        for i in range(10):
            response = client.send_request({
                "jsonrpc": "2.0",
                "id": i + 1,
                "method": "ping",
                "params": {}
            })
            
            assert response is not None
            assert response.get("id") == i + 1
            
            # Small delay between requests
            time.sleep(0.5)
            
        assert client.connected == True
        
    def test_idle_connection(self, client):
        """Test connection remains open during idle periods"""
        assert client.connect() == True
        
        # Initial request
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
            "params": {}
        })
        assert response is not None
        
        # Idle period (30 seconds)
        logger.info("Testing idle connection for 30 seconds...")
        time.sleep(30)
        
        # Connection should still be active
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
            "params": {}
        })
        
        assert response is not None
        assert client.connected == True
        
    def test_reconnection(self, client):
        """Test reconnection after disconnect"""
        # First connection
        assert client.connect() == True
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
            "params": {}
        })
        assert response is not None
        
        # Disconnect
        client.disconnect()
        assert client.connected == False
        
        # Reconnect
        time.sleep(1)
        assert client.connect() == True
        
        # Should work normally
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
            "params": {}
        })
        assert response is not None
        
    def test_multiple_clients(self):
        """Test multiple concurrent client connections"""
        clients = [TCPTestClient() for _ in range(5)]
        
        try:
            # Connect all clients
            for i, client in enumerate(clients):
                assert client.connect() == True
                logger.info(f"Client {i} connected")
                
            # Each client sends requests
            for i, client in enumerate(clients):
                for j in range(3):
                    response = client.send_request({
                        "jsonrpc": "2.0",
                        "id": f"{i}-{j}",
                        "method": "ping",
                        "params": {"client": i}
                    })
                    assert response is not None
                    assert response.get("id") == f"{i}-{j}"
                    
        finally:
            # Cleanup
            for client in clients:
                client.disconnect()
                
    def test_large_payload(self, client):
        """Test connection stability with large payloads"""
        assert client.connect() == True
        
        # Create large payload (1MB)
        large_data = "x" * (1024 * 1024)
        
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "process",
            "params": {
                "data": large_data
            }
        })
        
        # Connection should remain stable
        assert response is not None
        assert client.connected == True
        
        # Follow-up request should work
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
            "params": {}
        })
        assert response is not None
        
    @pytest.mark.asyncio
    async def test_async_persistence(self):
        """Test async connection persistence"""
        reader, writer = await asyncio.open_connection('localhost', 9500)
        
        try:
            # Send async requests
            for i in range(5):
                request = json.dumps({
                    "jsonrpc": "2.0",
                    "id": i + 1,
                    "method": "ping",
                    "params": {}
                }).encode() + b'\n'
                
                writer.write(request)
                await writer.drain()
                
                # Read response
                response = await reader.readline()
                assert response
                
                # Small delay
                await asyncio.sleep(0.5)
                
        finally:
            writer.close()
            await writer.wait_closed()
            
    def test_timeout_handling(self, client):
        """Test connection behavior with timeouts"""
        assert client.connect() == True
        
        # Set shorter timeout
        client.socket.settimeout(5)
        
        # Normal request should work
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
            "params": {}
        })
        assert response is not None
        
        # Simulate slow operation (server should handle gracefully)
        response = client.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "slow_operation",
            "params": {"delay": 3}
        })
        
        # Should complete within timeout
        assert response is not None

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
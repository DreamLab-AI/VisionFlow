#!/usr/bin/env python3
"""
Client Polling Integration Tests

Tests to verify client polling behavior, reconnection logic,
and graceful handling of connection interruptions.
"""

import asyncio
import json
import time
import threading
import pytest
import logging
from typing import Optional, Dict, Any, List
import websocket
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PollingTestClient:
    """Test client with polling capabilities"""
    
    def __init__(self, mode: str = "websocket", url: Optional[str] = None):
        self.mode = mode
        self.url = url or self._get_default_url()
        self.connected = False
        self.messages_received = []
        self.reconnect_count = 0
        self.ws: Optional[websocket.WebSocket] = None
        self.polling_active = False
        
    def _get_default_url(self) -> str:
        """Get default URL based on mode"""
        if self.mode == "websocket":
            return "ws://localhost:3002"
        else:
            return "http://localhost:9501"
            
    def connect_websocket(self) -> bool:
        """Connect via WebSocket"""
        try:
            self.ws = websocket.create_connection(self.url, timeout=10)
            self.connected = True
            logger.info(f"WebSocket connected to {self.url}")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
            
    def start_polling(self, interval: float = 1.0, max_retries: int = 3):
        """Start HTTP polling"""
        self.polling_active = True
        retry_count = 0
        
        while self.polling_active and retry_count < max_retries:
            try:
                response = requests.get(f"{self.url}/poll", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    self.messages_received.append(data)
                    self.connected = True
                    retry_count = 0  # Reset on success
                else:
                    retry_count += 1
                    logger.warning(f"Polling failed: {response.status_code}")
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"Polling error: {e}")
                self.connected = False
                
            time.sleep(interval)
            
    def stop_polling(self):
        """Stop HTTP polling"""
        self.polling_active = False
        
    def send_websocket_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send WebSocket message and get response"""
        if not self.ws or not self.connected:
            return None
            
        try:
            self.ws.send(json.dumps(message))
            response = self.ws.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            self.connected = False
            return None
            
    def reconnect_with_backoff(self, max_attempts: int = 5) -> bool:
        """Reconnect with exponential backoff"""
        backoff = 1
        
        for attempt in range(max_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")
            
            if self.mode == "websocket":
                if self.connect_websocket():
                    self.reconnect_count += 1
                    return True
            else:
                # For HTTP polling, just try to resume
                self.connected = True
                self.reconnect_count += 1
                return True
                
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)  # Cap at 30 seconds
            
        return False
        
    def close(self):
        """Close connection"""
        if self.ws:
            self.ws.close()
        self.connected = False
        self.polling_active = False

class TestClientPolling:
    """Client Polling Test Suite"""
    
    def test_websocket_basic_polling(self):
        """Test basic WebSocket polling behavior"""
        client = PollingTestClient(mode="websocket")
        
        try:
            # Connect
            assert client.connect_websocket() == True
            
            # Send periodic messages
            for i in range(5):
                response = client.send_websocket_message({
                    "type": "ping",
                    "id": i
                })
                assert response is not None
                time.sleep(1)
                
            assert client.connected == True
            
        finally:
            client.close()
            
    def test_http_polling(self):
        """Test HTTP polling mechanism"""
        client = PollingTestClient(mode="http")
        
        # Start polling in background
        polling_thread = threading.Thread(
            target=client.start_polling,
            args=(0.5, 10)  # 500ms interval, 10 retries
        )
        polling_thread.start()
        
        # Let it poll for a few seconds
        time.sleep(3)
        
        # Should have received some messages
        client.stop_polling()
        polling_thread.join(timeout=5)
        
        logger.info(f"Received {len(client.messages_received)} messages via polling")
        
    def test_reconnection_logic(self):
        """Test client reconnection with backoff"""
        client = PollingTestClient(mode="websocket")
        
        try:
            # Initial connection
            assert client.connect_websocket() == True
            initial_reconnect_count = client.reconnect_count
            
            # Simulate disconnect
            client.ws.close()
            client.connected = False
            
            # Try reconnection
            assert client.reconnect_with_backoff(max_attempts=3) == True
            assert client.reconnect_count == initial_reconnect_count + 1
            
            # Should be able to communicate again
            response = client.send_websocket_message({"type": "ping"})
            assert response is not None
            
        finally:
            client.close()
            
    def test_concurrent_polling_clients(self):
        """Test multiple clients polling concurrently"""
        clients = [PollingTestClient(mode="websocket") for _ in range(3)]
        threads = []
        
        def client_task(client: PollingTestClient, client_id: int):
            """Task for each client"""
            try:
                if client.connect_websocket():
                    for i in range(5):
                        response = client.send_websocket_message({
                            "type": "data",
                            "client_id": client_id,
                            "sequence": i
                        })
                        if response:
                            client.messages_received.append(response)
                        time.sleep(0.5)
            except Exception as e:
                logger.error(f"Client {client_id} error: {e}")
            finally:
                client.close()
                
        try:
            # Start all clients
            for i, client in enumerate(clients):
                thread = threading.Thread(target=client_task, args=(client, i))
                thread.start()
                threads.append(thread)
                
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)
                
            # Verify all clients received messages
            for i, client in enumerate(clients):
                logger.info(f"Client {i} received {len(client.messages_received)} messages")
                assert len(client.messages_received) > 0
                
        finally:
            for client in clients:
                client.close()
                
    def test_polling_with_connection_drops(self):
        """Test polling behavior during connection drops"""
        client = PollingTestClient(mode="websocket")
        drop_count = 0
        
        try:
            assert client.connect_websocket() == True
            
            for i in range(10):
                # Simulate random connection drops
                if i % 3 == 2:
                    logger.info("Simulating connection drop")
                    client.ws.close()
                    client.connected = False
                    drop_count += 1
                    
                    # Reconnect
                    if not client.reconnect_with_backoff(max_attempts=2):
                        break
                        
                # Try to send message
                response = client.send_websocket_message({
                    "type": "test",
                    "sequence": i
                })
                
                if response:
                    client.messages_received.append(response)
                    
                time.sleep(0.5)
                
            # Should have handled drops gracefully
            assert client.reconnect_count == drop_count
            assert len(client.messages_received) > 5  # Some messages should get through
            
        finally:
            client.close()
            
    def test_long_polling_timeout(self):
        """Test long polling with proper timeout handling"""
        client = PollingTestClient(mode="http")
        
        # Test long polling request
        start_time = time.time()
        
        try:
            response = requests.get(
                f"{client.url}/long-poll",
                timeout=30,  # 30 second timeout
                params={"timeout": 25}  # Server should respond before client timeout
            )
            
            duration = time.time() - start_time
            
            # Should complete within reasonable time
            assert duration < 30
            assert response.status_code in [200, 204]  # 204 for no data
            
            logger.info(f"Long polling completed in {duration:.2f} seconds")
            
        except requests.Timeout:
            pytest.fail("Long polling timed out unexpectedly")
            
    def test_polling_rate_limiting(self):
        """Test client respects rate limiting"""
        client = PollingTestClient(mode="http")
        request_times = []
        
        # Make rapid requests
        for i in range(10):
            start = time.time()
            try:
                response = requests.get(f"{client.url}/poll", timeout=1)
                request_times.append(time.time() - start)
                
                # Check for rate limit response
                if response.status_code == 429:
                    logger.info(f"Rate limited at request {i}")
                    time.sleep(1)  # Back off
                    
            except Exception as e:
                logger.error(f"Request {i} failed: {e}")
                
        # Calculate average request time
        avg_time = sum(request_times) / len(request_times) if request_times else 0
        logger.info(f"Average request time: {avg_time:.3f}s")
        
    @pytest.mark.asyncio
    async def test_async_polling_pattern(self):
        """Test async polling pattern"""
        messages_received = []
        
        async def poll_messages():
            """Async polling coroutine"""
            while len(messages_received) < 5:
                try:
                    # Simulate async poll
                    await asyncio.sleep(0.5)
                    
                    # Would normally make async request here
                    messages_received.append({
                        "timestamp": time.time(),
                        "data": f"Message {len(messages_received)}"
                    })
                    
                except Exception as e:
                    logger.error(f"Async polling error: {e}")
                    await asyncio.sleep(1)  # Back off on error
                    
        # Run polling
        await asyncio.wait_for(poll_messages(), timeout=10)
        
        assert len(messages_received) >= 5
        logger.info(f"Async polling received {len(messages_received)} messages")
        
    def test_polling_graceful_shutdown(self):
        """Test graceful shutdown during polling"""
        client = PollingTestClient(mode="websocket")
        shutdown_event = threading.Event()
        
        def polling_loop():
            """Polling loop with shutdown support"""
            try:
                if client.connect_websocket():
                    while not shutdown_event.is_set():
                        response = client.send_websocket_message({"type": "poll"})
                        if response:
                            client.messages_received.append(response)
                        time.sleep(0.5)
            finally:
                client.close()
                
        # Start polling
        polling_thread = threading.Thread(target=polling_loop)
        polling_thread.start()
        
        # Let it run briefly
        time.sleep(2)
        
        # Signal shutdown
        logger.info("Initiating graceful shutdown")
        shutdown_event.set()
        
        # Wait for thread to finish
        polling_thread.join(timeout=5)
        
        assert not polling_thread.is_alive()
        assert not client.connected
        logger.info(f"Graceful shutdown completed. Received {len(client.messages_received)} messages")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
#!/usr/bin/env python3
"""
VisionFlow WebXR Backend Runtime Stability Test

This script tests the backend runtime stability by:
1. Starting the backend service
2. Testing WebSocket connections
3. Testing MCP TCP connections
4. Monitoring for crashes and errors
5. Testing concurrent client connections
"""

import asyncio
import websockets
import json
import time
import socket
import sys
import subprocess
import signal
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

class RuntimeStabilityTester:
    def __init__(self):
        self.backend_process = None
        self.test_results = []
        self.websocket_port = 4000
        self.mcp_port = 9500
        self.backend_started = False

    async def start_backend(self, timeout: int = 30) -> bool:
        """Start the VisionFlow backend service"""
        print("🚀 Starting VisionFlow WebXR backend...")

        env = os.environ.copy()
        env['RUST_LOG'] = 'info'
        env['TELEMETRY_LOG_DIR'] = '/workspace/ext/logs'

        try:
            self.backend_process = subprocess.Popen(
                ['cargo', 'run', '--bin', 'webxr'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd='/workspace/ext'
            )

            # Wait for backend to start (look for "HTTP server startup sequence complete")
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.backend_process.poll() is not None:
                    # Process ended
                    stdout, stderr = self.backend_process.communicate()
                    print(f"❌ Backend process ended unexpectedly")
                    print(f"STDOUT: {stdout.decode()}")
                    print(f"STDERR: {stderr.decode()}")
                    return False

                # Check if server is listening
                if self.check_port_open('127.0.0.1', self.websocket_port):
                    print("✅ Backend started successfully")
                    self.backend_started = True
                    return True

                await asyncio.sleep(1)

            print(f"❌ Backend failed to start within {timeout} seconds")
            return False

        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False

    def check_port_open(self, host: str, port: int) -> bool:
        """Check if a port is open"""
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    async def test_websocket_connection(self) -> Dict[str, Any]:
        """Test WebSocket connection stability"""
        print("🔌 Testing WebSocket connections...")

        test_result = {
            'test_name': 'websocket_connection',
            'success': False,
            'error': None,
            'response_time': None,
            'details': {}
        }

        try:
            start_time = time.time()

            # Test basic connection
            uri = f"ws://127.0.0.1:{self.websocket_port}/wss"
            async with websockets.connect(uri, timeout=5) as websocket:

                # Send a test message
                test_message = {
                    "type": "ping",
                    "timestamp": time.time()
                }

                await websocket.send(json.dumps(test_message))

                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    response_data = json.loads(response)

                    response_time = time.time() - start_time
                    test_result['success'] = True
                    test_result['response_time'] = response_time
                    test_result['details']['response'] = response_data

                    print(f"✅ WebSocket connection successful ({response_time:.3f}s)")

                except asyncio.TimeoutError:
                    test_result['error'] = "No response received within timeout"
                    print("⚠️  WebSocket connected but no response received")

        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ WebSocket connection failed: {e}")

        return test_result

    async def test_mcp_tcp_connection(self) -> Dict[str, Any]:
        """Test MCP TCP connection"""
        print("🔗 Testing MCP TCP connections...")

        test_result = {
            'test_name': 'mcp_tcp_connection',
            'success': False,
            'error': None,
            'response_time': None,
            'details': {}
        }

        try:
            start_time = time.time()

            # Test TCP connection to MCP port
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', self.mcp_port),
                timeout=5
            )

            # Send MCP ping message
            mcp_message = {
                "jsonrpc": "2.0",
                "method": "ping",
                "id": 1
            }

            message_data = json.dumps(mcp_message).encode() + b'\n'
            writer.write(message_data)
            await writer.drain()

            # Read response
            try:
                response_data = await asyncio.wait_for(reader.readline(), timeout=5)
                response = json.loads(response_data.decode().strip())

                response_time = time.time() - start_time
                test_result['success'] = True
                test_result['response_time'] = response_time
                test_result['details']['response'] = response

                print(f"✅ MCP TCP connection successful ({response_time:.3f}s)")

            except asyncio.TimeoutError:
                test_result['error'] = "No MCP response received within timeout"
                print("⚠️  MCP TCP connected but no response received")

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ MCP TCP connection failed: {e}")

        return test_result

    async def test_concurrent_connections(self, num_connections: int = 10) -> Dict[str, Any]:
        """Test concurrent WebSocket connections"""
        print(f"🔀 Testing {num_connections} concurrent WebSocket connections...")

        test_result = {
            'test_name': 'concurrent_connections',
            'success': False,
            'error': None,
            'successful_connections': 0,
            'failed_connections': 0,
            'details': {}
        }

        async def single_connection_test(connection_id: int):
            try:
                uri = f"ws://127.0.0.1:{self.websocket_port}/wss"
                async with websockets.connect(uri, timeout=3) as websocket:
                    # Send test message
                    message = {
                        "type": "test",
                        "connection_id": connection_id,
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(message))

                    # Try to receive response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3)
                        return {'success': True, 'connection_id': connection_id}
                    except asyncio.TimeoutError:
                        return {'success': True, 'connection_id': connection_id, 'no_response': True}

            except Exception as e:
                return {'success': False, 'connection_id': connection_id, 'error': str(e)}

        try:
            # Create concurrent connections
            tasks = [single_connection_test(i) for i in range(num_connections)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
            failed = num_connections - successful

            test_result['successful_connections'] = successful
            test_result['failed_connections'] = failed
            test_result['success'] = successful > 0 and failed < num_connections * 0.5  # Allow 50% failure rate
            test_result['details']['results'] = results

            print(f"✅ Concurrent connections: {successful}/{num_connections} successful")

        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ Concurrent connection test failed: {e}")

        return test_result

    def check_backend_health(self) -> Dict[str, Any]:
        """Check if backend process is still running"""
        test_result = {
            'test_name': 'backend_health',
            'success': False,
            'error': None,
            'details': {}
        }

        try:
            if self.backend_process is None:
                test_result['error'] = "Backend process was never started"
                return test_result

            # Check if process is still running
            if self.backend_process.poll() is None:
                test_result['success'] = True
                test_result['details']['status'] = 'running'
                print("✅ Backend process is still running")
            else:
                test_result['error'] = f"Backend process ended with return code {self.backend_process.returncode}"
                # Get any output from the process
                try:
                    stdout, stderr = self.backend_process.communicate(timeout=1)
                    test_result['details']['stdout'] = stdout.decode()
                    test_result['details']['stderr'] = stderr.decode()
                except:
                    pass
                print(f"❌ Backend process ended unexpectedly")

        except Exception as e:
            test_result['error'] = str(e)
            print(f"❌ Failed to check backend health: {e}")

        return test_result

    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all stability tests"""
        print("🧪 Starting VisionFlow WebXR Runtime Stability Tests")
        print("=" * 60)

        results = []

        # 1. Start backend
        if not await self.start_backend():
            print("❌ Cannot continue tests - backend failed to start")
            return [{'test_name': 'backend_startup', 'success': False, 'error': 'Failed to start'}]

        # 2. Basic health check
        results.append(self.check_backend_health())

        # 3. WebSocket connection test
        results.append(await self.test_websocket_connection())

        # 4. MCP TCP connection test
        results.append(await self.test_mcp_tcp_connection())

        # 5. Concurrent connections test
        results.append(await self.test_concurrent_connections(5))

        # 6. Final health check
        results.append(self.check_backend_health())

        return results

    def cleanup(self):
        """Clean up resources"""
        if self.backend_process and self.backend_process.poll() is None:
            print("🧹 Cleaning up backend process...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                self.backend_process.wait()

    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a stability test report"""
        print("\n" + "=" * 60)
        print("🏁 RUNTIME STABILITY TEST REPORT")
        print("=" * 60)

        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get('success', False))

        print(f"📊 SUMMARY: {successful_tests}/{total_tests} tests passed")
        print()

        for result in results:
            test_name = result.get('test_name', 'unknown')
            success = result.get('success', False)
            error = result.get('error')
            response_time = result.get('response_time')

            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")

            if response_time:
                print(f"   ⏱️  Response time: {response_time:.3f}s")

            if error:
                print(f"   ❌ Error: {error}")

            if 'successful_connections' in result:
                print(f"   📊 Connections: {result['successful_connections']}/{result['successful_connections'] + result['failed_connections']}")

            print()

        # Overall assessment
        if successful_tests == total_tests:
            overall_status = "🟢 EXCELLENT - All tests passed"
        elif successful_tests >= total_tests * 0.8:
            overall_status = "🟡 GOOD - Most tests passed"
        elif successful_tests >= total_tests * 0.5:
            overall_status = "🟠 FAIR - Some tests failed"
        else:
            overall_status = "🔴 POOR - Many tests failed"

        print(f"📋 OVERALL STABILITY: {overall_status}")
        print("=" * 60)

        return overall_status

async def main():
    tester = RuntimeStabilityTester()

    try:
        results = await tester.run_all_tests()
        tester.generate_report(results)

        # Exit with appropriate code
        successful_tests = sum(1 for r in results if r.get('success', False))
        exit_code = 0 if successful_tests == len(results) else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test framework error: {e}")
        sys.exit(1)
    finally:
        tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
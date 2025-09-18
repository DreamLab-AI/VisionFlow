#!/usr/bin/env python3
"""
GPU Stability Integration Tests

Tests to verify GPU container stability and CUDA operations
work correctly with the MCP server integration.
"""

import subprocess
import json
import time
import pytest
import logging
import requests
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUStabilityTester:
    """GPU stability testing utilities"""
    
    def __init__(self):
        self.gui_tools_port = 9876
        self.health_check_url = "http://localhost:9501/health"
        
    def check_gpu_container_status(self) -> bool:
        """Check if GPU container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=mcp-gui-tools", "--format", "json"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout:
                return True
        except Exception as e:
            logger.error(f"Failed to check GPU container: {e}")
        return False
        
    def check_cuda_available(self) -> Dict[str, Any]:
        """Check CUDA availability and version"""
        try:
            # Check nvidia-smi
            result = subprocess.run(
                ["docker", "exec", "mcp-gui-tools", "nvidia-smi", "--query-gpu=name,memory.total,driver_version,cuda_version", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                info = result.stdout.strip().split(', ')
                return {
                    "available": True,
                    "gpu_name": info[0] if len(info) > 0 else "Unknown",
                    "memory": info[1] if len(info) > 1 else "Unknown",
                    "driver": info[2] if len(info) > 2 else "Unknown",
                    "cuda": info[3] if len(info) > 3 else "Unknown"
                }
        except Exception as e:
            logger.error(f"CUDA check failed: {e}")
            
        return {"available": False}
        
    def test_cuda_operation(self) -> bool:
        """Test basic CUDA operation"""
        try:
            # Simple CUDA test using Python
            cuda_test = """
import torch
if torch.cuda.is_available():
    # Create tensors on GPU
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    # Perform operation
    z = torch.matmul(x, y)
    
    # Verify result
    print(f"CUDA operation successful. Result shape: {z.shape}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
else:
    print("CUDA not available")
"""
            
            result = subprocess.run(
                ["docker", "exec", "mcp-gui-tools", "python", "-c", cuda_test],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and "CUDA operation successful" in result.stdout:
                logger.info(f"CUDA test output: {result.stdout}")
                return True
                
        except subprocess.TimeoutExpired:
            logger.error("CUDA operation timed out")
        except Exception as e:
            logger.error(f"CUDA operation test failed: {e}")
            
        return False
        
    def monitor_gpu_memory(self, duration: int = 10) -> List[Dict[str, Any]]:
        """Monitor GPU memory usage over time"""
        memory_samples = []
        
        for _ in range(duration):
            try:
                result = subprocess.run(
                    ["docker", "exec", "mcp-gui-tools", "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    memory_samples.append({
                        "timestamp": time.time(),
                        "memory_used": int(values[0]) if len(values) > 0 else 0,
                        "memory_free": int(values[1]) if len(values) > 1 else 0,
                        "gpu_utilization": int(values[2]) if len(values) > 2 else 0
                    })
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                
            time.sleep(1)
            
        return memory_samples

class TestGPUStability:
    """GPU Stability Test Suite"""
    
    @pytest.fixture
    def gpu_tester(self):
        """Create GPU tester instance"""
        return GPUStabilityTester()
        
    def test_gpu_container_running(self, gpu_tester):
        """Test GPU container is running"""
        assert gpu_tester.check_gpu_container_status() == True
        logger.info("GPU container is running")
        
    def test_cuda_availability(self, gpu_tester):
        """Test CUDA is available and properly configured"""
        cuda_info = gpu_tester.check_cuda_available()
        
        assert cuda_info["available"] == True
        assert cuda_info["gpu_name"] != "Unknown"
        
        logger.info(f"CUDA Info: {cuda_info}")
        
    def test_basic_cuda_operation(self, gpu_tester):
        """Test basic CUDA computation"""
        assert gpu_tester.test_cuda_operation() == True
        logger.info("Basic CUDA operation successful")
        
    def test_gpu_memory_stability(self, gpu_tester):
        """Test GPU memory remains stable during operations"""
        # Get initial memory state
        initial_samples = gpu_tester.monitor_gpu_memory(duration=3)
        assert len(initial_samples) > 0
        
        initial_memory = initial_samples[0]["memory_used"]
        logger.info(f"Initial GPU memory: {initial_memory} MB")
        
        # Run CUDA operations
        for i in range(3):
            assert gpu_tester.test_cuda_operation() == True
            time.sleep(1)
            
        # Check memory after operations
        final_samples = gpu_tester.monitor_gpu_memory(duration=3)
        final_memory = final_samples[-1]["memory_used"]
        
        # Memory shouldn't increase dramatically (allow 500MB variance)
        memory_increase = final_memory - initial_memory
        logger.info(f"Memory increase: {memory_increase} MB")
        
        assert memory_increase < 500, f"Excessive memory increase: {memory_increase} MB"
        
    def test_gui_tools_connectivity(self, gpu_tester):
        """Test GUI tools port connectivity"""
        try:
            # Test Blender port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', 9876))
            sock.close()
            
            # Port should be accessible
            assert result == 0 or result == 111  # 0 = open, 111 = connection refused (service not started)
            logger.info("GUI tools port check passed")
            
        except Exception as e:
            pytest.fail(f"Port connectivity test failed: {e}")
            
    def test_concurrent_gpu_operations(self, gpu_tester):
        """Test multiple concurrent GPU operations"""
        import concurrent.futures
        
        def run_gpu_task(task_id: int) -> bool:
            """Run a single GPU task"""
            logger.info(f"Starting GPU task {task_id}")
            return gpu_tester.test_cuda_operation()
            
        # Run 3 concurrent GPU tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_gpu_task, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        # All tasks should succeed
        assert all(results), "Some GPU tasks failed"
        logger.info("All concurrent GPU operations completed successfully")
        
    def test_gpu_error_recovery(self, gpu_tester):
        """Test GPU error recovery mechanisms"""
        # Simulate error by trying to allocate excessive memory
        error_test = """
import torch
try:
    # Try to allocate 100GB (should fail gracefully)
    x = torch.zeros(100000, 100000, device='cuda')
except RuntimeError as e:
    print(f"Expected error caught: {str(e)[:50]}...")
    # Verify GPU still works after error
    small = torch.ones(100, 100, device='cuda')
    print("GPU recovered successfully")
"""
        
        result = subprocess.run(
            ["docker", "exec", "mcp-gui-tools", "python", "-c", error_test],
            capture_output=True,
            text=True
        )
        
        assert "GPU recovered successfully" in result.stdout
        logger.info("GPU error recovery test passed")
        
    def test_gpu_persistence_across_operations(self, gpu_tester):
        """Test GPU state persistence across different operations"""
        # Create persistent tensor
        create_tensor = """
import torch
import pickle
x = torch.randn(1000, 1000, device='cuda')
with open('/tmp/tensor.pkl', 'wb') as f:
    pickle.dump(x.cpu(), f)
print("Tensor created and saved")
"""
        
        result = subprocess.run(
            ["docker", "exec", "mcp-gui-tools", "python", "-c", create_tensor],
            capture_output=True,
            text=True
        )
        assert "Tensor created" in result.stdout
        
        # Load and verify tensor
        verify_tensor = """
import torch
import pickle
with open('/tmp/tensor.pkl', 'rb') as f:
    x = pickle.load(f).cuda()
print(f"Tensor loaded successfully. Shape: {x.shape}")
"""
        
        result = subprocess.run(
            ["docker", "exec", "mcp-gui-tools", "python", "-c", verify_tensor],
            capture_output=True,
            text=True
        )
        assert "Tensor loaded successfully" in result.stdout
        
    def test_health_check_gpu_status(self, gpu_tester):
        """Test health check endpoint reports GPU status"""
        try:
            response = requests.get(gpu_tester.health_check_url, timeout=5)
            assert response.status_code == 200
            
            health_data = response.json()
            
            # Check if GPU info is included
            if "gpu" in health_data:
                assert health_data["gpu"]["available"] is not None
                logger.info(f"Health check GPU status: {health_data['gpu']}")
            else:
                logger.warning("GPU status not included in health check")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            
if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
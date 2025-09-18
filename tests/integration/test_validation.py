#!/usr/bin/env python3
"""
Test Validation Script

Simple validation script to check if the integration test environment is ready.
"""

import sys
import socket
import subprocess
import json
from pathlib import Path

def check_python():
    """Check Python version"""
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False

def check_imports():
    """Check if required modules can be imported"""
    required_modules = {
        'json': 'json',
        'socket': 'socket',
        'subprocess': 'subprocess',
        'time': 'time',
        'threading': 'threading'
    }
    
    optional_modules = {
        'pytest': 'pytest',
        'requests': 'requests',
        'websocket': 'websocket-client'
    }
    
    print("\nChecking required modules:")
    all_required_available = True
    
    for name, package in required_modules.items():
        try:
            __import__(name)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} missing (install: pip install {package})")
            all_required_available = False
    
    print("\nChecking optional modules:")
    for name, package in optional_modules.items():
        try:
            __import__(name)
            print(f"✓ {name} available")
        except ImportError:
            print(f"? {name} missing (install: pip install {package})")
    
    return all_required_available

def check_services():
    """Check if required services are running"""
    services = {
        'TCP Server': ('localhost', 9500),
        'WebSocket Bridge': ('localhost', 3002),
        'Health Check': ('localhost', 9501)
    }
    
    print("\nChecking services:")
    available_services = 0
    
    for name, (host, port) in services.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"✓ {name} ({port}) - Available")
                available_services += 1
            else:
                print(f"? {name} ({port}) - Not responding")
        except Exception as e:
            print(f"? {name} ({port}) - Error: {e}")
    
    return available_services

def check_docker():
    """Check Docker and GPU container"""
    print("\nChecking Docker environment:")
    
    try:
        # Check if docker command is available
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Docker available: {result.stdout.strip()}")
            
            # Check for GPU container
            result = subprocess.run(['docker', 'ps', '--filter', 'name=mcp-gui-tools', '--format', 'json'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                print("✓ GPU container (mcp-gui-tools) - Running")
                return True
            else:
                print("? GPU container (mcp-gui-tools) - Not running")
                return False
        else:
            print("? Docker not available")
            return False
            
    except Exception as e:
        print(f"? Docker check failed: {e}")
        return False

def check_test_files():
    """Check if test files exist"""
    test_dir = Path(__file__).parent
    test_files = [
        'tcp_persistence_test.py',
        'gpu_stability_test.py',
        'client_polling_test.py',
        'security_validation_test.py',
        'test_runner.py',
        'run_tests.sh'
    ]
    
    print("\nChecking test files:")
    available_files = 0
    
    for test_file in test_files:
        file_path = test_dir / test_file
        if file_path.exists():
            print(f"✓ {test_file}")
            available_files += 1
        else:
            print(f"✗ {test_file} - Missing")
    
    return available_files == len(test_files)

def generate_summary():
    """Generate environment summary"""
    print("\n" + "="*60)
    print("INTEGRATION TEST ENVIRONMENT SUMMARY")
    print("="*60)
    
    checks = {
        "Python Version": check_python(),
        "Required Modules": check_imports(),
        "Test Files": check_test_files(),
        "Services": check_services() >= 2,  # At least 2 services should be available
        "Docker/GPU": check_docker()
    }
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    print(f"\nOverall Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ Environment is ready for integration testing")
        return True
    else:
        print("? Environment needs setup before running tests")
        print("\nTo set up the environment:")
        print("1. Install missing Python packages: pip install -r requirements.txt")
        print("2. Start required services (TCP server, WebSocket bridge, etc.)")
        print("3. Ensure Docker and GPU container are running")
        print("4. Run: ./run_tests.sh setup")
        return False

def main():
    """Main validation function"""
    print("Integration Test Environment Validation")
    print("=" * 50)
    
    try:
        ready = generate_summary()
        sys.exit(0 if ready else 1)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nValidation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
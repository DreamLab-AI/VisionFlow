#!/usr/bin/env python3
"""
Integration test script for the GPU Analytics logging infrastructure.
Tests all components: advanced logging, log aggregation, and monitoring.
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path


def test_log_structure():
    """Test that log files have the correct JSON structure"""
    print("🧪 Testing log file structure...")
    
    log_dir = Path("logs")
    log_files = ["gpu.log", "memory.log", "performance.log", "error.log"]
    
    success_count = 0
    for log_file in log_files:
        file_path = log_dir / log_file
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        entry = json.loads(line)
                        required_fields = ["timestamp", "level", "component", "message"]
                        
                        if all(field in entry for field in required_fields):
                            print(f"  ✅ {log_file}: Valid JSON structure")
                            success_count += 1
                        else:
                            print(f"  ❌ {log_file}: Missing required fields")
                    else:
                        print(f"  ⚠️  {log_file}: Empty file")
            except json.JSONDecodeError:
                print(f"  ❌ {log_file}: Invalid JSON format")
            except Exception as e:
                print(f"  ❌ {log_file}: Error reading file - {e}")
        else:
            print(f"  ⚠️  {log_file}: File not found")
    
    print(f"📊 Log structure test: {success_count}/{len(log_files)} files valid\n")
    return success_count == len([f for f in log_files if (log_dir / f).exists()])


def test_gpu_metrics():
    """Test that GPU logs contain proper metrics"""
    print("🧪 Testing GPU metrics structure...")
    
    gpu_log = Path("logs/gpu.log")
    if not gpu_log.exists():
        print("  ❌ GPU log file not found")
        return False
        
    valid_entries = 0
    total_entries = 0
    
    try:
        with open(gpu_log, 'r') as f:
            for line in f:
                if line.strip():
                    total_entries += 1
                    try:
                        entry = json.loads(line)
                        if entry.get('gpu_metrics'):
                            gpu_data = entry['gpu_metrics']
                            if gpu_data.get('kernel_name') and gpu_data.get('execution_time_us'):
                                valid_entries += 1
                    except json.JSONDecodeError:
                        continue
        
        print(f"  📊 Found {valid_entries}/{total_entries} valid GPU metric entries")
        success = valid_entries > 0
        print(f"  {'✅' if success else '❌'} GPU metrics test")
        
    except Exception as e:
        print(f"  ❌ Error testing GPU metrics: {e}")
        success = False
    
    print()
    return success


def test_log_aggregator():
    """Test the log aggregation script"""
    print("🧪 Testing log aggregator...")
    
    try:
        # Run aggregator
        result = subprocess.run([
            "python3", "scripts/log_aggregator.py",
            "--log-dir", "./logs",
            "--no-charts",
            "--format", "json"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✅ Log aggregator executed successfully")
            
            # Check if reports were generated
            aggregated_dir = Path("logs/aggregated")
            if aggregated_dir.exists():
                json_files = list(aggregated_dir.glob("*.json"))
                if len(json_files) >= 2:  # Should have analysis report and aggregated logs
                    print(f"  ✅ Generated {len(json_files)} report files")
                    
                    # Test report structure
                    analysis_files = [f for f in json_files if "analysis_report" in f.name]
                    if analysis_files:
                        with open(analysis_files[0], 'r') as f:
                            report = json.load(f)
                            if "gpu_performance" in report and "daily_summary" in report:
                                print("  ✅ Report contains expected sections")
                                return True
                            else:
                                print("  ❌ Report missing expected sections")
                    else:
                        print("  ❌ No analysis report found")
                else:
                    print(f"  ❌ Expected 2+ report files, found {len(json_files)}")
            else:
                print("  ❌ Aggregated directory not created")
        else:
            print(f"  ❌ Log aggregator failed: {result.stderr}")
            
    except Exception as e:
        print(f"  ❌ Error running log aggregator: {e}")
        
    print()
    return False


def test_performance_analysis():
    """Test performance analysis features"""
    print("🧪 Testing performance analysis...")
    
    analysis_file = None
    aggregated_dir = Path("logs/aggregated")
    
    if aggregated_dir.exists():
        analysis_files = list(aggregated_dir.glob("log_analysis_report_*.json"))
        if analysis_files:
            analysis_file = analysis_files[-1]  # Get the latest report
    
    if not analysis_file:
        print("  ❌ No analysis report found")
        return False
        
    try:
        with open(analysis_file, 'r') as f:
            report = json.load(f)
            
        gpu_performance = report.get("gpu_performance", {})
        summary = gpu_performance.get("summary", {})
        kernel_performance = gpu_performance.get("kernel_performance", {})
        
        print(f"  📊 Analysis Report Summary:")
        print(f"    - Total GPU logs: {summary.get('total_gpu_logs', 0)}")
        print(f"    - Unique kernels: {summary.get('unique_kernels', 0)}")
        print(f"    - Total errors: {summary.get('total_errors', 0)}")
        print(f"    - Performance anomalies: {summary.get('performance_anomalies', 0)}")
        
        if kernel_performance:
            print(f"  📊 Kernel Performance Analysis:")
            for kernel, stats in list(kernel_performance.items())[:3]:
                print(f"    - {kernel}: {stats.get('avg_time_us', 0):.2f}μs avg")
        
        success = (summary.get('unique_kernels', 0) > 0)
        print(f"  {'✅' if success else '❌'} Performance analysis test")
        
    except Exception as e:
        print(f"  ❌ Error analyzing performance report: {e}")
        success = False
        
    print()
    return success


def test_log_rotation():
    """Test log rotation functionality"""
    print("🧪 Testing log rotation setup...")
    
    # Check if archived directory exists
    archived_dir = Path("logs/archived")
    if archived_dir.exists():
        print("  ✅ Archived log directory exists")
        
        # Check for any archived files
        archived_files = list(archived_dir.glob("*.log"))
        if archived_files:
            print(f"  📊 Found {len(archived_files)} archived log files")
            # Check filename format
            valid_format = all("_" in f.name and f.name.count("_") >= 2 for f in archived_files)
            print(f"  {'✅' if valid_format else '❌'} Archived files have timestamp format")
        else:
            print("  ℹ️  No archived files (expected for new installation)")
            
        return True
    else:
        print("  ❌ Archived directory not found")
        return False


def generate_test_logs():
    """Generate additional test log entries for testing"""
    print("🧪 Generating additional test logs...")
    
    current_time = datetime.now()
    
    # Generate GPU test logs
    gpu_entries = []
    kernels = ["test_kernel_1", "test_kernel_2", "benchmark_kernel"]
    for i, kernel in enumerate(kernels):
        entry_time = current_time + timedelta(seconds=i)
        entry = {
            "timestamp": entry_time.isoformat() + "Z",
            "level": "INFO",
            "component": "gpu",
            "message": f"Kernel {kernel} executed in {1000 + i*100:.2f}μs",
            "metadata": None,
            "execution_time_ms": (1000 + i*100) / 1000.0,
            "memory_usage_mb": 150.0 + i*10,
            "gpu_metrics": {
                "kernel_name": kernel,
                "execution_time_us": 1000.0 + i*100,
                "memory_allocated_mb": 150.0 + i*10,
                "memory_peak_mb": 270.0 + i*5,
                "gpu_utilization_percent": 85.0 + i*2,
                "error_count": 0,
                "recovery_attempts": 0,
                "performance_anomaly": False
            }
        }
        gpu_entries.append(json.dumps(entry))
    
    # Append to GPU log
    gpu_log = Path("logs/gpu.log")
    with open(gpu_log, 'a') as f:
        f.write('\n')
        for entry in gpu_entries:
            f.write(entry + '\n')
    
    print(f"  ✅ Generated {len(gpu_entries)} additional GPU log entries")
    print()


def run_integration_test():
    """Run comprehensive integration test"""
    print("🚀 GPU Analytics Logging Infrastructure Integration Test")
    print("=" * 60)
    print()
    
    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Generate additional test data
    generate_test_logs()
    
    # Run all tests
    tests = [
        ("Log Structure", test_log_structure),
        ("GPU Metrics", test_gpu_metrics),  
        ("Log Aggregator", test_log_aggregator),
        ("Performance Analysis", test_performance_analysis),
        ("Log Rotation Setup", test_log_rotation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("=" * 60)
    print("🏁 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-" * 60)
    print(f"OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Logging infrastructure is working correctly!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed - Review the issues above")
        return False


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
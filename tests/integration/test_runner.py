#!/usr/bin/env python3
"""
Integration Test Runner

Main test runner for all integration tests with reporting and documentation.
"""

import subprocess
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationTestRunner:
    """Main test runner for integration tests"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {}
        self.start_time = datetime.now()
        
    def run_test_suite(self, test_file: str) -> dict:
        """Run a specific test suite"""
        logger.info(f"Running test suite: {test_file}")
        
        start = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(self.test_dir / test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per suite
            )
            
            duration = time.time() - start
            
            return {
                "file": test_file,
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "file": test_file,
                "success": False,
                "duration": time.time() - start,
                "error": "Test suite timed out",
                "return_code": -1
            }
        except Exception as e:
            return {
                "file": test_file,
                "success": False,
                "duration": time.time() - start,
                "error": str(e),
                "return_code": -1
            }
            
    def run_all_tests(self):
        """Run all integration test suites"""
        test_suites = [
            "tcp_persistence_test.py",
            "gpu_stability_test.py", 
            "client_polling_test.py",
            "security_validation_test.py"
        ]
        
        logger.info("Starting integration test execution")
        logger.info(f"Found {len(test_suites)} test suites")
        
        for test_suite in test_suites:
            result = self.run_test_suite(test_suite)
            self.results[test_suite] = result
            
            if result["success"]:
                logger.info(f"✓ {test_suite} passed in {result['duration']:.2f}s")
            else:
                logger.error(f"✗ {test_suite} failed in {result['duration']:.2f}s")
                
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        report = []
        report.append("# Integration Test Results")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Duration: {total_duration:.2f} seconds")
        report.append("")
        
        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        report.append("## Summary")
        report.append(f"- Total Test Suites: {total_tests}")
        report.append(f"- Passed: {passed_tests}")
        report.append(f"- Failed: {failed_tests}")
        report.append(f"- Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        
        for suite_name, result in self.results.items():
            status = "✓ PASSED" if result["success"] else "✗ FAILED"
            report.append(f"### {suite_name} - {status}")
            report.append(f"Duration: {result['duration']:.2f} seconds")
            
            if not result["success"]:
                report.append("**Error Output:**")
                if "error" in result:
                    report.append(f"```\n{result['error']}\n```")
                if result.get("stderr"):
                    report.append(f"```\n{result['stderr']}\n```")
                    
            if result.get("stdout"):
                # Extract key information from pytest output
                lines = result["stdout"].split('\n')
                test_lines = [line for line in lines if '::' in line and ('PASSED' in line or 'FAILED' in line)]
                
                if test_lines:
                    report.append("**Test Cases:**")
                    for line in test_lines[:10]:  # Show first 10 test cases
                        report.append(f"- {line.strip()}")
                        
            report.append("")
            
        # Recommendations
        report.append("## Recommendations")
        
        if failed_tests > 0:
            report.append("### Issues Found")
            for suite_name, result in self.results.items():
                if not result["success"]:
                    report.append(f"- **{suite_name}**: Review test failures and fix underlying issues")
                    
        else:
            report.append("All integration tests passed successfully!")
            
        report.append("")
        report.append("### Performance")
        
        # Performance analysis
        slowest = max(self.results.items(), key=lambda x: x[1]["duration"])
        fastest = min(self.results.items(), key=lambda x: x[1]["duration"])
        
        report.append(f"- Slowest test suite: {slowest[0]} ({slowest[1]['duration']:.2f}s)")
        report.append(f"- Fastest test suite: {fastest[0]} ({fastest[1]['duration']:.2f}s)")
        
        avg_duration = sum(r["duration"] for r in self.results.values()) / len(self.results)
        report.append(f"- Average duration: {avg_duration:.2f}s")
        
        return "\n".join(report)
        
    def save_results(self):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_file = self.test_dir / f"test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "timestamp": self.start_time.isoformat(),
                "duration": (datetime.now() - self.start_time).total_seconds(),
                "results": self.results
            }, f, indent=2)
            
        logger.info(f"Detailed results saved to: {json_file}")
        
        # Save readable report
        report_file = self.test_dir / f"test_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(self.generate_report())
            
        logger.info(f"Test report saved to: {report_file}")
        
        # Always save latest report
        latest_report = self.test_dir / "latest_test_report.md"
        with open(latest_report, 'w') as f:
            f.write(self.generate_report())
            
        return latest_report

def main():
    """Main function"""
    runner = IntegrationTestRunner()
    
    try:
        runner.run_all_tests()
        report_file = runner.save_results()
        
        # Print summary
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        total_tests = len(runner.results)
        passed_tests = sum(1 for r in runner.results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\nFull report available at: {report_file}")
        
        # Return appropriate exit code
        sys.exit(0 if failed_tests == 0 else 1)
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
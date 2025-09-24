# Hybrid Architecture Testing Strategy
## Comprehensive Testing Plan for Docker-MCP Integration

### Executive Summary

This document outlines a comprehensive testing strategy for validating the hybrid Docker exec + TCP/MCP architecture. The testing approach ensures reliability, performance, and graceful degradation while minimizing risks during migration.

## Testing Framework Overview

### Testing Phases
1. **Unit Testing**: Individual component validation
2. **Integration Testing**: Cross-component interaction validation
3. **System Testing**: End-to-end workflow validation
4. **Performance Testing**: Load and stress testing
5. **Resilience Testing**: Fault injection and recovery testing
6. **Migration Testing**: A/B testing and rollback procedures

## Phase 1: Unit Testing

### 1.1 Docker Hive Mind Module Tests
```rust
// Test file: /workspace/ext/src/utils/docker_hive_mind/tests.rs

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_docker_connection_creation() {
        let hive_mind = DockerHiveMind::new("test-container".to_string());

        // Test successful connection
        let result = hive_mind.health_check().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_swarm_spawning() {
        let hive_mind = create_docker_hive_mind();
        let config = SwarmConfig::default();

        let result = hive_mind.spawn_swarm("test task", config).await;
        assert!(result.is_ok());

        let session_id = result.unwrap();
        assert!(!session_id.is_empty());
        assert!(session_id.starts_with("swarm-"));
    }

    #[tokio::test]
    async fn test_session_management() {
        let hive_mind = create_docker_hive_mind();

        // Spawn test swarm
        let config = SwarmConfig::default();
        let session_id = hive_mind.spawn_swarm("test task", config).await.unwrap();

        // Get sessions
        let sessions = hive_mind.get_sessions().await.unwrap();
        assert!(!sessions.is_empty());

        // Check session exists
        let found = sessions.iter().any(|s| s.session_id == session_id);
        assert!(found);

        // Stop swarm
        let result = hive_mind.stop_swarm(&session_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_error_handling() {
        let hive_mind = DockerHiveMind::new("non-existent-container".to_string());

        let config = SwarmConfig::default();
        let result = hive_mind.spawn_swarm("test task", config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cleanup_orphaned_processes() {
        let hive_mind = create_docker_hive_mind();

        let result = hive_mind.cleanup_orphaned_processes().await;
        assert!(result.is_ok());

        let cleaned_count = result.unwrap();
        assert!(cleaned_count >= 0);
    }
}
```

### 1.2 Hybrid Connection Pool Tests
```rust
// Test file: /workspace/ext/src/utils/hybrid_performance_optimizer/tests.rs

#[tokio::test]
async fn test_connection_pool_management() {
    let pool_config = PoolConfig::default();
    let pool = DockerConnectionPool::new(pool_config);

    // Test connection acquisition
    let conn = pool.get_connection().await.unwrap();
    assert!(conn.is_healthy);

    // Test connection return
    pool.return_connection(conn).await;

    // Test pool metrics
    let metrics = pool.get_metrics().await;
    assert_eq!(metrics.total_connections, 1);
    assert_eq!(metrics.active_connections, 0);
}

#[tokio::test]
async fn test_cache_functionality() {
    let cache = HybridCache::new(CacheConfig::default());

    // Test session caching
    let session = create_test_session();
    cache.cache_session("test-session".to_string(), session.clone()).await;

    let cached_session = cache.get_session("test-session").await;
    assert!(cached_session.is_some());
    assert_eq!(cached_session.unwrap().session_id, session.session_id);

    // Test cache expiration
    tokio::time::sleep(Duration::from_secs(61)).await; // Exceed TTL
    let expired_session = cache.get_session("test-session").await;
    assert!(expired_session.is_none());
}
```

### 1.3 Fault Tolerance Tests
```rust
// Test file: /workspace/ext/src/utils/hybrid_fault_tolerance/tests.rs

#[tokio::test]
async fn test_circuit_breaker() {
    let circuit_breaker = CircuitBreaker::new(3, Duration::from_secs(10), 2);

    // Test normal operation (closed state)
    assert!(circuit_breaker.can_execute().await);

    // Simulate failures
    for _ in 0..3 {
        circuit_breaker.record_failure().await;
    }

    // Should be open now
    assert!(!circuit_breaker.can_execute().await);

    // Wait for recovery period
    tokio::time::sleep(Duration::from_secs(11)).await;

    // Should be half-open
    assert!(circuit_breaker.can_execute().await);

    // Record successes to close
    circuit_breaker.record_success().await;
    circuit_breaker.record_success().await;

    assert_eq!(circuit_breaker.get_state().await, CircuitState::Closed);
}

#[tokio::test]
async fn test_network_recovery() {
    let recovery_manager = create_test_recovery_manager();

    // Test container down recovery
    let action = recovery_manager.recover_from_failure(NetworkFailure::ContainerDown).await;
    assert!(matches!(action, RecoveryAction::RestartContainer));

    // Test network partition recovery
    let action = recovery_manager.recover_from_failure(NetworkFailure::NetworkPartition).await;
    assert!(matches!(action, RecoveryAction::WaitAndRetry(_)));
}
```

## Phase 2: Integration Testing

### 2.1 Hybrid API Handler Tests
```rust
// Test file: /workspace/ext/src/handlers/hybrid_health_handler/tests.rs

#[tokio::test]
async fn test_hybrid_status_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/hybrid/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let status: HybridSystemStatus = serde_json::from_slice(&body).unwrap();

    assert!(!status.docker_health.is_empty());
    assert!(!status.mcp_health.is_empty());
}

#[tokio::test]
async fn test_swarm_spawning_endpoint() {
    let app = create_test_app().await;

    let request_body = SpawnSwarmRequest {
        task: "Test integration task".to_string(),
        priority: Some("medium".to_string()),
        strategy: Some("hive-mind".to_string()),
        method: Some("docker".to_string()),
        max_workers: Some(4),
        auto_scale: Some(true),
        config: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/hybrid/spawn-swarm")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["success"], true);
    assert!(result["sessionId"].is_string());
}

#[tokio::test]
async fn test_fallback_behavior() {
    // Disable Docker to test MCP fallback
    let app = create_test_app_with_docker_disabled().await;

    let request_body = SpawnSwarmRequest {
        task: "Fallback test task".to_string(),
        method: Some("hybrid".to_string()),
        ..Default::default()
    };

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/hybrid/spawn-swarm")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let result: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["success"], true);
    assert_eq!(result["method"], "mcp-fallback");
}
```

### 2.2 WebSocket Integration Tests
```rust
#[tokio::test]
async fn test_websocket_status_updates() {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    // Connect to WebSocket
    let ws_stream = connect_async("ws://localhost:3000/ws/hybrid-status").await.unwrap().0;

    // Send status request
    let request = json!({
        "type": "request_status"
    });

    ws_stream.send(Message::Text(request.to_string())).await.unwrap();

    // Receive response
    if let Some(message) = ws_stream.next().await {
        let text = message.unwrap().into_text().unwrap();
        let response: Value = serde_json::from_str(&text).unwrap();

        assert_eq!(response["type"], "status_update");
        assert!(response["payload"].is_object());
    }
}
```

## Phase 3: System Testing

### 3.1 End-to-End Workflow Tests
```bash
#!/bin/bash
# Test script: /workspace/ext/tests/e2e/hybrid_workflow_test.sh

set -e

echo "🚀 Starting hybrid architecture E2E tests"

# Test 1: Full system startup
echo "📋 Test 1: System startup and health check"
curl -f http://localhost:3000/api/hybrid/status || exit 1
echo "✅ System health check passed"

# Test 2: Swarm spawning via Docker
echo "📋 Test 2: Docker swarm spawning"
SWARM_RESULT=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"task":"E2E test task","method":"docker","priority":"medium"}' \
    http://localhost:3000/api/hybrid/spawn-swarm)

SESSION_ID=$(echo $SWARM_RESULT | jq -r '.sessionId')
echo "✅ Swarm spawned: $SESSION_ID"

# Test 3: Session monitoring
echo "📋 Test 3: Session status monitoring"
sleep 5 # Allow swarm to initialize

STATUS_RESULT=$(curl -s http://localhost:3000/api/hybrid/status)
SESSION_COUNT=$(echo $STATUS_RESULT | jq '.activeSessions | length')

if [ "$SESSION_COUNT" -gt 0 ]; then
    echo "✅ Active sessions detected: $SESSION_COUNT"
else
    echo "❌ No active sessions found"
    exit 1
fi

# Test 4: Swarm termination
echo "📋 Test 4: Swarm termination"
curl -f -X POST http://localhost:3000/api/hybrid/swarm/$SESSION_ID/stop || exit 1
echo "✅ Swarm terminated successfully"

# Test 5: Performance report
echo "📋 Test 5: Performance report generation"
PERF_REPORT=$(curl -s http://localhost:3000/api/hybrid/performance-report)
TOTAL_REQUESTS=$(echo $PERF_REPORT | jq '.overall_metrics.total_requests')

if [ "$TOTAL_REQUESTS" -gt 0 ]; then
    echo "✅ Performance report generated: $TOTAL_REQUESTS total requests"
else
    echo "❌ Performance report generation failed"
    exit 1
fi

echo "🎉 All E2E tests passed!"
```

### 3.2 Client-Server Integration Test
```typescript
// Test file: /workspace/ext/client/src/__tests__/hybrid-integration.test.ts

describe('Hybrid System Integration', () => {
  let mockWebSocket: any;

  beforeEach(() => {
    mockWebSocket = {
      send: jest.fn(),
      close: jest.fn(),
      addEventListener: jest.fn(),
    };

    global.WebSocket = jest.fn(() => mockWebSocket);
  });

  test('should connect to hybrid system status', async () => {
    const { result } = renderHook(() => useHybridSystemStatus({
      enableWebSocket: true,
      pollingInterval: 1000,
    }));

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    expect(global.WebSocket).toHaveBeenCalledWith(
      expect.stringContaining('/ws/hybrid-status')
    );
  });

  test('should spawn swarm successfully', async () => {
    const mockResponse = {
      success: true,
      sessionId: 'test-session-123',
      method: 'docker',
    };

    fetchMock.mockResponseOnce(JSON.stringify(mockResponse));

    const { result } = renderHook(() => useHybridSystemStatus());

    const response = await result.current.spawnSwarm('Test task', {
      priority: 'medium',
      method: 'docker',
    });

    expect(response.success).toBe(true);
    expect(response.sessionId).toBe('test-session-123');
  });

  test('should handle fallback gracefully', async () => {
    // Mock Docker failure, then MCP success
    fetchMock
      .mockRejectOnce(new Error('Docker unavailable'))
      .mockResponseOnce(JSON.stringify({
        success: true,
        method: 'mcp-fallback',
        result: { taskId: 'mcp-task-123' }
      }));

    const { result } = renderHook(() => useHybridSystemStatus());

    // This would be handled internally by the hybrid endpoint
    const response = await result.current.spawnSwarm('Fallback test', {
      method: 'hybrid',
    });

    expect(response.success).toBe(true);
    expect(response.method).toBe('mcp-fallback');
  });

  test('should display system health correctly', () => {
    const mockStatus = {
      dockerHealth: 'healthy',
      mcpHealth: 'connected',
      systemStatus: 'healthy',
      activeSessions: [],
      performance: {
        totalRequests: 100,
        successfulRequests: 95,
        cacheHitRatio: 0.8,
      },
    };

    const { getByText } = render(
      <HybridSystemDashboard
        position={[0, 0, 0]}
        onSystemChange={jest.fn()}
      />
    );

    // Mock the hook to return our test status
    jest.spyOn(require('../hooks/useHybridSystemStatus'), 'default')
      .mockReturnValue({
        status: mockStatus,
        isSystemHealthy: true,
        isLoading: false,
        error: null,
        refresh: jest.fn(),
        spawnSwarm: jest.fn(),
        stopSwarm: jest.fn(),
      });

    expect(getByText(/System Status/)).toBeInTheDocument();
    expect(getByText(/Docker: healthy/)).toBeInTheDocument();
    expect(getByText(/MCP: connected/)).toBeInTheDocument();
  });
});
```

## Phase 4: Performance Testing

### 4.1 Load Testing Script
```bash
#!/bin/bash
# Load test script: /workspace/ext/tests/performance/load_test.sh

echo "🔥 Starting hybrid architecture load tests"

# Test concurrent swarm spawning
echo "📋 Test 1: Concurrent swarm spawning (10 simultaneous)"
for i in {1..10}; do
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "{\"task\":\"Load test task $i\",\"method\":\"hybrid\"}" \
        http://localhost:3000/api/hybrid/spawn-swarm &
done

wait
echo "✅ Concurrent spawning completed"

# Test status endpoint under load
echo "📋 Test 2: Status endpoint load (100 requests)"
for i in {1..100}; do
    curl -s http://localhost:3000/api/hybrid/status > /dev/null &
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Completed $i/100 requests"
    fi
done

wait
echo "✅ Status endpoint load test completed"

# Measure response times
echo "📋 Test 3: Response time measurement"
START_TIME=$(date +%s%3N)
curl -s http://localhost:3000/api/hybrid/status > /dev/null
END_TIME=$(date +%s%3N)
RESPONSE_TIME=$((END_TIME - START_TIME))

echo "✅ Response time: ${RESPONSE_TIME}ms"

if [ "$RESPONSE_TIME" -lt 1000 ]; then
    echo "✅ Response time within acceptable limits"
else
    echo "⚠️  Response time exceeds 1000ms threshold"
fi
```

### 4.2 Memory and Resource Monitoring
```python
#!/usr/bin/env python3
# Resource monitor: /workspace/ext/tests/performance/resource_monitor.py

import psutil
import requests
import time
import json
from datetime import datetime

def monitor_system_resources(duration_seconds=300):
    """Monitor system resources during hybrid operation"""

    start_time = time.time()
    metrics = []

    while time.time() - start_time < duration_seconds:
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Get hybrid system status
        try:
            response = requests.get('http://localhost:3000/api/hybrid/status', timeout=5)
            if response.status_code == 200:
                hybrid_status = response.json()
                active_sessions = len(hybrid_status.get('activeSessions', []))
                network_latency = hybrid_status.get('networkLatency', 0)
            else:
                active_sessions = -1
                network_latency = -1
        except Exception as e:
            print(f"Failed to get hybrid status: {e}")
            active_sessions = -1
            network_latency = -1

        metric = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'active_sessions': active_sessions,
            'network_latency_ms': network_latency,
        }

        metrics.append(metric)
        print(f"CPU: {cpu_percent:5.1f}% | Memory: {memory.percent:5.1f}% | "
              f"Sessions: {active_sessions:2d} | Latency: {network_latency:3d}ms")

        time.sleep(1)

    # Save metrics to file
    with open('resource_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Calculate averages
    avg_cpu = sum(m['cpu_percent'] for m in metrics) / len(metrics)
    avg_memory = sum(m['memory_percent'] for m in metrics) / len(metrics)
    avg_latency = sum(m['network_latency_ms'] for m in metrics if m['network_latency_ms'] >= 0) / len(metrics)

    print(f"\n📊 Performance Summary:")
    print(f"   Average CPU: {avg_cpu:.1f}%")
    print(f"   Average Memory: {avg_memory:.1f}%")
    print(f"   Average Latency: {avg_latency:.1f}ms")
    print(f"   Total Metrics Collected: {len(metrics)}")

if __name__ == "__main__":
    monitor_system_resources(300)  # 5 minutes
```

## Phase 5: Resilience Testing

### 5.1 Fault Injection Tests
```bash
#!/bin/bash
# Chaos testing script: /workspace/ext/tests/resilience/chaos_test.sh

echo "🌪️  Starting chaos engineering tests"

# Test 1: Container restart during operation
echo "📋 Test 1: Container restart resilience"
# Spawn some swarms first
for i in {1..3}; do
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "{\"task\":\"Chaos test task $i\",\"method\":\"docker\"}" \
        http://localhost:3000/api/hybrid/spawn-swarm
done

sleep 5

# Restart the container
echo "  Restarting multi-agent-container..."
docker restart multi-agent-container

# Wait for recovery
sleep 15

# Check system status
STATUS=$(curl -s http://localhost:3000/api/hybrid/status)
DOCKER_HEALTH=$(echo $STATUS | jq -r '.dockerHealth')

if [ "$DOCKER_HEALTH" == "healthy" ]; then
    echo "✅ Container restart recovery successful"
else
    echo "❌ Container restart recovery failed: $DOCKER_HEALTH"
fi

# Test 2: Network partition simulation
echo "📋 Test 2: Network partition resilience"
# Block container network access temporarily
docker exec multi-agent-container iptables -A OUTPUT -j DROP
sleep 10

# Restore network
docker exec multi-agent-container iptables -F OUTPUT

# Check recovery
sleep 5
STATUS=$(curl -s http://localhost:3000/api/hybrid/status)
NETWORK_LATENCY=$(echo $STATUS | jq -r '.networkLatency')

if [ "$NETWORK_LATENCY" -lt 5000 ]; then
    echo "✅ Network partition recovery successful"
else
    echo "❌ Network partition recovery failed: ${NETWORK_LATENCY}ms latency"
fi

# Test 3: High load during failure
echo "📋 Test 3: High load with simulated failures"
# Generate load while injecting failures
for i in {1..20}; do
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "{\"task\":\"Load test during chaos $i\",\"method\":\"hybrid\"}" \
        http://localhost:3000/api/hybrid/spawn-swarm &

    # Randomly kill processes
    if [ $((RANDOM % 5)) -eq 0 ]; then
        docker exec multi-agent-container pkill -f "claude-flow" || true
    fi
done

wait

# Check final system state
sleep 10
STATUS=$(curl -s http://localhost:3000/api/hybrid/status)
SYSTEM_STATUS=$(echo $STATUS | jq -r '.systemStatus')

if [ "$SYSTEM_STATUS" != "critical" ]; then
    echo "✅ System survived chaos testing: $SYSTEM_STATUS"
else
    echo "❌ System failed under chaos: $SYSTEM_STATUS"
fi

echo "🏁 Chaos testing completed"
```

### 5.2 Circuit Breaker Testing
```rust
#[tokio::test]
async fn test_circuit_breaker_under_load() {
    let circuit_breaker = CircuitBreaker::new(5, Duration::from_secs(30), 3);

    // Simulate high failure rate
    for _ in 0..10 {
        circuit_breaker.record_failure().await;
    }

    // Circuit should be open
    assert!(!circuit_breaker.can_execute().await);

    // Try to execute operations - should be blocked
    let mut blocked_count = 0;
    for _ in 0..100 {
        if !circuit_breaker.can_execute().await {
            blocked_count += 1;
        }
    }

    assert_eq!(blocked_count, 100);

    // Wait for recovery window
    tokio::time::sleep(Duration::from_secs(31)).await;

    // Should allow some traffic through (half-open)
    assert!(circuit_breaker.can_execute().await);

    // Simulate recovery with successful operations
    for _ in 0..5 {
        circuit_breaker.record_success().await;
    }

    // Should be fully open again
    assert_eq!(circuit_breaker.get_state().await, CircuitState::Closed);
}
```

## Phase 6: Migration Testing

### 6.1 A/B Testing Setup
```yaml
# A/B testing configuration: /workspace/ext/config/ab_test.yaml
ab_test:
  enabled: true
  traffic_split:
    legacy_mcp: 20%      # Route 20% to old MCP-only system
    hybrid: 80%          # Route 80% to new hybrid system

  feature_flags:
    docker_primary: true
    mcp_fallback: true
    performance_monitoring: true
    circuit_breakers: true

  metrics:
    success_rate_threshold: 95%
    latency_threshold_ms: 1000
    error_rate_threshold: 5%

  rollback_triggers:
    - success_rate < 90%
    - average_latency > 2000ms
    - error_rate > 10%
    - container_health == "critical"
```

### 6.2 Migration Rollback Test
```bash
#!/bin/bash
# Rollback test script: /workspace/ext/tests/migration/rollback_test.sh

echo "🔄 Testing migration rollback procedure"

# Step 1: Enable hybrid system
echo "📋 Step 1: Enabling hybrid system"
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"feature":"hybrid_architecture","enabled":true}' \
    http://localhost:3000/api/admin/feature-flags

# Step 2: Run validation tests
echo "📋 Step 2: Running validation tests"
./tests/e2e/hybrid_workflow_test.sh

# Step 3: Simulate failure condition
echo "📋 Step 3: Simulating failure condition"
# Artificially increase error rate
docker exec multi-agent-container rm -f /app/node_modules/.bin/claude-flow

# Step 4: Trigger rollback
echo "📋 Step 4: Triggering rollback"
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"feature":"hybrid_architecture","enabled":false}' \
    http://localhost:3000/api/admin/feature-flags

# Step 5: Validate rollback success
echo "📋 Step 5: Validating rollback"
sleep 5

STATUS=$(curl -s http://localhost:3000/api/hybrid/status)
FAILOVER_ACTIVE=$(echo $STATUS | jq -r '.failoverActive')

if [ "$FAILOVER_ACTIVE" == "true" ]; then
    echo "✅ Rollback successful - system running on MCP fallback"
else
    echo "❌ Rollback failed - system still in hybrid mode"
    exit 1
fi

# Step 6: Restore normal operation
echo "📋 Step 6: Restoring normal operation"
docker exec multi-agent-container ln -s /app/claude-flow/bin/claude-flow /app/node_modules/.bin/claude-flow

curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"feature":"hybrid_architecture","enabled":true}' \
    http://localhost:3000/api/admin/feature-flags

echo "🏁 Rollback test completed successfully"
```

## Test Automation and CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/hybrid-architecture-tests.yml
name: Hybrid Architecture Tests

on:
  pull_request:
    paths:
      - 'src/utils/docker_hive_mind.rs'
      - 'src/utils/hybrid_*.rs'
      - 'src/handlers/hybrid_*.rs'
      - 'client/src/hooks/useHybridSystemStatus.ts'
      - 'client/src/components/HybridSystemDashboard.tsx'

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run unit tests
        run: |
          cd ext
          cargo test docker_hive_mind --lib
          cargo test hybrid_fault_tolerance --lib
          cargo test hybrid_performance_optimizer --lib

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      docker:
        image: docker:dind
        options: --privileged
    steps:
      - uses: actions/checkout@v3

      - name: Setup Docker Compose
        run: |
          docker compose -f docker-compose.test.yml up -d
          sleep 30  # Wait for services to start

      - name: Run integration tests
        run: |
          cd ext
          cargo test test_hybrid_status_endpoint
          cargo test test_swarm_spawning_endpoint
          cargo test test_fallback_behavior

      - name: Run E2E tests
        run: |
          cd ext/tests/e2e
          ./hybrid_workflow_test.sh

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'pull_request' && contains(github.event.pull_request.labels.*.name, 'performance')
    steps:
      - uses: actions/checkout@v3

      - name: Setup test environment
        run: |
          docker compose -f docker-compose.test.yml up -d
          sleep 30

      - name: Run load tests
        run: |
          cd ext/tests/performance
          ./load_test.sh
          python3 resource_monitor.py 180  # 3 minutes

      - name: Upload performance reports
        uses: actions/upload-artifact@v3
        with:
          name: performance-reports
          path: ext/tests/performance/*.json

  chaos-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'pull_request' && contains(github.event.pull_request.labels.*.name, 'resilience')
    steps:
      - uses: actions/checkout@v3

      - name: Setup chaos testing
        run: |
          docker compose -f docker-compose.test.yml up -d
          sleep 30

      - name: Run chaos tests
        run: |
          cd ext/tests/resilience
          ./chaos_test.sh
```

## Success Metrics and Validation Criteria

### Performance Benchmarks
- **Task Spawn Latency**: < 500ms (vs. current 2-5s)
- **Success Rate**: > 95% (vs. current ~60%)
- **Memory Usage**: < 100MB per client (vs. current ~300MB)
- **Network Latency**: < 100ms for status updates
- **Cache Hit Ratio**: > 80%

### Reliability Metrics
- **MTBF**: > 24 hours continuous operation
- **Recovery Time**: < 30 seconds after container restart
- **Failover Success Rate**: > 99%
- **Circuit Breaker Effectiveness**: < 1% false positives

### User Experience Metrics
- **UI Response Time**: < 200ms for status updates
- **WebSocket Connection Stability**: > 99.9% uptime
- **Error Message Clarity**: User-friendly error descriptions
- **Graceful Degradation**: No complete system failures

## Test Environment Setup

### Docker Compose for Testing
```yaml
# docker-compose.test.yml
version: '3.8'

services:
  multi-agent-test:
    build:
      context: .
      dockerfile: Dockerfile.test
    container_name: multi-agent-test-container
    networks:
      - test-network
    environment:
      - RUST_LOG=debug
      - CLAUDE_FLOW_TEST_MODE=true
    volumes:
      - ./tests:/tests
      - /var/run/docker.sock:/var/run/docker.sock

  test-coordinator:
    image: alpine:latest
    container_name: test-coordinator
    networks:
      - test-network
    volumes:
      - ./tests:/tests
    command: /tests/run_all_tests.sh

networks:
  test-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Monitoring and Observability

### Test Metrics Dashboard
```bash
# Set up Grafana dashboard for test metrics
docker run -d \
  --name=test-grafana \
  -p 3001:3000 \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana

# Import test dashboard configuration
curl -X POST \
  http://admin:admin@localhost:3001/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @tests/monitoring/test_dashboard.json
```

## Conclusion

This comprehensive testing strategy ensures the hybrid architecture is thoroughly validated across all dimensions:

1. **Correctness**: Unit and integration tests verify functionality
2. **Performance**: Load tests validate scalability and responsiveness
3. **Reliability**: Resilience tests ensure fault tolerance
4. **Migration Safety**: A/B testing and rollback procedures minimize risk

The testing approach balances thorough validation with practical implementation timelines, providing confidence in the hybrid architecture's production readiness while maintaining the ability to quickly rollback if issues arise.
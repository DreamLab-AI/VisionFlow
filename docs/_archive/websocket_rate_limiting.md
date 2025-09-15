# WebSocket Rate Limiting Implementation

## Overview
This document describes the WebSocket rate limiting implementation for real-time position updates in the VisionFlow system.

## Problem
- Previous rate limit of 60 requests/minute was too restrictive for 5Hz position updates
- Real-time position updates require 300 requests/minute (5 updates/second × 60 seconds)
- Clients were experiencing dropped updates and disconnections
- No proper reconnection state synchronization

## Solution

### 1. Specialized Rate Limit Configuration
Added a new rate limit configuration specifically for WebSocket position updates:

```rust
// In src/utils/validation/rate_limit.rs
pub fn socket_flow_updates() -> RateLimitConfig {
    RateLimitConfig {
        requests_per_minute: 300,  // 5Hz * 60s = 300/min
        burst_size: 50,           // Allow burst of 50 updates
        cleanup_interval: Duration::from_secs(600), // Cleanup every 10 minutes
        ban_duration: Duration::from_secs(600),     // Shorter ban for real-time updates
        max_violations: 10,       // More lenient for real-time data
    }
}
```

### 2. WebSocket Handler Enhancements

#### Rate Limiting Integration
- Added global rate limiter for WebSocket connections using lazy_static
- Check rate limits on connection establishment
- Check rate limits on binary message reception (position updates)
- Graceful handling - warn clients instead of dropping connections

#### Reconnection State Sync
- Track client IP addresses for identifying reconnections
- Detect reconnections via `X-Client-Session` header
- Always send full state sync on connection (new or reconnection)
- Include reconnection status in connection establishment message

#### Client-Side Rate Limit Awareness
- Validate requested update intervals against rate limits
- Automatically adjust intervals if they exceed rate limits
- Send rate limit information to clients in subscription confirmations

### 3. Implementation Details

#### Connection Handling
```rust
// Extract client IP for rate limiting
let client_ip = extract_client_id(&req);

// Check rate limit for WebSocket connections
if !WEBSOCKET_RATE_LIMITER.is_allowed(&client_ip) {
    warn!("WebSocket rate limit exceeded for client: {}", client_ip);
    return create_rate_limit_response(&client_ip, &WEBSOCKET_RATE_LIMITER);
}
```

#### Position Update Subscription
```rust
// Validate interval against rate limits
let min_allowed_interval = 1000 / (EndpointRateLimits::socket_flow_updates().requests_per_minute / 60);
let actual_interval = interval.max(min_allowed_interval as u64);

// Send rate limit info to client
let response = json!({
    "type": "subscription_confirmed",
    "interval": actual_interval,
    "rate_limit": {
        "requests_per_minute": 300,
        "min_interval_ms": min_allowed_interval
    }
});
```

#### State Synchronization
```rust
// Always send full state sync on connection
self.send_full_state_sync(ctx);
self.state_synced = true;

// Include graph state, settings version, and initial positions
let state_sync = json!({
    "type": "state_sync",
    "data": {
        "graph": {
            "nodes_count": graph_data.nodes.len(),
            "edges_count": graph_data.edges.len(),
        },
        "settings": {
            "version": settings.version,
        },
        "timestamp": timestamp,
    }
});
```

## Benefits
1. **Higher Update Frequency**: Supports 5Hz (300/min) position updates
2. **Burst Handling**: Allows bursts of up to 50 updates for smooth interactions
3. **Graceful Degradation**: Warns clients instead of dropping connections
4. **Reconnection Support**: Full state sync ensures consistency after disconnects
5. **Client Awareness**: Clients know rate limits and can adjust behavior

## Testing
Tests are provided in `/workspace/ext/tests/test_websocket_rate_limit.rs` to verify:
- Rate limit configuration values
- Calculation of minimum intervals for 5Hz updates
- Compatibility of 5Hz update rate with rate limits

## Future Improvements
1. Per-client rate limit tracking with different tiers
2. Adaptive rate limiting based on server load
3. WebSocket message queueing for exceeded limits
4. Persistent session tracking across reconnections
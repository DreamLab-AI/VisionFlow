use webxr::utils::validation::rate_limit::{EndpointRateLimits, RateLimitConfig};

#[test]
fn test_socket_flow_updates_rate_limit() {
    // Get the WebSocket rate limit configuration
    let config = EndpointRateLimits::socket_flow_updates();

    // Verify it allows 300 requests per minute for 5Hz updates
    assert_eq!(
        config.requests_per_minute, 300,
        "Should allow 300 requests per minute for 5Hz updates"
    );

    // Verify burst size is appropriate
    assert_eq!(config.burst_size, 50, "Should allow burst of 50 updates");

    // Verify more lenient violation settings
    assert_eq!(
        config.max_violations, 10,
        "Should be more lenient with violations for real-time data"
    );

    println!("WebSocket rate limit configuration verified:");
    println!("  - Requests per minute: {}", config.requests_per_minute);
    println!("  - Burst size: {}", config.burst_size);
    println!("  - Max violations: {}", config.max_violations);
    println!("  - Ban duration: {:?}", config.ban_duration);
}

#[test]
fn test_rate_limit_calculation() {
    // Calculate minimum interval for 5Hz updates
    let requests_per_minute = 300;
    let min_interval_ms = 1000 / (requests_per_minute / 60);

    assert_eq!(
        min_interval_ms, 200,
        "Minimum interval should be 200ms for 5Hz updates"
    );

    // Verify that 5Hz (200ms interval) fits within rate limit
    let updates_per_second = 1000 / 200; // 5 updates per second
    let updates_per_minute = updates_per_second * 60; // 300 updates per minute

    assert!(
        updates_per_minute <= requests_per_minute,
        "5Hz update rate should fit within rate limit"
    );
}

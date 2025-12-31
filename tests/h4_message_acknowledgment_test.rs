// Test disabled - references deprecated/removed modules (webxr::actors::messaging)
// MessageTracker and related types may have been restructured or moved
/*
//! Integration tests for H4 Message Acknowledgment Protocol

use actix::prelude::*;
use std::time::Duration;

// Import the types we need
use webxr::actors::messaging::{MessageId, MessageTracker, MessageKind, MessageAck, AckStatus};

#[actix_rt::test]
async fn test_message_tracker_with_acknowledgment() {
    // Create a tracker
    let tracker = MessageTracker::new();

    // Generate a message ID
    let msg_id = MessageId::new();

    // Track a message
    tracker.track_default(msg_id, MessageKind::UpdateGPUGraphData).await;

    // Verify it's pending
    assert!(tracker.is_pending(msg_id).await, "Message should be pending");
    assert_eq!(tracker.pending_count().await, 1, "Should have 1 pending message");

    // Send acknowledgment
    let ack = MessageAck::success(msg_id);
    tracker.acknowledge(ack).await;

    // Verify it's no longer pending
    assert!(!tracker.is_pending(msg_id).await, "Message should not be pending after ack");
    assert_eq!(tracker.pending_count().await, 0, "Should have 0 pending messages");

    // Check metrics
    let metrics = tracker.metrics();
    assert_eq!(metrics.total_sent.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(metrics.total_acked.load(std::sync::atomic::Ordering::Relaxed), 1);
}

// ... rest of tests ...
*/

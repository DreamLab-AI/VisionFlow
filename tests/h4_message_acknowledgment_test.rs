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

#[actix_rt::test]
async fn test_message_timeout_and_retry() {
    let tracker = MessageTracker::new();
    let msg_id = MessageId::new();

    // Track with very short timeout
    tracker.track(
        msg_id,
        MessageKind::ComputeForces,
        Duration::from_millis(50),
        3
    ).await;

    assert!(tracker.is_pending(msg_id).await);

    // Wait for timeout
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Manually check timeouts
    tracker.check_timeouts().await;

    // Message should still be pending (with retry)
    assert!(tracker.is_pending(msg_id).await, "Message should still be pending after timeout (retry scheduled)");

    // Verify retry was recorded
    let metrics = tracker.metrics();
    assert!(metrics.total_retried.load(std::sync::atomic::Ordering::Relaxed) > 0, "Should have retry recorded");
}

#[actix_rt::test]
async fn test_multiple_message_tracking() {
    let tracker = MessageTracker::new();

    // Track multiple messages
    let msg_id1 = MessageId::new();
    let msg_id2 = MessageId::new();
    let msg_id3 = MessageId::new();

    tracker.track_default(msg_id1, MessageKind::InitializeGPU).await;
    tracker.track_default(msg_id2, MessageKind::UpdateGPUGraphData).await;
    tracker.track_default(msg_id3, MessageKind::UploadConstraintsToGPU).await;

    assert_eq!(tracker.pending_count().await, 3);

    // Acknowledge one
    tracker.acknowledge(MessageAck::success(msg_id2)).await;
    assert_eq!(tracker.pending_count().await, 2);

    // Acknowledge another
    tracker.acknowledge(MessageAck::success(msg_id1)).await;
    assert_eq!(tracker.pending_count().await, 1);

    // The third should still be pending
    assert!(tracker.is_pending(msg_id3).await);

    // Acknowledge the last one
    tracker.acknowledge(MessageAck::success(msg_id3)).await;
    assert_eq!(tracker.pending_count().await, 0);
}

#[actix_rt::test]
async fn test_failed_message_acknowledgment() {
    let tracker = MessageTracker::new();
    let msg_id = MessageId::new();

    tracker.track_default(msg_id, MessageKind::ComputeForces).await;

    // Send failure acknowledgment
    let ack = MessageAck::failure(msg_id, "GPU computation failed".to_string());
    tracker.acknowledge(ack).await;

    // Message should be removed from pending
    assert!(!tracker.is_pending(msg_id).await);

    // Check metrics
    let metrics = tracker.metrics();
    assert_eq!(metrics.total_failed.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[actix_rt::test]
async fn test_message_with_metadata() {
    let tracker = MessageTracker::new();
    let msg_id = MessageId::new();

    tracker.track_default(msg_id, MessageKind::InitializeGPU).await;

    // Send acknowledgment with metadata
    let ack = MessageAck::success(msg_id)
        .with_metadata("nodes", "1000")
        .with_metadata("edges", "5000")
        .with_metadata("processing_time_ms", "42");

    tracker.acknowledge(ack).await;

    assert!(!tracker.is_pending(msg_id).await);
}

#[actix_rt::test]
async fn test_retry_delay_calculation() {
    assert_eq!(
        MessageTracker::calculate_retry_delay(1),
        Duration::from_millis(100)
    );
    assert_eq!(
        MessageTracker::calculate_retry_delay(2),
        Duration::from_millis(200)
    );
    assert_eq!(
        MessageTracker::calculate_retry_delay(3),
        Duration::from_millis(400)
    );
    assert_eq!(
        MessageTracker::calculate_retry_delay(4),
        Duration::from_millis(800)
    );

    // Test capping at 30 seconds
    assert_eq!(
        MessageTracker::calculate_retry_delay(20),
        Duration::from_secs(30)
    );
}

#[actix_rt::test]
async fn test_metrics_summary() {
    let tracker = MessageTracker::new();

    // Send several messages
    for i in 0..5 {
        let msg_id = MessageId::new();
        tracker.track_default(msg_id, MessageKind::UpdateGPUGraphData).await;

        // Acknowledge 4 out of 5
        if i < 4 {
            tracker.acknowledge(MessageAck::success(msg_id)).await;
        }
    }

    // Wait for async metric updates
    tokio::time::sleep(Duration::from_millis(20)).await;

    let summary = tracker.metrics().summary().await;
    assert_eq!(summary.total_sent, 5);
    assert_eq!(summary.total_acked, 4);
    assert!(summary.overall_success_rate > 0.5); // At least 50% success rate
}

#[test]
fn test_message_kind_defaults() {
    assert_eq!(
        MessageKind::InitializeGPU.default_timeout(),
        Duration::from_secs(10)
    );
    assert_eq!(MessageKind::InitializeGPU.default_max_retries(), 5);

    assert_eq!(
        MessageKind::UpdateNodePositions.default_timeout(),
        Duration::from_secs(1)
    );
    assert_eq!(MessageKind::UpdateNodePositions.default_max_retries(), 2);

    assert_eq!(MessageKind::ComputeForces.name(), "ComputeForces");
}

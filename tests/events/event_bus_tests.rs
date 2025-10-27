use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use chrono::Utc;
use std::collections::HashMap;

use visionflow::events::*;

#[tokio::test]
async fn test_event_bus_basic_publish_subscribe() {
    let bus = EventBus::new();
    let call_count = Arc::new(AtomicUsize::new(0));

    // Create a simple test handler
    struct TestHandler {
        id: String,
        count: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl EventHandler for TestHandler {
        fn event_type(&self) -> &'static str {
            "NodeAdded"
        }

        fn handler_id(&self) -> &str {
            &self.id
        }

        async fn handle(&self, _event: &StoredEvent) -> Result<(), EventError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let handler = Arc::new(TestHandler {
        id: "test-handler".to_string(),
        count: call_count.clone(),
    });

    bus.subscribe(handler).await;

    let event = NodeAddedEvent {
        node_id: "node-1".to_string(),
        label: "Test Node".to_string(),
        node_type: "Person".to_string(),
        properties: HashMap::new(),
        timestamp: Utc::now(),
    };

    bus.publish(event).await.unwrap();

    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_multiple_subscribers() {
    let bus = EventBus::new();
    let count1 = Arc::new(AtomicUsize::new(0));
    let count2 = Arc::new(AtomicUsize::new(0));
    let count3 = Arc::new(AtomicUsize::new(0));

    struct TestHandler {
        id: String,
        count: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl EventHandler for TestHandler {
        fn event_type(&self) -> &'static str {
            "NodeAdded"
        }

        fn handler_id(&self) -> &str {
            &self.id
        }

        async fn handle(&self, _event: &StoredEvent) -> Result<(), EventError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    bus.subscribe(Arc::new(TestHandler {
        id: "handler-1".to_string(),
        count: count1.clone(),
    })).await;

    bus.subscribe(Arc::new(TestHandler {
        id: "handler-2".to_string(),
        count: count2.clone(),
    })).await;

    bus.subscribe(Arc::new(TestHandler {
        id: "handler-3".to_string(),
        count: count3.clone(),
    })).await;

    let event = NodeAddedEvent {
        node_id: "node-1".to_string(),
        label: "Test".to_string(),
        node_type: "Person".to_string(),
        properties: HashMap::new(),
        timestamp: Utc::now(),
    };

    bus.publish(event).await.unwrap();

    assert_eq!(count1.load(Ordering::SeqCst), 1);
    assert_eq!(count2.load(Ordering::SeqCst), 1);
    assert_eq!(count3.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_event_ordering() {
    let bus = EventBus::new();
    let processed_events = Arc::new(tokio::sync::Mutex::new(Vec::new()));

    struct OrderHandler {
        id: String,
        events: Arc<tokio::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl EventHandler for OrderHandler {
        fn event_type(&self) -> &'static str {
            "NodeAdded"
        }

        fn handler_id(&self) -> &str {
            &self.id
        }

        async fn handle(&self, event: &StoredEvent) -> Result<(), EventError> {
            let mut events = self.events.lock().await;
            events.push(event.metadata.aggregate_id.clone());
            Ok(())
        }
    }

    bus.subscribe(Arc::new(OrderHandler {
        id: "order-handler".to_string(),
        events: processed_events.clone(),
    })).await;

    // Publish multiple events
    for i in 0..10 {
        let event = NodeAddedEvent {
            node_id: format!("node-{}", i),
            label: format!("Node {}", i),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: Utc::now(),
        };
        bus.publish(event).await.unwrap();
    }

    let events = processed_events.lock().await;
    assert_eq!(events.len(), 10);
    for i in 0..10 {
        assert_eq!(events[i], format!("node-{}", i));
    }
}

#[tokio::test]
async fn test_handler_error_isolation() {
    let bus = EventBus::new();
    let success_count = Arc::new(AtomicUsize::new(0));

    struct FailingHandler {
        id: String,
    }

    #[async_trait::async_trait]
    impl EventHandler for FailingHandler {
        fn event_type(&self) -> &'static str {
            "NodeAdded"
        }

        fn handler_id(&self) -> &str {
            &self.id
        }

        async fn handle(&self, _event: &StoredEvent) -> Result<(), EventError> {
            Err(EventError::Handler("Intentional failure".to_string()))
        }
    }

    struct SuccessHandler {
        id: String,
        count: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl EventHandler for SuccessHandler {
        fn event_type(&self) -> &'static str {
            "NodeAdded"
        }

        fn handler_id(&self) -> &str {
            &self.id
        }

        async fn handle(&self, _event: &StoredEvent) -> Result<(), EventError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    // Subscribe failing handler first
    bus.subscribe(Arc::new(FailingHandler {
        id: "failing-handler".to_string(),
    })).await;

    // Subscribe success handler second
    bus.subscribe(Arc::new(SuccessHandler {
        id: "success-handler".to_string(),
        count: success_count.clone(),
    })).await;

    let event = NodeAddedEvent {
        node_id: "node-1".to_string(),
        label: "Test".to_string(),
        node_type: "Person".to_string(),
        properties: HashMap::new(),
        timestamp: Utc::now(),
    };

    // Publish should succeed even though one handler fails
    let result = bus.publish(event).await;
    assert!(result.is_ok());

    // Success handler should still have been called
    assert_eq!(success_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_middleware_pipeline() {
    let bus = EventBus::new();
    let metrics = Arc::new(MetricsMiddleware::new());
    let validation = Arc::new(ValidationMiddleware::new());

    bus.add_middleware(validation.clone()).await;
    bus.add_middleware(metrics.clone()).await;

    struct DummyHandler {
        id: String,
    }

    #[async_trait::async_trait]
    impl EventHandler for DummyHandler {
        fn event_type(&self) -> &'static str {
            "NodeAdded"
        }

        fn handler_id(&self) -> &str {
            &self.id
        }

        async fn handle(&self, _event: &StoredEvent) -> Result<(), EventError> {
            Ok(())
        }
    }

    bus.subscribe(Arc::new(DummyHandler {
        id: "dummy".to_string(),
    })).await;

    let event = NodeAddedEvent {
        node_id: "node-1".to_string(),
        label: "Test".to_string(),
        node_type: "Person".to_string(),
        properties: HashMap::new(),
        timestamp: Utc::now(),
    };

    bus.publish(event).await.unwrap();

    assert_eq!(metrics.get_published_count("NodeAdded").await, 1);
    assert_eq!(metrics.get_handler_count("dummy").await, 1);
}

#[tokio::test]
async fn test_event_bus_can_be_disabled() {
    let bus = EventBus::new();
    let call_count = Arc::new(AtomicUsize::new(0));

    struct TestHandler {
        id: String,
        count: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl EventHandler for TestHandler {
        fn event_type(&self) -> &'static str {
            "NodeAdded"
        }

        fn handler_id(&self) -> &str {
            &self.id
        }

        async fn handle(&self, _event: &StoredEvent) -> Result<(), EventError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    bus.subscribe(Arc::new(TestHandler {
        id: "test".to_string(),
        count: call_count.clone(),
    })).await;

    // Disable the bus
    bus.set_enabled(false).await;

    let event = NodeAddedEvent {
        node_id: "node-1".to_string(),
        label: "Test".to_string(),
        node_type: "Person".to_string(),
        properties: HashMap::new(),
        timestamp: Utc::now(),
    };

    bus.publish(event).await.unwrap();

    // Handler should not have been called
    assert_eq!(call_count.load(Ordering::SeqCst), 0);
}

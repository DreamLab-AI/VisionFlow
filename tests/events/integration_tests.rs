use std::sync::Arc;
use chrono::Utc;
use std::collections::HashMap;

use webxr::events::*;

#[tokio::test]
async fn test_graph_event_handler_integration() {
    let bus = EventBus::new();
    let graph_handler = Arc::new(GraphEventHandler::new("graph-handler"));

    bus.subscribe(graph_handler.clone()).await;

    // Add nodes
    for i in 0..5 {
        let event = NodeAddedEvent {
            node_id: format!("node-{}", i),
            label: format!("Node {}", i),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: Utc::now(),
        };
        bus.publish(event).await.unwrap();
    }

    assert_eq!(graph_handler.get_node_count().await, 5);

    // Add edges
    for i in 0..3 {
        let event = EdgeAddedEvent {
            edge_id: format!("edge-{}", i),
            source_id: format!("node-{}", i),
            target_id: format!("node-{}", i + 1),
            edge_type: "knows".to_string(),
            weight: 1.0,
            timestamp: Utc::now(),
        };
        bus.publish(event).await.unwrap();
    }

    assert_eq!(graph_handler.get_edge_count().await, 3);

    // Remove a node
    let remove_event = NodeRemovedEvent {
        node_id: "node-0".to_string(),
        timestamp: Utc::now(),
    };
    bus.publish(remove_event).await.unwrap();

    assert_eq!(graph_handler.get_node_count().await, 4);
}

#[tokio::test]
async fn test_ontology_event_handler_integration() {
    let bus = EventBus::new();
    let ontology_handler = Arc::new(OntologyEventHandler::new("ontology-handler"));

    bus.subscribe(ontology_handler.clone()).await;

    // Add classes
    for i in 0..10 {
        let event = ClassAddedEvent {
            class_id: format!("class-{}", i),
            class_iri: format!("http://example.org/Class{}", i),
            label: Some(format!("Class {}", i)),
            parent_classes: vec![],
            timestamp: Utc::now(),
        };
        bus.publish(event).await.unwrap();
    }

    assert_eq!(ontology_handler.get_class_count().await, 10);
    assert!(ontology_handler.is_inference_pending().await);

    // Complete inference
    let inference_event = InferenceCompletedEvent {
        ontology_id: "onto-1".to_string(),
        reasoner_type: "HermiT".to_string(),
        inferred_axioms: 50,
        duration_ms: 500,
        timestamp: Utc::now(),
    };
    bus.publish(inference_event).await.unwrap();

    assert!(!ontology_handler.is_inference_pending().await);
    assert_eq!(ontology_handler.get_last_inference_duration().await, Some(500));
}

#[tokio::test]
async fn test_audit_handler_integration() {
    let bus = EventBus::new();
    let audit_handler = Arc::new(AuditEventHandler::new("audit-handler"));

    bus.subscribe(audit_handler.clone()).await;

    // Publish various events
    let node_event = NodeAddedEvent {
        node_id: "node-1".to_string(),
        label: "Test".to_string(),
        node_type: "Person".to_string(),
        properties: HashMap::new(),
        timestamp: Utc::now(),
    };
    bus.publish(node_event).await.unwrap();

    let edge_event = EdgeAddedEvent {
        edge_id: "edge-1".to_string(),
        source_id: "node-1".to_string(),
        target_id: "node-2".to_string(),
        edge_type: "knows".to_string(),
        weight: 1.0,
        timestamp: Utc::now(),
    };
    bus.publish(edge_event).await.unwrap();

    assert_eq!(audit_handler.get_log_count().await, 2);

    let entries = audit_handler.get_entries_by_type("NodeAdded").await;
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].aggregate_id, "node-1");
}

#[tokio::test]
async fn test_notification_handler_integration() {
    let bus = EventBus::new();
    let notification_handler = Arc::new(NotificationEventHandler::new("notification-handler"));

    bus.subscribe(notification_handler.clone()).await;

    // Publish events
    let event = SimulationStartedEvent {
        simulation_id: "sim-1".to_string(),
        physics_profile: "force-directed".to_string(),
        node_count: 100,
        timestamp: Utc::now(),
    };
    bus.publish(event).await.unwrap();

    let notifications = notification_handler.get_unsent_notifications().await;
    assert_eq!(notifications.len(), 1);
    assert!(notifications[0].message.contains("sim-1"));

    // Mark as sent
    notification_handler.mark_sent(&notifications[0].notification_id).await;
    assert_eq!(notification_handler.get_unsent_notifications().await.len(), 0);
}

#[tokio::test]
async fn test_multiple_handlers_working_together() {
    let bus = EventBus::new();

    let graph_handler = Arc::new(GraphEventHandler::new("graph"));
    let audit_handler = Arc::new(AuditEventHandler::new("audit"));
    let notification_handler = Arc::new(NotificationEventHandler::new("notification"));

    bus.subscribe(graph_handler.clone()).await;
    bus.subscribe(audit_handler.clone()).await;
    bus.subscribe(notification_handler.clone()).await;

    // Publish event
    let event = NodeAddedEvent {
        node_id: "node-1".to_string(),
        label: "Test Node".to_string(),
        node_type: "Person".to_string(),
        properties: HashMap::new(),
        timestamp: Utc::now(),
    };
    bus.publish(event).await.unwrap();

    // All handlers should have processed the event
    assert_eq!(graph_handler.get_node_count().await, 1);
    assert_eq!(audit_handler.get_log_count().await, 1);
    assert_eq!(notification_handler.get_notifications().await.len(), 1);
}

#[tokio::test]
async fn test_event_store_integration() {
    let repo = Arc::new(InMemoryEventRepository::new());
    let store = EventStore::new(repo.clone());

    // Store events
    for i in 0..5 {
        let event = NodeAddedEvent {
            node_id: format!("node-{}", i),
            label: format!("Node {}", i),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: Utc::now(),
        };
        store.append(&event).await.unwrap();
    }

    // Retrieve all events for a specific node
    let events = store.get_events("node-2").await.unwrap();
    assert_eq!(events.len(), 1);

    // Get events by type
    let node_events = store.get_events_by_type("NodeAdded").await.unwrap();
    assert_eq!(node_events.len(), 5);

    // Replay events
    let replayed = store.replay_events("node-3").await.unwrap();
    assert_eq!(replayed.len(), 1);
}

#[tokio::test]
async fn test_event_bus_with_middleware() {
    let bus = EventBus::new();

    let metrics = Arc::new(MetricsMiddleware::new());
    let validation = Arc::new(ValidationMiddleware::new());
    let logging = Arc::new(LoggingMiddleware::new(false));

    bus.add_middleware(validation).await;
    bus.add_middleware(metrics.clone()).await;
    bus.add_middleware(logging).await;

    let graph_handler = Arc::new(GraphEventHandler::new("graph"));
    bus.subscribe(graph_handler.clone()).await;

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

    assert_eq!(metrics.get_published_count("NodeAdded").await, 10);
    assert_eq!(graph_handler.get_node_count().await, 10);
}

#[tokio::test]
async fn test_complete_event_flow() {
    // Setup
    let bus = Arc::new(EventBus::new());
    let repo = Arc::new(InMemoryEventRepository::new());
    let store = Arc::new(EventStore::new(repo.clone()));

    // Handlers
    let graph_handler = Arc::new(GraphEventHandler::new("graph"));
    let ontology_handler = Arc::new(OntologyEventHandler::new("ontology"));
    let audit_handler = Arc::new(AuditEventHandler::new("audit"));
    let notification_handler = Arc::new(NotificationEventHandler::new("notification"));

    bus.subscribe(graph_handler.clone()).await;
    bus.subscribe(ontology_handler.clone()).await;
    bus.subscribe(audit_handler.clone()).await;
    bus.subscribe(notification_handler.clone()).await;

    // Middleware
    let metrics = Arc::new(MetricsMiddleware::new());
    bus.add_middleware(metrics.clone()).await;

    // Simulate a workflow
    // 1. Add nodes
    for i in 0..3 {
        let event = NodeAddedEvent {
            node_id: format!("node-{}", i),
            label: format!("Node {}", i),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: Utc::now(),
        };
        bus.publish(event.clone()).await.unwrap();
        store.append(&event).await.unwrap();
    }

    // 2. Add edges
    for i in 0..2 {
        let event = EdgeAddedEvent {
            edge_id: format!("edge-{}", i),
            source_id: format!("node-{}", i),
            target_id: format!("node-{}", i + 1),
            edge_type: "knows".to_string(),
            weight: 1.0,
            timestamp: Utc::now(),
        };
        bus.publish(event.clone()).await.unwrap();
        store.append(&event).await.unwrap();
    }

    // 3. Import ontology
    let onto_event = OntologyImportedEvent {
        ontology_id: "onto-1".to_string(),
        file_path: "/test.owl".to_string(),
        format: "RDF/XML".to_string(),
        class_count: 50,
        property_count: 25,
        individual_count: 100,
        timestamp: Utc::now(),
    };
    bus.publish(onto_event.clone()).await.unwrap();
    store.append(&onto_event).await.unwrap();

    // Verify results
    assert_eq!(graph_handler.get_node_count().await, 3);
    assert_eq!(graph_handler.get_edge_count().await, 2);
    assert_eq!(ontology_handler.get_class_count().await, 50);
    assert_eq!(audit_handler.get_log_count().await, 6);
    assert_eq!(notification_handler.get_notifications().await.len(), 6);
    assert_eq!(metrics.get_published_count("NodeAdded").await, 3);
    assert_eq!(metrics.get_published_count("EdgeAdded").await, 2);
}

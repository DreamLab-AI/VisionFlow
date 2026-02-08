use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::events::types::{
    DomainEvent, EventError, EventHandler, EventMetadata, EventMiddleware, EventResult, StoredEvent,
};

pub struct EventBus {
    
    subscribers: Arc<RwLock<HashMap<String, Vec<Arc<dyn EventHandler>>>>>,

    
    middleware: Arc<RwLock<Vec<Arc<dyn EventMiddleware>>>>,

    
    sequence: Arc<RwLock<i64>>,

    
    enabled: Arc<RwLock<bool>>,
}

impl EventBus {
    
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            middleware: Arc::new(RwLock::new(Vec::new())),
            sequence: Arc::new(RwLock::new(0)),
            enabled: Arc::new(RwLock::new(true)),
        }
    }

    
    pub async fn publish<E: DomainEvent>(&self, event: E) -> EventResult<()> {
        
        if !*self.enabled.read().await {
            return Ok(());
        }

        
        let metadata = EventMetadata::new(
            event.aggregate_id().to_string(),
            event.aggregate_type().to_string(),
            event.event_type().to_string(),
        );

        
        let data = event.to_json_string().map_err(|e| EventError::Serialization(e.to_string()))?;

        
        let mut seq = self.sequence.write().await;
        *seq += 1;
        let sequence = *seq;
        drop(seq);

        
        let mut stored_event = StoredEvent {
            metadata,
            data,
            sequence,
        };

        
        let middleware = self.middleware.read().await;
        for mw in middleware.iter() {
            mw.before_publish(&mut stored_event).await?;
        }
        drop(middleware);

        
        let subscribers = self.subscribers.read().await;
        let mut handlers = subscribers
            .get(event.event_type())
            .cloned()
            .unwrap_or_default();
        // Also dispatch to wildcard ("*") handlers (audit, notification, etc.)
        if event.event_type() != "*" {
            if let Some(wildcard_handlers) = subscribers.get("*") {
                handlers.extend(wildcard_handlers.iter().cloned());
            }
        }
        drop(subscribers);

        
        let handler_count = handlers.len();
        let mut errors = Vec::new();
        for handler in handlers {
            match self.execute_handler(handler.clone(), &stored_event).await {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Handler {} failed: {}", handler.handler_id(), e);
                    errors.push((handler.handler_id().to_string(), e));
                }
            }
        }

        
        let middleware = self.middleware.read().await;
        for mw in middleware.iter() {
            mw.after_publish(&stored_event).await?;
        }
        drop(middleware);

        
        if !errors.is_empty() && errors.len() == handler_count {
            return Err(EventError::Handler(format!(
                "All {} handlers failed",
                errors.len()
            )));
        }

        Ok(())
    }

    
    pub async fn subscribe(&self, handler: Arc<dyn EventHandler>) {
        let event_type = handler.event_type().to_string();
        let mut subscribers = self.subscribers.write().await;

        subscribers
            .entry(event_type)
            .or_insert_with(Vec::new)
            .push(handler);
    }

    
    pub async fn unsubscribe(&self, handler_id: &str, event_type: &str) {
        let mut subscribers = self.subscribers.write().await;

        if let Some(handlers) = subscribers.get_mut(event_type) {
            handlers.retain(|h| h.handler_id() != handler_id);
        }
    }

    
    pub async fn add_middleware(&self, middleware: Arc<dyn EventMiddleware>) {
        let mut mw_list = self.middleware.write().await;
        mw_list.push(middleware);
    }

    
    async fn execute_handler(
        &self,
        handler: Arc<dyn EventHandler>,
        event: &StoredEvent,
    ) -> EventResult<()> {
        let handler_id = handler.handler_id();
        let max_retries = handler.max_retries();

        
        let middleware = self.middleware.read().await;
        for mw in middleware.iter() {
            mw.before_handle(event, handler_id).await?;
        }
        drop(middleware);

        
        let mut last_error = None;
        for attempt in 0..=max_retries {
            match handler.handle(event).await {
                Ok(_) => {
                    
                    let middleware = self.middleware.read().await;
                    for mw in middleware.iter() {
                        mw.after_handle(event, handler_id, &Ok(())).await?;
                    }
                    return Ok(());
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < max_retries {
                        
                        let delay = std::time::Duration::from_millis(100 * 2_u64.pow(attempt));
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        
        let error = last_error.unwrap();
        let result = Err(error.clone());

        
        let middleware = self.middleware.read().await;
        for mw in middleware.iter() {
            let _ = mw.after_handle(event, handler_id, &result).await;
        }

        Err(error)
    }

    
    pub async fn subscriber_count(&self, event_type: &str) -> usize {
        let subscribers = self.subscribers.read().await;
        subscribers.get(event_type).map(|h| h.len()).unwrap_or(0)
    }

    
    pub async fn clear_subscribers(&self) {
        let mut subscribers = self.subscribers.write().await;
        subscribers.clear();
    }

    
    pub async fn set_enabled(&self, enabled: bool) {
        let mut flag = self.enabled.write().await;
        *flag = enabled;
    }

    
    pub async fn is_enabled(&self) -> bool {
        *self.enabled.read().await
    }

    
    pub async fn current_sequence(&self) -> i64 {
        *self.sequence.read().await
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use crate::events::domain_events::NodeAddedEvent;
    use std::sync::atomic::{AtomicUsize, Ordering};
use crate::utils::time;

    struct TestHandler {
        id: String,
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl EventHandler for TestHandler {
        fn event_type(&self) -> &'static str {
            "NodeAdded"
        }

        fn handler_id(&self) -> &str {
            &self.id
        }

        async fn handle(&self, _event: &StoredEvent) -> Result<(), EventError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_event_bus_publish() {
        let bus = EventBus::new();
        let call_count = Arc::new(AtomicUsize::new(0));

        let handler = Arc::new(TestHandler {
            id: "test-handler".to_string(),
            call_count: call_count.clone(),
        });

        bus.subscribe(handler).await;

        let event = NodeAddedEvent {
            node_id: "node-1".to_string(),
            label: "Test".to_string(),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: time::now(),
        };

        bus.publish(event).await.unwrap();

        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let bus = EventBus::new();
        let count1 = Arc::new(AtomicUsize::new(0));
        let count2 = Arc::new(AtomicUsize::new(0));

        bus.subscribe(Arc::new(TestHandler {
            id: "handler-1".to_string(),
            call_count: count1.clone(),
        }))
        .await;

        bus.subscribe(Arc::new(TestHandler {
            id: "handler-2".to_string(),
            call_count: count2.clone(),
        }))
        .await;

        let event = NodeAddedEvent {
            node_id: "node-1".to_string(),
            label: "Test".to_string(),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: time::now(),
        };

        bus.publish(event).await.unwrap();

        assert_eq!(count1.load(Ordering::SeqCst), 1);
        assert_eq!(count2.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_unsubscribe() {
        let bus = EventBus::new();
        let call_count = Arc::new(AtomicUsize::new(0));

        let handler = Arc::new(TestHandler {
            id: "test-handler".to_string(),
            call_count: call_count.clone(),
        });

        bus.subscribe(handler).await;
        assert_eq!(bus.subscriber_count("NodeAdded").await, 1);

        bus.unsubscribe("test-handler", "NodeAdded").await;
        assert_eq!(bus.subscriber_count("NodeAdded").await, 0);
    }

    #[tokio::test]
    async fn test_disabled_bus() {
        let bus = EventBus::new();
        bus.set_enabled(false).await;

        let call_count = Arc::new(AtomicUsize::new(0));
        bus.subscribe(Arc::new(TestHandler {
            id: "test-handler".to_string(),
            call_count: call_count.clone(),
        }))
        .await;

        let event = NodeAddedEvent {
            node_id: "node-1".to_string(),
            label: "Test".to_string(),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: time::now(),
        };

        bus.publish(event).await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 0);
    }
}

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::events::types::{
    DomainEvent, EventError, EventMetadata, EventResult, EventSnapshot, StoredEvent,
};

///
#[async_trait]
pub trait EventRepository: Send + Sync {
    
    async fn append(&self, event: StoredEvent) -> EventResult<()>;

    
    async fn get_events(&self, aggregate_id: &str) -> EventResult<Vec<StoredEvent>>;

    
    async fn get_events_after(&self, sequence: i64) -> EventResult<Vec<StoredEvent>>;

    
    async fn get_events_by_type(&self, event_type: &str) -> EventResult<Vec<StoredEvent>>;

    
    async fn save_snapshot(&self, snapshot: EventSnapshot) -> EventResult<()>;

    
    async fn get_snapshot(&self, aggregate_id: &str) -> EventResult<Option<EventSnapshot>>;

    
    async fn get_event_count(&self, aggregate_id: &str) -> EventResult<usize>;
}

///
pub struct InMemoryEventRepository {
    events: Arc<RwLock<Vec<StoredEvent>>>,
    snapshots: Arc<RwLock<HashMap<String, EventSnapshot>>>,
}

impl InMemoryEventRepository {
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn clear(&self) {
        self.events.write().await.clear();
        self.snapshots.write().await.clear();
    }

    pub async fn event_count(&self) -> usize {
        self.events.read().await.len()
    }
}

impl Default for InMemoryEventRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EventRepository for InMemoryEventRepository {
    async fn append(&self, event: StoredEvent) -> EventResult<()> {
        let mut events = self.events.write().await;
        events.push(event);
        Ok(())
    }

    async fn get_events(&self, aggregate_id: &str) -> EventResult<Vec<StoredEvent>> {
        let events = self.events.read().await;
        Ok(events
            .iter()
            .filter(|e| e.metadata.aggregate_id == aggregate_id)
            .cloned()
            .collect())
    }

    async fn get_events_after(&self, sequence: i64) -> EventResult<Vec<StoredEvent>> {
        let events = self.events.read().await;
        Ok(events
            .iter()
            .filter(|e| e.sequence > sequence)
            .cloned()
            .collect())
    }

    async fn get_events_by_type(&self, event_type: &str) -> EventResult<Vec<StoredEvent>> {
        let events = self.events.read().await;
        Ok(events
            .iter()
            .filter(|e| e.metadata.event_type == event_type)
            .cloned()
            .collect())
    }

    async fn save_snapshot(&self, snapshot: EventSnapshot) -> EventResult<()> {
        let mut snapshots = self.snapshots.write().await;
        snapshots.insert(snapshot.aggregate_id.clone(), snapshot);
        Ok(())
    }

    async fn get_snapshot(&self, aggregate_id: &str) -> EventResult<Option<EventSnapshot>> {
        let snapshots = self.snapshots.read().await;
        Ok(snapshots.get(aggregate_id).cloned())
    }

    async fn get_event_count(&self, aggregate_id: &str) -> EventResult<usize> {
        let events = self.get_events(aggregate_id).await?;
        Ok(events.len())
    }
}

///
pub struct EventStore {
    repo: Arc<dyn EventRepository>,
    snapshot_threshold: usize,
}

impl EventStore {
    
    pub fn new(repo: Arc<dyn EventRepository>) -> Self {
        Self {
            repo,
            snapshot_threshold: 100, 
        }
    }

    
    pub fn with_snapshot_threshold(mut self, threshold: usize) -> Self {
        self.snapshot_threshold = threshold;
        self
    }

    
    pub async fn append(&self, event: &dyn DomainEvent) -> EventResult<()> {
        let metadata = EventMetadata::new(
            event.aggregate_id().to_string(),
            event.aggregate_type().to_string(),
            event.event_type().to_string(),
        );

        let data = event.to_json_string().map_err(|e| EventError::Serialization(e.to_string()))?;

        let stored_event = StoredEvent {
            metadata,
            data,
            sequence: 0, 
        };

        self.repo.append(stored_event).await?;

        
        let count = self.repo.get_event_count(event.aggregate_id()).await?;
        if count % self.snapshot_threshold == 0 {
            
            
        }

        Ok(())
    }

    
    pub async fn get_events(&self, aggregate_id: &str) -> EventResult<Vec<StoredEvent>> {
        self.repo.get_events(aggregate_id).await
    }

    
    pub async fn get_events_after(&self, sequence: i64) -> EventResult<Vec<StoredEvent>> {
        self.repo.get_events_after(sequence).await
    }

    
    pub async fn get_events_by_type(&self, event_type: &str) -> EventResult<Vec<StoredEvent>> {
        self.repo.get_events_by_type(event_type).await
    }

    
    pub async fn replay_events(&self, aggregate_id: &str) -> EventResult<Vec<StoredEvent>> {
        
        if let Some(snapshot) = self.repo.get_snapshot(aggregate_id).await? {
            
            let events = self.get_events(aggregate_id).await?;
            Ok(events
                .into_iter()
                .filter(|e| e.sequence > snapshot.sequence)
                .collect())
        } else {
            
            self.get_events(aggregate_id).await
        }
    }

    
    pub async fn get_event_count(&self, aggregate_id: &str) -> EventResult<usize> {
        self.repo.get_event_count(aggregate_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::events::domain_events::NodeAddedEvent;
    use std::collections::HashMap;
use crate::utils::time;

    #[tokio::test]
    async fn test_in_memory_repository() {
        let repo = InMemoryEventRepository::new();

        let event = StoredEvent {
            metadata: EventMetadata::new(
                "node-1".to_string(),
                "Node".to_string(),
                "NodeAdded".to_string(),
            ),
            data: "{}".to_string(),
            sequence: 1,
        };

        repo.append(event.clone()).await.unwrap();

        let events = repo.get_events("node-1").await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].metadata.aggregate_id, "node-1");
    }

    #[tokio::test]
    async fn test_event_store() {
        let repo = Arc::new(InMemoryEventRepository::new());
        let store = EventStore::new(repo);

        let event = NodeAddedEvent {
            node_id: "node-1".to_string(),
            label: "Test".to_string(),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: time::now(),
        };

        store.append(&event).await.unwrap();

        let events = store.get_events("node-1").await.unwrap();
        assert_eq!(events.len(), 1);
    }

    #[tokio::test]
    async fn test_get_events_after() {
        let repo = Arc::new(InMemoryEventRepository::new());
        let store = EventStore::new(repo.clone());

        
        for i in 0..5 {
            let event = NodeAddedEvent {
                node_id: format!("node-{}", i),
                label: "Test".to_string(),
                node_type: "Person".to_string(),
                properties: HashMap::new(),
                timestamp: time::now(),
            };
            store.append(&event).await.unwrap();
        }

        let events = store.get_events_after(2).await.unwrap();
        assert!(events.len() > 0);
    }

    #[tokio::test]
    async fn test_get_events_by_type() {
        let repo = Arc::new(InMemoryEventRepository::new());
        let store = EventStore::new(repo);

        let event = NodeAddedEvent {
            node_id: "node-1".to_string(),
            label: "Test".to_string(),
            node_type: "Person".to_string(),
            properties: HashMap::new(),
            timestamp: time::now(),
        };

        store.append(&event).await.unwrap();

        let events = store.get_events_by_type("NodeAdded").await.unwrap();
        assert_eq!(events.len(), 1);
    }
}

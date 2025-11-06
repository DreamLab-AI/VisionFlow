//! Unique message identifiers for tracking

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Unique identifier for tracking messages through the actor system
///
/// # Example
/// ```
/// use webxr::actors::messaging::MessageId;
///
/// let msg_id = MessageId::new();
/// println!("Tracking message: {}", msg_id);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(Uuid);

impl MessageId {
    /// Generate a new unique message identifier
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the inner UUID
    pub fn into_inner(self) -> Uuid {
        self.0
    }

    /// Get a reference to the inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for MessageId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<MessageId> for Uuid {
    fn from(msg_id: MessageId) -> Self {
        msg_id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_id_uniqueness() {
        let id1 = MessageId::new();
        let id2 = MessageId::new();
        assert_ne!(id1, id2, "MessageIds should be unique");
    }

    #[test]
    fn test_message_id_serialization() {
        let id = MessageId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: MessageId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_message_id_display() {
        let id = MessageId::new();
        let display = format!("{}", id);
        assert!(!display.is_empty());
        assert!(display.contains('-')); // UUID format includes dashes
    }
}

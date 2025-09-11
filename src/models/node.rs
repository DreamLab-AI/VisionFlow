use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::types::vec3::Vec3Data;
use crate::config::dev_config;

// Static counter for generating unique numeric IDs
static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(1);  // Start from 1 (0 could be reserved)

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    // Core data
    pub id: u32,
    pub metadata_id: String,  // Store the original filename for lookup
    pub label: String,
    pub data: BinaryNodeData,

    // Metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    #[serde(skip)]
    pub file_size: u64,

    // Rendering properties
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_data: Option<HashMap<String, String>>,
}

impl Node {
    pub fn new(metadata_id: String) -> Self {
        Self::new_with_id(metadata_id, None)
    }

    pub fn new_with_id(metadata_id: String, provided_id: Option<u32>) -> Self {
        // Always generate a new ID on the server side
        // Use provided ID only if it's valid (non-zero)
        let id = match provided_id {
            Some(id) if id != 0 => {
                // Use the provided ID only if it's a valid non-zero ID
                id
            },
            _ => {
                NEXT_NODE_ID.fetch_add(1, Ordering::SeqCst)
            }
        };
        
        // BREADCRUMB: Use node ID to generate deterministic but spread out initial positions
        // This prevents all nodes from starting at the origin which causes clustering
        let id_hash = id as f32;
        let physics = dev_config::physics();
        
        // Use golden ratio for even distribution in 3D space
        let theta = 2.0 * std::f32::consts::PI * ((id_hash * physics.golden_ratio) % 1.0);
        
        // FIXED: Better phi calculation to avoid extreme z-coordinates
        // Map id_hash to [0, 1] range then to [0, PI] for uniform sphere distribution
        let normalized_id = (id_hash * 0.1618) % 1.0; // Use golden ratio fragment for better distribution
        let phi = normalized_id * std::f32::consts::PI; // Maps to [0, PI]
        
        // Spread nodes using configured radius range
        let radius = physics.initial_radius_min + (id_hash % physics.initial_radius_range);
        
        Self {
            id,
            metadata_id: metadata_id.clone(),
            label: String::new(), // Initialize as empty string, will be set from metadata later
            data: BinaryNodeData {
                // Use spherical coordinates for better 3D distribution
                position: Vec3Data::new(
                    radius * phi.sin() * theta.cos(),
                    radius * phi.sin() * theta.sin(),
                    radius * phi.cos(),
                ),
                velocity: Vec3Data::new(
                    0.0,  // Start with zero velocity
                    0.0,  // Physics will handle the movement
                    0.0,
                ),
                mass: 0,
                flags: 1, // Active by default
                padding: [0, 0],
            },
            metadata: HashMap::new(),
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        }
    }

    pub fn set_file_size(&mut self, size: u64) {
        self.file_size = size;
        // Update mass based on new file size using the centralized calculation
        self.data.mass = Self::calculate_mass(size);
        
        // Add the file_size to the metadata HashMap so it gets serialized to the client
        // This is our workaround since we can't directly serialize the file_size field
        if size > 0 {
            self.metadata.insert("fileSize".to_string(), size.to_string());
        }
    }

    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.data.position = Vec3Data::new(x, y, z);
        self
    }

    pub fn with_velocity(mut self, vx: f32, vy: f32, vz: f32) -> Self {
        self.data.velocity = Vec3Data::new(vx, vy, vz);
        self
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label = label;
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn with_type(mut self, node_type: String) -> Self {
        self.node_type = Some(node_type);
        self
    }

    pub fn with_size(mut self, size: f32) -> Self {
        self.size = Some(size);
        self
    }

    pub fn with_color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }

    pub fn with_group(mut self, group: String) -> Self {
        self.group = Some(group);
        self
    }

    /// Create a new node with a specific ID or use a stored ID if available
    pub fn new_with_stored_id(metadata_id: String, stored_node_id: Option<u32>) -> Self {
        // Use stored ID if available, otherwise generate a new one
        let id = match stored_node_id {
            Some(stored_id) => stored_id,
            None => NEXT_NODE_ID.fetch_add(1, Ordering::SeqCst),
        };
        
        // Use similar position initialization logic as the main constructor
        let id_hash = id as f32;
        let angle = id_hash * 0.618033988749895; // Golden ratio for good distribution
        let radius = (id_hash * 0.1).min(100.0); // Spread nodes up to radius 100
        
        Self {
            id,
            metadata_id: metadata_id.clone(),
            label: metadata_id,
            data: BinaryNodeData {
                position: Vec3Data::new(
                    radius * angle.cos() * 2.0,
                    radius * angle.sin() * 2.0,
                    (id_hash * 0.01 - 50.0).max(-100.0).min(100.0),
                ),
                velocity: Vec3Data::new(0.0, 0.0, 0.0),
                mass: 0,
                flags: 1, // Active by default
                padding: [0, 0],
            },
            metadata: HashMap::new(),
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        }
    }

    pub fn calculate_mass(file_size: u64) -> u8 {
        // Use log scale to prevent extremely large masses
        // Add 1 to file_size to handle empty files (log(0) is undefined)
        let base_mass = ((file_size + 1) as f32).log10() / 4.0;
        // Ensure minimum mass of 0.1 and maximum of 10.0
        let mass = base_mass.max(0.1).min(10.0);
        (mass * 255.0 / 10.0) as u8
    }

    // Convenience getters/setters for position and velocity
    pub fn x(&self) -> f32 { self.data.position.x }
    pub fn y(&self) -> f32 { self.data.position.y }
    pub fn z(&self) -> f32 { self.data.position.z }
    pub fn vx(&self) -> f32 { self.data.velocity.x }
    pub fn vy(&self) -> f32 { self.data.velocity.y }
    pub fn vz(&self) -> f32 { self.data.velocity.z }
    
    pub fn set_x(&mut self, val: f32) { self.data.position.x = val; }
    pub fn set_y(&mut self, val: f32) { self.data.position.y = val; }
    pub fn set_z(&mut self, val: f32) { self.data.position.z = val; }
    pub fn set_vx(&mut self, val: f32) { self.data.velocity.x = val; }
    pub fn set_vy(&mut self, val: f32) { self.data.velocity.y = val; }
    pub fn set_vz(&mut self, val: f32) { self.data.velocity.z = val; }

    /// Get the node ID as a string for socket/wire protocol compatibility
    pub fn id_as_string(&self) -> String {
        self.id.to_string()
    }

    /// Create a Node from a string ID (for socket/wire protocol compatibility)
    pub fn from_string_id(id_str: &str, metadata_id: String) -> Result<Self, std::num::ParseIntError> {
        let id: u32 = id_str.parse()?;
        Ok(Self::new_with_stored_id(metadata_id, Some(id)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use super::*;

    #[test]
    fn test_numeric_id_generation() {
        // Read the current value of the counter (it might have been incremented elsewhere)
        let start_value = NEXT_NODE_ID.load(Ordering::SeqCst);
        
        // Create two nodes with different metadata IDs
        let node1 = Node::new("test-file-1.md".to_string());
        let node2 = Node::new("test-file-2.md".to_string());
        
        // Verify each node has a unique numeric ID
        assert_ne!(node1.id, node2.id);
        
        // Verify metadata_id is stored correctly
        assert_eq!(node1.metadata_id, "test-file-1.md");
        assert_eq!(node2.metadata_id, "test-file-2.md");
        
        // Verify IDs are consecutive numbers
        assert_eq!(node1.id + 1, node2.id);
        
        // Verify final counter value
        let end_value = NEXT_NODE_ID.load(Ordering::SeqCst);
        assert_eq!(end_value, start_value + 2);
    }

    #[test]
    fn test_node_creation() {
        let node = Node::new("test".to_string())
            .with_label("Test Node".to_string())
            .with_position(1.0, 2.0, 3.0)
            .with_velocity(0.1, 0.2, 0.3)
            .with_type("test_type".to_string())
            .with_size(1.5)
            .with_color("#FF0000".to_string())
            .with_weight(2.0)
            .with_group("group1".to_string());

        // ID should be a numeric u32 now, not "test"
        assert!(node.id > 0, "ID should be positive, got: {}", node.id);
        assert_eq!(node.metadata_id, "test");
        assert_eq!(node.label, "Test Node");
        assert_eq!(node.data.position.x, 1.0);
        assert_eq!(node.data.position.y, 2.0);
        assert_eq!(node.data.position.z, 3.0);
        assert_eq!(node.data.velocity.x, 0.1);
        assert_eq!(node.data.velocity.y, 0.2);
        assert_eq!(node.data.velocity.z, 0.3);
        assert_eq!(node.node_type, Some("test_type".to_string()));
        assert_eq!(node.size, Some(1.5));
        assert_eq!(node.color, Some("#FF0000".to_string()));
        assert_eq!(node.weight, Some(2.0));
        assert_eq!(node.group, Some("group1".to_string()));
    }

    #[test]
    fn test_position_velocity_getters_setters() {
        let mut node = Node::new("test".to_string());
        
        node.set_x(1.0);
        node.set_y(2.0);
        node.set_z(3.0);
        node.set_vx(0.1);
        node.set_vy(0.2);
        node.set_vz(0.3);

        assert_eq!(node.x(), 1.0);
        assert_eq!(node.y(), 2.0);
        assert_eq!(node.z(), 3.0);
        assert_eq!(node.vx(), 0.1);
        assert_eq!(node.vy(), 0.2);
        assert_eq!(node.vz(), 0.3);
    }

    #[test]
    fn test_mass_calculation() {
        let mut node = Node::new("test".to_string());
        
        // Test small file
        node.set_file_size(100);  // 100 bytes
        assert!(node.data.mass > 0 && node.data.mass < 128);

        // Test large file
        node.set_file_size(1_000_000);  // 1MB
        assert!(node.data.mass > 128 && node.data.mass < 255);
    }
}

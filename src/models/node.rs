use crate::config::dev_config;
use crate::utils::socket_flow_messages::BinaryNodeData;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

// Static counter for generating unique numeric IDs
static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(1); // Start from 1 (0 could be reserved)

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    // Core data
    pub id: u32,
    pub metadata_id: String, // Store the original filename for lookup
    pub label: String,
    pub data: BinaryNodeData,

    // Physics fields (direct access to match schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vx: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vy: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vz: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mass: Option<f32>,

    // OWL Ontology linkage (matches unified_schema.sql)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owl_class_iri: Option<String>,

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
            }
            _ => NEXT_NODE_ID.fetch_add(1, Ordering::SeqCst),
        };

        // BREADCRUMB: Use random initial positions for force-directed graph
        // This ensures truly random starting positions for better force-directed layout
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let physics = dev_config::physics();

        // Generate random spherical coordinates for even distribution in 3D space
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI; // Random angle [0, 2π]
        let phi = rng.gen::<f32>() * std::f32::consts::PI; // Random angle [0, π]

        // Spread nodes using configured radius range with random distance
        let radius = physics.initial_radius_min + rng.gen::<f32>() * physics.initial_radius_range;

        let pos_x = radius * phi.sin() * theta.cos();
        let pos_y = radius * phi.sin() * theta.sin();
        let pos_z = radius * phi.cos();

        Self {
            id,
            metadata_id: metadata_id.clone(),
            label: String::new(), // Initialize as empty string, will be set from metadata later
            data: BinaryNodeData {
                node_id: id,
                // Use spherical coordinates for better 3D distribution with random positions
                x: pos_x,
                y: pos_y,
                z: pos_z,
                vx: 0.0, // Start with zero velocity
                vy: 0.0, // Physics will handle the movement
                vz: 0.0,
            },
            // Physics fields matching schema
            x: Some(pos_x),
            y: Some(pos_y),
            z: Some(pos_z),
            vx: Some(0.0),
            vy: Some(0.0),
            vz: Some(0.0),
            mass: Some(1.0), // Default mass
            owl_class_iri: None,
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
        // Note: Mass is no longer stored in BinaryNodeDataClient - handled separately

        // Add the file_size to the metadata HashMap so it gets serialized to the client
        // This is our workaround since we can't directly serialize the file_size field
        if size > 0 {
            self.metadata
                .insert("fileSize".to_string(), size.to_string());
        }
    }

    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.data.x = x;
        self.data.y = y;
        self.data.z = z;
        self.x = Some(x);
        self.y = Some(y);
        self.z = Some(z);
        self
    }

    pub fn with_velocity(mut self, vx: f32, vy: f32, vz: f32) -> Self {
        self.data.vx = vx;
        self.data.vy = vy;
        self.data.vz = vz;
        self.vx = Some(vx);
        self.vy = Some(vy);
        self.vz = Some(vz);
        self
    }

    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = Some(mass);
        self
    }

    pub fn with_owl_class_iri(mut self, iri: String) -> Self {
        self.owl_class_iri = Some(iri);
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

        let pos_x = radius * angle.cos() * 2.0;
        let pos_y = radius * angle.sin() * 2.0;
        let pos_z = (id_hash * 0.01 - 50.0).max(-100.0).min(100.0);

        Self {
            id,
            metadata_id: metadata_id.clone(),
            label: metadata_id,
            data: BinaryNodeData {
                node_id: id,
                x: pos_x,
                y: pos_y,
                z: pos_z,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            },
            // Physics fields matching schema
            x: Some(pos_x),
            y: Some(pos_y),
            z: Some(pos_z),
            vx: Some(0.0),
            vy: Some(0.0),
            vz: Some(0.0),
            mass: Some(1.0), // Default mass
            owl_class_iri: None,
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
    pub fn x(&self) -> f32 {
        self.data.x
    }
    pub fn y(&self) -> f32 {
        self.data.y
    }
    pub fn z(&self) -> f32 {
        self.data.z
    }
    pub fn vx(&self) -> f32 {
        self.data.vx
    }
    pub fn vy(&self) -> f32 {
        self.data.vy
    }
    pub fn vz(&self) -> f32 {
        self.data.vz
    }

    pub fn set_x(&mut self, val: f32) {
        self.data.x = val;
        self.x = Some(val);
    }
    pub fn set_y(&mut self, val: f32) {
        self.data.y = val;
        self.y = Some(val);
    }
    pub fn set_z(&mut self, val: f32) {
        self.data.z = val;
        self.z = Some(val);
    }
    pub fn set_vx(&mut self, val: f32) {
        self.data.vx = val;
        self.vx = Some(val);
    }
    pub fn set_vy(&mut self, val: f32) {
        self.data.vy = val;
        self.vy = Some(val);
    }
    pub fn set_vz(&mut self, val: f32) {
        self.data.vz = val;
        self.vz = Some(val);
    }

    pub fn set_mass(&mut self, val: f32) {
        self.mass = Some(val);
    }

    pub fn get_mass(&self) -> f32 {
        self.mass.unwrap_or(1.0)
    }

    /// Get the node ID as a string for socket/wire protocol compatibility
    pub fn id_as_string(&self) -> String {
        self.id.to_string()
    }

    /// Create a Node from a string ID (for socket/wire protocol compatibility)
    pub fn from_string_id(
        id_str: &str,
        metadata_id: String,
    ) -> Result<Self, std::num::ParseIntError> {
        let id: u32 = id_str.parse()?;
        Ok(Self::new_with_stored_id(metadata_id, Some(id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

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
        assert_eq!(node.data.x, 1.0);
        assert_eq!(node.data.y, 2.0);
        assert_eq!(node.data.z, 3.0);
        assert_eq!(node.data.vx, 0.1);
        assert_eq!(node.data.vy, 0.2);
        assert_eq!(node.data.vz, 0.3);
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

    // #[test]
    // fn test_mass_calculation() {
    //     // NOTE: Mass field not available in BinaryNodeDataClient
    //     // This test has been disabled as the client-side binary format
    //     // only includes position and velocity data (28 bytes total)
    //     // Mass calculation is handled server-side in BinaryNodeDataGPU
    // }
}

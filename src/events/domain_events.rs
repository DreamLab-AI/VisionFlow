use crate::events::types::DomainEvent;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ==================== Graph Events ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAddedEvent {
    pub node_id: String,
    pub label: String,
    pub node_type: String,
    pub properties: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for NodeAddedEvent {
    fn event_type(&self) -> &'static str {
        "NodeAdded"
    }
    fn aggregate_id(&self) -> &str {
        &self.node_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Node"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeUpdatedEvent {
    pub node_id: String,
    pub label: Option<String>,
    pub properties: Option<std::collections::HashMap<String, String>>,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for NodeUpdatedEvent {
    fn event_type(&self) -> &'static str {
        "NodeUpdated"
    }
    fn aggregate_id(&self) -> &str {
        &self.node_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Node"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRemovedEvent {
    pub node_id: String,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for NodeRemovedEvent {
    fn event_type(&self) -> &'static str {
        "NodeRemoved"
    }
    fn aggregate_id(&self) -> &str {
        &self.node_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Node"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeAddedEvent {
    pub edge_id: String,
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
    pub weight: f64,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for EdgeAddedEvent {
    fn event_type(&self) -> &'static str {
        "EdgeAdded"
    }
    fn aggregate_id(&self) -> &str {
        &self.edge_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Edge"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRemovedEvent {
    pub edge_id: String,
    pub source_id: String,
    pub target_id: String,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for EdgeRemovedEvent {
    fn event_type(&self) -> &'static str {
        "EdgeRemoved"
    }
    fn aggregate_id(&self) -> &str {
        &self.edge_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Edge"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSavedEvent {
    pub graph_id: String,
    pub file_path: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for GraphSavedEvent {
    fn event_type(&self) -> &'static str {
        "GraphSaved"
    }
    fn aggregate_id(&self) -> &str {
        &self.graph_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Graph"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphClearedEvent {
    pub graph_id: String,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for GraphClearedEvent {
    fn event_type(&self) -> &'static str {
        "GraphCleared"
    }
    fn aggregate_id(&self) -> &str {
        &self.graph_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Graph"
    }
}

// ==================== Ontology Events ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassAddedEvent {
    pub class_id: String,
    pub class_iri: String,
    pub label: Option<String>,
    pub parent_classes: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for ClassAddedEvent {
    fn event_type(&self) -> &'static str {
        "ClassAdded"
    }
    fn aggregate_id(&self) -> &str {
        &self.class_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "OntologyClass"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyAddedEvent {
    pub property_id: String,
    pub property_iri: String,
    pub property_type: String, // "ObjectProperty" | "DataProperty" | "AnnotationProperty"
    pub domain: Option<String>,
    pub range: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for PropertyAddedEvent {
    fn event_type(&self) -> &'static str {
        "PropertyAdded"
    }
    fn aggregate_id(&self) -> &str {
        &self.property_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "OntologyProperty"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomAddedEvent {
    pub axiom_id: String,
    pub axiom_type: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for AxiomAddedEvent {
    fn event_type(&self) -> &'static str {
        "AxiomAdded"
    }
    fn aggregate_id(&self) -> &str {
        &self.axiom_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Axiom"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyImportedEvent {
    pub ontology_id: String,
    pub file_path: String,
    pub format: String, // "RDF/XML" | "Turtle" | "OWL/XML"
    pub class_count: usize,
    pub property_count: usize,
    pub individual_count: usize,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for OntologyImportedEvent {
    fn event_type(&self) -> &'static str {
        "OntologyImported"
    }
    fn aggregate_id(&self) -> &str {
        &self.ontology_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Ontology"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCompletedEvent {
    pub ontology_id: String,
    pub reasoner_type: String,
    pub inferred_axioms: usize,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for InferenceCompletedEvent {
    fn event_type(&self) -> &'static str {
        "InferenceCompleted"
    }
    fn aggregate_id(&self) -> &str {
        &self.ontology_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Ontology"
    }
}

// ==================== Physics Events ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStartedEvent {
    pub simulation_id: String,
    pub physics_profile: String,
    pub node_count: usize,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for SimulationStartedEvent {
    fn event_type(&self) -> &'static str {
        "SimulationStarted"
    }
    fn aggregate_id(&self) -> &str {
        &self.simulation_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Simulation"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStoppedEvent {
    pub simulation_id: String,
    pub iterations: u32,
    pub final_energy: f64,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for SimulationStoppedEvent {
    fn event_type(&self) -> &'static str {
        "SimulationStopped"
    }
    fn aggregate_id(&self) -> &str {
        &self.simulation_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Simulation"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutOptimizedEvent {
    pub layout_id: String,
    pub algorithm: String,
    pub node_count: usize,
    pub optimization_score: f64,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for LayoutOptimizedEvent {
    fn event_type(&self) -> &'static str {
        "LayoutOptimized"
    }
    fn aggregate_id(&self) -> &str {
        &self.layout_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Layout"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionsUpdatedEvent {
    pub graph_id: String,
    pub updated_nodes: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for PositionsUpdatedEvent {
    fn event_type(&self) -> &'static str {
        "PositionsUpdated"
    }
    fn aggregate_id(&self) -> &str {
        &self.graph_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Graph"
    }
}

// ==================== Settings Events ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingUpdatedEvent {
    pub setting_key: String,
    pub old_value: Option<String>,
    pub new_value: String,
    pub category: String,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for SettingUpdatedEvent {
    fn event_type(&self) -> &'static str {
        "SettingUpdated"
    }
    fn aggregate_id(&self) -> &str {
        &self.setting_key
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Setting"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsProfileSavedEvent {
    pub profile_id: String,
    pub profile_name: String,
    pub parameters: std::collections::HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for PhysicsProfileSavedEvent {
    fn event_type(&self) -> &'static str {
        "PhysicsProfileSaved"
    }
    fn aggregate_id(&self) -> &str {
        &self.profile_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "PhysicsProfile"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingsImportedEvent {
    pub settings_id: String,
    pub file_path: String,
    pub imported_count: usize,
    pub timestamp: DateTime<Utc>,
}

impl DomainEvent for SettingsImportedEvent {
    fn event_type(&self) -> &'static str {
        "SettingsImported"
    }
    fn aggregate_id(&self) -> &str {
        &self.settings_id
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn aggregate_type(&self) -> &'static str {
        "Settings"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_added_event() {
        let event = NodeAddedEvent {
            node_id: "node-1".to_string(),
            label: "Test Node".to_string(),
            node_type: "Person".to_string(),
            properties: std::collections::HashMap::new(),
            timestamp: Utc::now(),
        };

        assert_eq!(event.event_type(), "NodeAdded");
        assert_eq!(event.aggregate_id(), "node-1");
        assert_eq!(event.aggregate_type(), "Node");
    }

    #[test]
    fn test_ontology_imported_event() {
        let event = OntologyImportedEvent {
            ontology_id: "onto-1".to_string(),
            file_path: "/test.owl".to_string(),
            format: "RDF/XML".to_string(),
            class_count: 100,
            property_count: 50,
            individual_count: 200,
            timestamp: Utc::now(),
        };

        assert_eq!(event.event_type(), "OntologyImported");
        assert_eq!(event.aggregate_type(), "Ontology");
    }

    #[test]
    fn test_simulation_events() {
        let start = SimulationStartedEvent {
            simulation_id: "sim-1".to_string(),
            physics_profile: "force-directed".to_string(),
            node_count: 100,
            timestamp: Utc::now(),
        };

        let stop = SimulationStoppedEvent {
            simulation_id: "sim-1".to_string(),
            iterations: 1000,
            final_energy: 0.05,
            timestamp: Utc::now(),
        };

        assert_eq!(start.aggregate_id(), stop.aggregate_id());
    }
}

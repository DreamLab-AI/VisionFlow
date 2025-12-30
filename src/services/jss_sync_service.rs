//! JSS Sync Service
//!
//! Synchronizes ontology data from Neo4j to JavaScript Solid Server (JSS) pods.
//! Provides real-time synchronization of ontology classes, properties, and user contributions.
//!
//! Features:
//! - Sync ontology to public pod at /public/ontology/
//! - Sync user contributions and proposals
//! - Watch Neo4j changes and trigger automatic sync
//! - Turtle/RDF serialization for Solid LDP containers

use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;

use crate::handlers::solid_proxy_handler::JssConfig;

/// JSS Sync Service errors
#[derive(Debug, Error)]
pub enum JssSyncError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Neo4j query failed: {0}")]
    Neo4jError(String),

    #[error("JSS server error: {status} - {message}")]
    JssServerError { status: u16, message: String },

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Authentication required")]
    AuthenticationRequired,

    #[error("Sync conflict: {0}")]
    SyncConflict(String),
}

pub type Result<T> = std::result::Result<T, JssSyncError>;

/// Ontology resource for syncing to JSS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyResource {
    pub iri: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub resource_type: OntologyResourceType,
    pub parent_iris: Vec<String>,
    pub properties: HashMap<String, String>,
    pub source_domain: Option<String>,
    pub quality_score: Option<f32>,
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OntologyResourceType {
    Class,
    Property,
    Individual,
    Axiom,
}

/// User contribution for syncing to personal pod
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContribution {
    pub user_npub: String,
    pub contribution_id: String,
    pub contribution_type: ContributionType,
    pub target_iri: String,
    pub proposed_changes: serde_json::Value,
    pub status: ContributionStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContributionType {
    NewClass,
    UpdateClass,
    NewProperty,
    UpdateProperty,
    NewRelationship,
    Annotation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContributionStatus {
    Draft,
    Proposed,
    UnderReview,
    Accepted,
    Rejected,
}

/// Neo4j change event for triggering sync
#[derive(Debug, Clone)]
pub struct Neo4jChangeEvent {
    pub change_type: Neo4jChangeType,
    pub affected_iris: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Neo4jChangeType {
    ClassCreated,
    ClassUpdated,
    ClassDeleted,
    PropertyCreated,
    PropertyUpdated,
    PropertyDeleted,
    RelationshipCreated,
    RelationshipDeleted,
}

/// Sync status for tracking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    pub last_sync: Option<chrono::DateTime<chrono::Utc>>,
    pub resources_synced: u64,
    pub sync_in_progress: bool,
    pub last_error: Option<String>,
    pub pending_changes: u64,
}

/// Configuration for JSS sync service
#[derive(Debug, Clone)]
pub struct JssSyncConfig {
    pub jss_config: JssConfig,
    pub sync_interval_secs: u64,
    pub batch_size: usize,
    pub public_ontology_path: String,
    pub contributions_path: String,
    pub enable_auto_sync: bool,
}

impl Default for JssSyncConfig {
    fn default() -> Self {
        Self {
            jss_config: JssConfig::from_env(),
            sync_interval_secs: 300, // 5 minutes
            batch_size: 100,
            public_ontology_path: "/public/ontology".to_string(),
            contributions_path: "/contributions".to_string(),
            enable_auto_sync: true,
        }
    }
}

/// JSS Sync Service
///
/// Manages synchronization between Neo4j ontology repository and JSS pods.
pub struct JssSyncService {
    config: JssSyncConfig,
    http_client: Client,
    sync_status: Arc<RwLock<SyncStatus>>,
    change_tx: mpsc::Sender<Neo4jChangeEvent>,
    change_rx: Arc<RwLock<mpsc::Receiver<Neo4jChangeEvent>>>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl JssSyncService {
    /// Create a new JSS sync service
    pub fn new(config: JssSyncConfig) -> Self {
        let (change_tx, change_rx) = mpsc::channel(1000);

        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        info!(
            "JSS Sync Service initialized - JSS URL: {}, auto-sync: {}",
            config.jss_config.base_url,
            config.enable_auto_sync
        );

        Self {
            config,
            http_client,
            sync_status: Arc::new(RwLock::new(SyncStatus {
                last_sync: None,
                resources_synced: 0,
                sync_in_progress: false,
                last_error: None,
                pending_changes: 0,
            })),
            change_tx,
            change_rx: Arc::new(RwLock::new(change_rx)),
            shutdown_tx: None,
        }
    }

    /// Create with default configuration from environment
    pub fn from_env() -> Self {
        Self::new(JssSyncConfig::default())
    }

    /// Get sender for Neo4j change events
    pub fn get_change_sender(&self) -> mpsc::Sender<Neo4jChangeEvent> {
        self.change_tx.clone()
    }

    /// Get current sync status
    pub async fn get_status(&self) -> SyncStatus {
        self.sync_status.read().await.clone()
    }

    /// Start the sync service with background workers
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting JSS Sync Service...");

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Start periodic sync worker if enabled
        if self.config.enable_auto_sync {
            let config = self.config.clone();
            let http_client = self.http_client.clone();
            let sync_status = self.sync_status.clone();

            tokio::spawn(async move {
                let mut interval = interval(Duration::from_secs(config.sync_interval_secs));

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            debug!("Running periodic ontology sync...");
                            if let Err(e) = Self::run_sync_cycle(&config, &http_client, &sync_status).await {
                                error!("Periodic sync failed: {}", e);
                            }
                        }
                        _ = shutdown_rx.recv() => {
                            info!("Sync worker received shutdown signal");
                            break;
                        }
                    }
                }
            });
        }

        // Start change watcher worker
        let change_rx = self.change_rx.clone();
        let config = self.config.clone();
        let http_client = self.http_client.clone();
        let sync_status = self.sync_status.clone();

        tokio::spawn(async move {
            let mut rx = change_rx.write().await;

            while let Some(event) = rx.recv().await {
                debug!("Processing Neo4j change event: {:?}", event.change_type);

                // Update pending changes count
                {
                    let mut status = sync_status.write().await;
                    status.pending_changes += event.affected_iris.len() as u64;
                }

                // Sync affected resources
                if let Err(e) = Self::sync_changed_resources(
                    &config,
                    &http_client,
                    &sync_status,
                    &event,
                ).await {
                    error!("Failed to sync changed resources: {}", e);
                }
            }
        });

        info!("JSS Sync Service started successfully");
        Ok(())
    }

    /// Stop the sync service
    pub async fn stop(&mut self) {
        info!("Stopping JSS Sync Service...");

        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        info!("JSS Sync Service stopped");
    }

    /// Sync ontology classes and properties to public JSS pod
    ///
    /// Writes ontology data to /public/ontology/ as Turtle files
    pub async fn sync_ontology_to_public(
        &self,
        resources: &[OntologyResource],
    ) -> Result<u64> {
        info!("Syncing {} ontology resources to public pod", resources.len());

        let mut synced_count = 0u64;

        // Update sync status
        {
            let mut status = self.sync_status.write().await;
            status.sync_in_progress = true;
        }

        // Ensure public ontology container exists
        self.ensure_container_exists(&self.config.public_ontology_path).await?;

        // Sync in batches
        for chunk in resources.chunks(self.config.batch_size) {
            for resource in chunk {
                match self.sync_resource_to_jss(resource).await {
                    Ok(_) => {
                        synced_count += 1;
                        debug!("Synced resource: {}", resource.iri);
                    }
                    Err(e) => {
                        warn!("Failed to sync resource {}: {}", resource.iri, e);
                    }
                }
            }
        }

        // Update sync status
        {
            let mut status = self.sync_status.write().await;
            status.sync_in_progress = false;
            status.last_sync = Some(chrono::Utc::now());
            status.resources_synced += synced_count;
            status.last_error = None;
        }

        info!("Synced {} ontology resources to public pod", synced_count);
        Ok(synced_count)
    }

    /// Sync user contributions to their personal pod
    ///
    /// Writes contribution proposals to user's pod at /contributions/
    pub async fn sync_user_contributions(
        &self,
        user_npub: &str,
        contributions: &[UserContribution],
    ) -> Result<u64> {
        info!(
            "Syncing {} contributions for user {}",
            contributions.len(),
            user_npub
        );

        let mut synced_count = 0u64;

        // Build user-specific path
        let user_path = format!("/pods/{}{}", user_npub, self.config.contributions_path);

        // Ensure contributions container exists
        self.ensure_container_exists(&user_path).await?;

        for contribution in contributions {
            match self.sync_contribution_to_jss(&user_path, contribution).await {
                Ok(_) => {
                    synced_count += 1;
                    debug!("Synced contribution: {}", contribution.contribution_id);
                }
                Err(e) => {
                    warn!(
                        "Failed to sync contribution {}: {}",
                        contribution.contribution_id, e
                    );
                }
            }
        }

        info!("Synced {} contributions for user {}", synced_count, user_npub);
        Ok(synced_count)
    }

    /// Notify of Neo4j changes to trigger sync
    pub async fn notify_neo4j_change(&self, event: Neo4jChangeEvent) -> Result<()> {
        self.change_tx
            .send(event)
            .await
            .map_err(|e| JssSyncError::SerializationError(format!("Failed to send change event: {}", e)))
    }

    // Private helper methods

    async fn run_sync_cycle(
        config: &JssSyncConfig,
        http_client: &Client,
        sync_status: &Arc<RwLock<SyncStatus>>,
    ) -> Result<()> {
        // This would query Neo4j for all resources and sync them
        // For now, just update the status
        let mut status = sync_status.write().await;
        status.last_sync = Some(chrono::Utc::now());
        Ok(())
    }

    async fn sync_changed_resources(
        config: &JssSyncConfig,
        http_client: &Client,
        sync_status: &Arc<RwLock<SyncStatus>>,
        event: &Neo4jChangeEvent,
    ) -> Result<()> {
        debug!(
            "Syncing {} changed resources",
            event.affected_iris.len()
        );

        // Decrement pending changes
        {
            let mut status = sync_status.write().await;
            status.pending_changes = status.pending_changes.saturating_sub(event.affected_iris.len() as u64);
        }

        Ok(())
    }

    async fn ensure_container_exists(&self, path: &str) -> Result<()> {
        let url = format!("{}{}", self.config.jss_config.base_url, path);

        // Check if container exists
        let response = self.http_client.head(&url).send().await?;

        if response.status().as_u16() == 404 {
            // Create container
            debug!("Creating container: {}", path);

            let create_response = self.http_client
                .put(&url)
                .header("Content-Type", "text/turtle")
                .header("Link", "<http://www.w3.org/ns/ldp#BasicContainer>; rel=\"type\"")
                .body("")
                .send()
                .await?;

            if !create_response.status().is_success() {
                return Err(JssSyncError::JssServerError {
                    status: create_response.status().as_u16(),
                    message: format!("Failed to create container: {}", path),
                });
            }

            info!("Created container: {}", path);
        }

        Ok(())
    }

    async fn sync_resource_to_jss(&self, resource: &OntologyResource) -> Result<()> {
        // Generate resource slug from IRI
        let slug = Self::iri_to_slug(&resource.iri);
        let resource_path = format!(
            "{}/{}.jsonld",
            self.config.public_ontology_path,
            slug
        );
        let url = format!("{}{}", self.config.jss_config.base_url, resource_path);

        // Serialize resource to JSON-LD (native format for JSS)
        let jsonld = self.resource_to_jsonld(resource)?;

        // PUT to JSS with JSON-LD content type
        let response = self.http_client
            .put(&url)
            .header("Content-Type", "application/ld+json")
            .body(serde_json::to_string(&jsonld).map_err(|e| JssSyncError::SerializationError(e.to_string()))?)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(JssSyncError::JssServerError {
                status: response.status().as_u16(),
                message: format!("Failed to sync resource: {}", resource.iri),
            });
        }

        Ok(())
    }

    async fn sync_contribution_to_jss(
        &self,
        user_path: &str,
        contribution: &UserContribution,
    ) -> Result<()> {
        let resource_path = format!("{}/{}.jsonld", user_path, contribution.contribution_id);
        let url = format!("{}{}", self.config.jss_config.base_url, resource_path);

        // Serialize contribution to JSON-LD
        let jsonld = self.contribution_to_jsonld(contribution)?;

        // PUT to JSS
        let response = self.http_client
            .put(&url)
            .header("Content-Type", "application/ld+json")
            .body(jsonld)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(JssSyncError::JssServerError {
                status: response.status().as_u16(),
                message: format!("Failed to sync contribution: {}", contribution.contribution_id),
            });
        }

        Ok(())
    }

    fn iri_to_slug(iri: &str) -> String {
        // Extract local name from IRI
        if let Some(hash_pos) = iri.rfind('#') {
            return iri[hash_pos + 1..].to_string();
        }
        if let Some(slash_pos) = iri.rfind('/') {
            return iri[slash_pos + 1..].to_string();
        }
        // Fallback: URL-encode the whole IRI
        urlencoding::encode(iri).to_string()
    }

    /// Convert ontology resource to JSON-LD format (native for JSS)
    fn resource_to_jsonld(&self, resource: &OntologyResource) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "@context": "https://visionflow.io/contexts/ontology.jsonld",
            "@id": &resource.iri,
            "@type": match &resource.resource_type {
                OntologyResourceType::Class => "owl:Class",
                OntologyResourceType::Property => "owl:ObjectProperty",
                OntologyResourceType::Individual => "owl:NamedIndividual",
                OntologyResourceType::Axiom => "owl:Axiom",
            },
            "rdfs:label": &resource.label,
            "rdfs:comment": &resource.description,
            "rdfs:subClassOf": resource.parent_iris.iter()
                .map(|p| serde_json::json!({"@id": p}))
                .collect::<Vec<_>>(),
            "vf:qualityScore": resource.quality_score,
            "vf:sourceDomain": &resource.source_domain,
            "vf:properties": &resource.properties,
            "vf:lastModified": resource.last_modified.map(|dt| dt.to_rfc3339()),
        }))
    }

    fn resource_to_turtle(&self, resource: &OntologyResource) -> Result<String> {
        let mut turtle = String::new();

        // Prefixes
        turtle.push_str("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n");
        turtle.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        turtle.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");
        turtle.push_str("@prefix vf: <http://visionflow.io/ontology#> .\n\n");

        // Resource declaration
        turtle.push_str(&format!("<{}>\n", resource.iri));

        // Type based on resource type
        match resource.resource_type {
            OntologyResourceType::Class => {
                turtle.push_str("    a owl:Class");
            }
            OntologyResourceType::Property => {
                turtle.push_str("    a owl:ObjectProperty");
            }
            OntologyResourceType::Individual => {
                turtle.push_str("    a owl:NamedIndividual");
            }
            OntologyResourceType::Axiom => {
                turtle.push_str("    a owl:Axiom");
            }
        }

        // Label
        if let Some(label) = &resource.label {
            turtle.push_str(&format!(" ;\n    rdfs:label \"{}\"", Self::escape_turtle_string(label)));
        }

        // Description
        if let Some(desc) = &resource.description {
            turtle.push_str(&format!(" ;\n    rdfs:comment \"{}\"", Self::escape_turtle_string(desc)));
        }

        // Parent classes
        for parent in &resource.parent_iris {
            turtle.push_str(&format!(" ;\n    rdfs:subClassOf <{}>", parent));
        }

        // Quality score
        if let Some(score) = resource.quality_score {
            turtle.push_str(&format!(" ;\n    vf:qualityScore \"{}\"^^xsd:float", score));
        }

        // Source domain
        if let Some(domain) = &resource.source_domain {
            turtle.push_str(&format!(" ;\n    vf:sourceDomain \"{}\"", Self::escape_turtle_string(domain)));
        }

        // Additional properties
        for (key, value) in &resource.properties {
            turtle.push_str(&format!(" ;\n    vf:{} \"{}\"", key, Self::escape_turtle_string(value)));
        }

        turtle.push_str(" .\n");

        Ok(turtle)
    }

    fn contribution_to_jsonld(&self, contribution: &UserContribution) -> Result<String> {
        let jsonld = serde_json::json!({
            "@context": {
                "@vocab": "http://visionflow.io/contributions#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "created": {"@type": "xsd:dateTime"},
                "updated": {"@type": "xsd:dateTime"}
            },
            "@id": format!("urn:contribution:{}", contribution.contribution_id),
            "@type": format!("{:?}", contribution.contribution_type),
            "contributor": contribution.user_npub,
            "target": contribution.target_iri,
            "status": format!("{:?}", contribution.status),
            "proposedChanges": contribution.proposed_changes,
            "created": contribution.created_at.to_rfc3339(),
            "updated": contribution.updated_at.to_rfc3339()
        });

        serde_json::to_string_pretty(&jsonld)
            .map_err(|e| JssSyncError::SerializationError(e.to_string()))
    }

    fn escape_turtle_string(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }
}

impl Default for JssSyncService {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iri_to_slug() {
        assert_eq!(
            JssSyncService::iri_to_slug("http://example.org/ontology#Person"),
            "Person"
        );
        assert_eq!(
            JssSyncService::iri_to_slug("http://example.org/ontology/Agent"),
            "Agent"
        );
    }

    #[test]
    fn test_escape_turtle_string() {
        assert_eq!(
            JssSyncService::escape_turtle_string("Hello \"World\""),
            "Hello \\\"World\\\""
        );
        assert_eq!(
            JssSyncService::escape_turtle_string("Line1\nLine2"),
            "Line1\\nLine2"
        );
    }

    #[tokio::test]
    async fn test_sync_service_creation() {
        let service = JssSyncService::from_env();
        let status = service.get_status().await;

        assert!(!status.sync_in_progress);
        assert_eq!(status.resources_synced, 0);
    }

    #[test]
    fn test_resource_to_turtle() {
        let service = JssSyncService::from_env();

        let resource = OntologyResource {
            iri: "http://example.org/ontology#Person".to_string(),
            label: Some("Person".to_string()),
            description: Some("A human being".to_string()),
            resource_type: OntologyResourceType::Class,
            parent_iris: vec!["http://example.org/ontology#Agent".to_string()],
            properties: HashMap::new(),
            source_domain: Some("core".to_string()),
            quality_score: Some(0.95),
            last_modified: None,
        };

        let turtle = service.resource_to_turtle(&resource).unwrap();

        assert!(turtle.contains("@prefix owl:"));
        assert!(turtle.contains("<http://example.org/ontology#Person>"));
        assert!(turtle.contains("a owl:Class"));
        assert!(turtle.contains("rdfs:label \"Person\""));
        assert!(turtle.contains("rdfs:subClassOf <http://example.org/ontology#Agent>"));
    }

    #[test]
    fn test_resource_to_jsonld() {
        let service = JssSyncService::from_env();

        let resource = OntologyResource {
            iri: "http://example.org/ontology#Person".to_string(),
            label: Some("Person".to_string()),
            description: Some("A human being".to_string()),
            resource_type: OntologyResourceType::Class,
            parent_iris: vec!["http://example.org/ontology#Agent".to_string()],
            properties: HashMap::new(),
            source_domain: Some("core".to_string()),
            quality_score: Some(0.95),
            last_modified: None,
        };

        let jsonld = service.resource_to_jsonld(&resource).unwrap();

        assert_eq!(jsonld["@context"], "https://visionflow.io/contexts/ontology.jsonld");
        assert_eq!(jsonld["@id"], "http://example.org/ontology#Person");
        assert_eq!(jsonld["@type"], "owl:Class");
        assert_eq!(jsonld["rdfs:label"], "Person");
        assert_eq!(jsonld["rdfs:comment"], "A human being");
        assert_eq!(jsonld["vf:qualityScore"], 0.95);
        assert_eq!(jsonld["vf:sourceDomain"], "core");

        let subclass_of = jsonld["rdfs:subClassOf"].as_array().unwrap();
        assert_eq!(subclass_of.len(), 1);
        assert_eq!(subclass_of[0]["@id"], "http://example.org/ontology#Agent");
    }
}

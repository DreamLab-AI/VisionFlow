//! JSS Sync Service
//!
//! Synchronizes ontology data from Neo4j (authoritative source) to JavaScript Solid Server (JSS) pods.
//! Provides unidirectional sync: Neo4j -> JSS for federation/distribution.
//!
//! Architecture:
//! - Neo4j is the authoritative source of truth for ontology data
//! - JSS serves as the distribution/federation layer for Solid protocol access
//! - Sync is unidirectional: changes flow from Neo4j to JSS only
//!
//! Sync Strategies:
//! - Full sync on startup (ensures JSS mirrors Neo4j state)
//! - Incremental sync on Neo4j change events
//! - Scheduled periodic sync (configurable interval, default 5 minutes)
//!
//! Features:
//! - Sync ontology classes, properties, and axioms to /public/ontology/
//! - Sync user contributions to personal pods at /pods/{npub}/contributions/
//! - Conflict resolution: Neo4j always wins (unidirectional)
//! - Metrics and logging for monitoring sync operations
//! - Turtle/JSON-LD serialization for Solid LDP containers

use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;

use crate::handlers::solid_proxy_handler::JssConfig;
use crate::ports::ontology_repository::{OntologyRepository, OwlClass, OwlProperty, OwlAxiom};

// ============================================================================
// Error Types
// ============================================================================

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

    #[error("Repository error: {0}")]
    RepositoryError(String),

    #[error("Service not started")]
    NotStarted,
}

pub type Result<T> = std::result::Result<T, JssSyncError>;

// ============================================================================
// Data Types
// ============================================================================

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
    pub last_modified: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContributionType {
    NewClass,
    UpdateClass,
    NewProperty,
    UpdateProperty,
    NewRelationship,
    Annotation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    pub timestamp: DateTime<Utc>,
    /// Source of the change for audit trail
    pub source: ChangeSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Neo4jChangeType {
    ClassCreated,
    ClassUpdated,
    ClassDeleted,
    PropertyCreated,
    PropertyUpdated,
    PropertyDeleted,
    RelationshipCreated,
    RelationshipDeleted,
    AxiomAdded,
    AxiomRemoved,
    BulkImport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeSource {
    GitHubSync,
    LocalMarkdownSync,
    UserContribution,
    Inference,
    Manual,
}

// ============================================================================
// Sync Status & Metrics
// ============================================================================

/// Sync status for tracking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    pub last_full_sync: Option<DateTime<Utc>>,
    pub last_incremental_sync: Option<DateTime<Utc>>,
    pub resources_synced: u64,
    pub sync_in_progress: bool,
    pub last_error: Option<String>,
    pub pending_changes: u64,
    pub sync_type: Option<String>,
}

/// Detailed sync metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMetrics {
    // Counters (total since startup)
    pub total_full_syncs: u64,
    pub total_incremental_syncs: u64,
    pub total_resources_synced: u64,
    pub total_errors: u64,
    pub total_conflicts_resolved: u64,

    // Current state
    pub classes_in_neo4j: u64,
    pub classes_in_jss: u64,
    pub properties_in_neo4j: u64,
    pub properties_in_jss: u64,

    // Timing
    pub avg_sync_duration_ms: f64,
    pub last_sync_duration_ms: u64,
    pub uptime_secs: u64,

    // Health
    pub neo4j_connected: bool,
    pub jss_connected: bool,
    pub last_neo4j_check: Option<DateTime<Utc>>,
    pub last_jss_check: Option<DateTime<Utc>>,
}

/// Internal metrics counters (atomic for thread safety)
struct MetricsCounters {
    total_full_syncs: AtomicU64,
    total_incremental_syncs: AtomicU64,
    total_resources_synced: AtomicU64,
    total_errors: AtomicU64,
    total_conflicts_resolved: AtomicU64,
    total_sync_duration_ms: AtomicU64,
    sync_count_for_avg: AtomicU64,
}

impl MetricsCounters {
    fn new() -> Self {
        Self {
            total_full_syncs: AtomicU64::new(0),
            total_incremental_syncs: AtomicU64::new(0),
            total_resources_synced: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_conflicts_resolved: AtomicU64::new(0),
            total_sync_duration_ms: AtomicU64::new(0),
            sync_count_for_avg: AtomicU64::new(0),
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for JSS sync service
#[derive(Debug, Clone)]
pub struct JssSyncConfig {
    pub jss_config: JssConfig,
    /// Interval for periodic sync (seconds)
    pub sync_interval_secs: u64,
    /// Batch size for resource syncing
    pub batch_size: usize,
    /// Path for public ontology resources
    pub public_ontology_path: String,
    /// Path for user contributions
    pub contributions_path: String,
    /// Enable automatic periodic sync
    pub enable_auto_sync: bool,
    /// Enable full sync on startup
    pub enable_startup_sync: bool,
    /// Timeout for HTTP requests (seconds)
    pub http_timeout_secs: u64,
    /// Maximum retries for failed sync operations
    pub max_retries: u32,
    /// Delay between retries (milliseconds)
    pub retry_delay_ms: u64,
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
            enable_startup_sync: true,
            http_timeout_secs: 30,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

// ============================================================================
// JSS Sync Service
// ============================================================================

/// JSS Sync Service
/// Manages unidirectional synchronization from Neo4j (authoritative) to JSS (distribution).
/// Sync Flow:
/// 1. Full sync on startup (if enabled)
/// 2. Listen for Neo4j change events via channel
/// 3. Periodic sync at configured interval
/// 4. Incremental sync triggered by change events
pub struct JssSyncService {
    config: JssSyncConfig,
    http_client: Client,
    ontology_repo: Option<Arc<dyn OntologyRepository>>,
    sync_status: Arc<RwLock<SyncStatus>>,
    metrics: Arc<MetricsCounters>,
    change_tx: mpsc::Sender<Neo4jChangeEvent>,
    change_rx: Arc<RwLock<mpsc::Receiver<Neo4jChangeEvent>>>,
    shutdown_tx: Arc<RwLock<Option<mpsc::Sender<()>>>>,
    started_at: Arc<RwLock<Option<Instant>>>,

    // Track last known state for conflict detection
    last_sync_checksums: Arc<RwLock<HashMap<String, String>>>,
}

impl JssSyncService {
    /// Create a new JSS sync service
    pub fn new(config: JssSyncConfig) -> Self {
        let (change_tx, change_rx) = mpsc::channel(1000);

        let http_client = Client::builder()
            .timeout(Duration::from_secs(config.http_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        info!(
            "JSS Sync Service initialized - JSS URL: {}, auto-sync: {}, interval: {}s",
            config.jss_config.base_url,
            config.enable_auto_sync,
            config.sync_interval_secs
        );

        Self {
            config,
            http_client,
            ontology_repo: None,
            sync_status: Arc::new(RwLock::new(SyncStatus {
                last_full_sync: None,
                last_incremental_sync: None,
                resources_synced: 0,
                sync_in_progress: false,
                last_error: None,
                pending_changes: 0,
                sync_type: None,
            })),
            metrics: Arc::new(MetricsCounters::new()),
            change_tx,
            change_rx: Arc::new(RwLock::new(change_rx)),
            shutdown_tx: Arc::new(RwLock::new(None)),
            started_at: Arc::new(RwLock::new(None)),
            last_sync_checksums: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration from environment
    pub fn from_env() -> Self {
        Self::new(JssSyncConfig::default())
    }

    /// Set the ontology repository for Neo4j access
    pub fn with_ontology_repository(mut self, repo: Arc<dyn OntologyRepository>) -> Self {
        self.ontology_repo = Some(repo);
        self
    }

    /// Get sender for Neo4j change events
    pub fn get_change_sender(&self) -> mpsc::Sender<Neo4jChangeEvent> {
        self.change_tx.clone()
    }

    /// Get current sync status
    pub async fn get_status(&self) -> SyncStatus {
        self.sync_status.read().await.clone()
    }

    /// Get detailed sync metrics
    pub async fn get_metrics(&self) -> SyncMetrics {
        let started_at = *self.started_at.read().await;
        let uptime = started_at.map(|s| s.elapsed().as_secs()).unwrap_or(0);

        let total_sync_duration = self.metrics.total_sync_duration_ms.load(Ordering::Relaxed);
        let sync_count = self.metrics.sync_count_for_avg.load(Ordering::Relaxed);
        let avg_duration = if sync_count > 0 {
            total_sync_duration as f64 / sync_count as f64
        } else {
            0.0
        };

        SyncMetrics {
            total_full_syncs: self.metrics.total_full_syncs.load(Ordering::Relaxed),
            total_incremental_syncs: self.metrics.total_incremental_syncs.load(Ordering::Relaxed),
            total_resources_synced: self.metrics.total_resources_synced.load(Ordering::Relaxed),
            total_errors: self.metrics.total_errors.load(Ordering::Relaxed),
            total_conflicts_resolved: self.metrics.total_conflicts_resolved.load(Ordering::Relaxed),
            classes_in_neo4j: 0, // Updated during sync
            classes_in_jss: 0,   // Updated during sync
            properties_in_neo4j: 0,
            properties_in_jss: 0,
            avg_sync_duration_ms: avg_duration,
            last_sync_duration_ms: 0, // Updated during sync
            uptime_secs: uptime,
            neo4j_connected: self.ontology_repo.is_some(),
            jss_connected: true, // Updated during health check
            last_neo4j_check: None,
            last_jss_check: None,
        }
    }

    /// Start the sync service with background workers
    pub async fn start(&self) -> Result<()> {
        info!("Starting JSS Sync Service...");

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        *self.shutdown_tx.write().await = Some(shutdown_tx);
        *self.started_at.write().await = Some(Instant::now());

        // Perform full sync on startup if enabled
        if self.config.enable_startup_sync {
            info!("Performing full sync on startup...");
            if let Err(e) = self.full_sync().await {
                error!("Startup full sync failed: {}", e);
                self.metrics.total_errors.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Start periodic sync worker if enabled
        if self.config.enable_auto_sync {
            let config = self.config.clone();
            let http_client = self.http_client.clone();
            let sync_status = self.sync_status.clone();
            let metrics = self.metrics.clone();
            let ontology_repo = self.ontology_repo.clone();
            let checksums = self.last_sync_checksums.clone();

            let mut shutdown_rx_clone = {
                let (tx, rx) = mpsc::channel::<()>(1);
                // We'll use a separate channel for the periodic worker
                rx
            };

            tokio::spawn(async move {
                let mut sync_interval = interval(Duration::from_secs(config.sync_interval_secs));

                loop {
                    tokio::select! {
                        _ = sync_interval.tick() => {
                            debug!("Running periodic ontology sync...");
                            if let Err(e) = Self::run_periodic_sync(
                                &config,
                                &http_client,
                                &sync_status,
                                &metrics,
                                ontology_repo.as_ref(),
                                &checksums,
                            ).await {
                                error!("Periodic sync failed: {}", e);
                                metrics.total_errors.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        _ = shutdown_rx_clone.recv() => {
                            info!("Periodic sync worker received shutdown signal");
                            break;
                        }
                    }
                }
            });
        }

        // Start change event worker
        let change_rx = self.change_rx.clone();
        let config = self.config.clone();
        let http_client = self.http_client.clone();
        let sync_status = self.sync_status.clone();
        let metrics = self.metrics.clone();
        let ontology_repo = self.ontology_repo.clone();
        let checksums = self.last_sync_checksums.clone();

        tokio::spawn(async move {
            let mut rx = change_rx.write().await;

            while let Some(event) = rx.recv().await {
                info!(
                    "Processing Neo4j change event: {:?} affecting {} resources",
                    event.change_type,
                    event.affected_iris.len()
                );

                // Update pending changes count
                {
                    let mut status = sync_status.write().await;
                    status.pending_changes += event.affected_iris.len() as u64;
                }

                // Perform incremental sync for affected resources
                if let Err(e) = Self::incremental_sync(
                    &config,
                    &http_client,
                    &sync_status,
                    &metrics,
                    ontology_repo.as_ref(),
                    &event,
                    &checksums,
                ).await {
                    error!("Incremental sync failed for event {:?}: {}", event.change_type, e);
                    metrics.total_errors.fetch_add(1, Ordering::Relaxed);

                    let mut status = sync_status.write().await;
                    status.last_error = Some(format!("{}", e));
                }
            }
        });

        info!("JSS Sync Service started successfully");
        Ok(())
    }

    /// Stop the sync service
    pub async fn stop(&self) {
        info!("Stopping JSS Sync Service...");

        if let Some(tx) = self.shutdown_tx.write().await.take() {
            let _ = tx.send(()).await;
        }

        info!("JSS Sync Service stopped");
    }

    /// Perform a full sync of all ontology data from Neo4j to JSS
    pub async fn full_sync(&self) -> Result<u64> {
        let start_time = Instant::now();
        info!("Starting full sync from Neo4j to JSS...");

        // Update status
        {
            let mut status = self.sync_status.write().await;
            status.sync_in_progress = true;
            status.sync_type = Some("full".to_string());
        }

        let repo = self.ontology_repo.as_ref()
            .ok_or(JssSyncError::RepositoryError("Ontology repository not configured".to_string()))?;

        // Load all classes from Neo4j
        let classes = repo.list_owl_classes().await
            .map_err(|e| JssSyncError::Neo4jError(format!("Failed to load classes: {}", e)))?;

        info!("Loaded {} classes from Neo4j", classes.len());

        // Load all properties from Neo4j
        let properties = repo.list_owl_properties().await
            .map_err(|e| JssSyncError::Neo4jError(format!("Failed to load properties: {}", e)))?;

        info!("Loaded {} properties from Neo4j", properties.len());

        // Convert to OntologyResource format
        let mut resources: Vec<OntologyResource> = Vec::with_capacity(classes.len() + properties.len());

        for class in &classes {
            resources.push(Self::owl_class_to_resource(class));
        }

        for prop in &properties {
            resources.push(Self::owl_property_to_resource(prop));
        }

        // Ensure public ontology container exists
        self.ensure_container_exists(&self.config.public_ontology_path).await?;

        // Sync all resources to JSS
        let synced_count = self.sync_resources_to_jss(&resources).await?;

        // Update checksums for conflict detection
        {
            let mut checksums = self.last_sync_checksums.write().await;
            for resource in &resources {
                let checksum = Self::compute_resource_checksum(resource);
                checksums.insert(resource.iri.clone(), checksum);
            }
        }

        let duration = start_time.elapsed();

        // Update metrics
        self.metrics.total_full_syncs.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_resources_synced.fetch_add(synced_count, Ordering::Relaxed);
        self.metrics.total_sync_duration_ms.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
        self.metrics.sync_count_for_avg.fetch_add(1, Ordering::Relaxed);

        // Update status
        {
            let mut status = self.sync_status.write().await;
            status.sync_in_progress = false;
            status.last_full_sync = Some(Utc::now());
            status.resources_synced += synced_count;
            status.last_error = None;
            status.sync_type = None;
        }

        info!(
            "Full sync completed: {} resources synced in {}ms",
            synced_count,
            duration.as_millis()
        );

        Ok(synced_count)
    }

    /// Sync ontology classes and properties to public JSS pod
    pub async fn sync_ontology_to_public(&self, resources: &[OntologyResource]) -> Result<u64> {
        info!("Syncing {} ontology resources to public pod", resources.len());

        // Update sync status
        {
            let mut status = self.sync_status.write().await;
            status.sync_in_progress = true;
        }

        // Ensure public ontology container exists
        self.ensure_container_exists(&self.config.public_ontology_path).await?;

        let synced_count = self.sync_resources_to_jss(resources).await?;

        // Update sync status
        {
            let mut status = self.sync_status.write().await;
            status.sync_in_progress = false;
            status.last_incremental_sync = Some(Utc::now());
            status.resources_synced += synced_count;
            status.last_error = None;
        }

        info!("Synced {} ontology resources to public pod", synced_count);
        Ok(synced_count)
    }

    /// Sync user contributions to their personal pod
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

    /// Check if JSS is reachable
    pub async fn check_jss_health(&self) -> bool {
        let health_url = format!("{}/health", self.config.jss_config.base_url);
        match self.http_client.get(&health_url).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    // ========================================================================
    // Private Implementation
    // ========================================================================

    /// Run periodic sync (called by background worker)
    async fn run_periodic_sync(
        config: &JssSyncConfig,
        http_client: &Client,
        sync_status: &Arc<RwLock<SyncStatus>>,
        metrics: &Arc<MetricsCounters>,
        ontology_repo: Option<&Arc<dyn OntologyRepository>>,
        checksums: &Arc<RwLock<HashMap<String, String>>>,
    ) -> Result<()> {
        let start_time = Instant::now();

        let repo = ontology_repo
            .ok_or(JssSyncError::RepositoryError("Repository not configured".to_string()))?;

        // Load current state from Neo4j
        let classes = repo.list_owl_classes().await
            .map_err(|e| JssSyncError::Neo4jError(format!("{}", e)))?;

        let mut resources_to_sync = Vec::new();

        {
            let current_checksums = checksums.read().await;

            for class in &classes {
                let resource = Self::owl_class_to_resource(class);
                let new_checksum = Self::compute_resource_checksum(&resource);

                // Only sync if changed
                if current_checksums.get(&resource.iri) != Some(&new_checksum) {
                    resources_to_sync.push(resource);
                }
            }
        }

        if resources_to_sync.is_empty() {
            debug!("No changes detected, skipping periodic sync");
            return Ok(());
        }

        info!("Periodic sync: {} resources changed", resources_to_sync.len());

        // Sync changed resources
        let mut synced = 0u64;
        for resource in &resources_to_sync {
            if let Err(e) = Self::sync_single_resource(config, http_client, resource).await {
                warn!("Failed to sync resource {}: {}", resource.iri, e);
            } else {
                synced += 1;
            }
        }

        // Update checksums
        {
            let mut current_checksums = checksums.write().await;
            for resource in &resources_to_sync {
                let checksum = Self::compute_resource_checksum(resource);
                current_checksums.insert(resource.iri.clone(), checksum);
            }
        }

        let duration = start_time.elapsed();

        // Update metrics
        metrics.total_incremental_syncs.fetch_add(1, Ordering::Relaxed);
        metrics.total_resources_synced.fetch_add(synced, Ordering::Relaxed);
        metrics.total_sync_duration_ms.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
        metrics.sync_count_for_avg.fetch_add(1, Ordering::Relaxed);

        // Update status
        {
            let mut status = sync_status.write().await;
            status.last_incremental_sync = Some(Utc::now());
            status.resources_synced += synced;
        }

        info!(
            "Periodic sync completed: {} resources in {}ms",
            synced,
            duration.as_millis()
        );

        Ok(())
    }

    /// Perform incremental sync for specific changed resources
    async fn incremental_sync(
        config: &JssSyncConfig,
        http_client: &Client,
        sync_status: &Arc<RwLock<SyncStatus>>,
        metrics: &Arc<MetricsCounters>,
        ontology_repo: Option<&Arc<dyn OntologyRepository>>,
        event: &Neo4jChangeEvent,
        checksums: &Arc<RwLock<HashMap<String, String>>>,
    ) -> Result<()> {
        let start_time = Instant::now();

        info!(
            "Incremental sync for {:?} event, {} affected IRIs",
            event.change_type,
            event.affected_iris.len()
        );

        let repo = ontology_repo
            .ok_or(JssSyncError::RepositoryError("Repository not configured".to_string()))?;

        let mut synced = 0u64;
        let mut deleted = 0u64;

        match event.change_type {
            Neo4jChangeType::ClassDeleted | Neo4jChangeType::PropertyDeleted => {
                // Handle deletions - remove from JSS
                for iri in &event.affected_iris {
                    if let Err(e) = Self::delete_resource_from_jss(config, http_client, iri).await {
                        warn!("Failed to delete resource {} from JSS: {}", iri, e);
                    } else {
                        deleted += 1;
                        // Remove from checksums
                        checksums.write().await.remove(iri);
                    }
                }
            }
            _ => {
                // Handle creates/updates - sync to JSS
                for iri in &event.affected_iris {
                    // Fetch the current state from Neo4j
                    if let Ok(Some(class)) = repo.get_owl_class(iri).await {
                        let resource = Self::owl_class_to_resource(&class);

                        if let Err(e) = Self::sync_single_resource(config, http_client, &resource).await {
                            warn!("Failed to sync resource {}: {}", iri, e);
                        } else {
                            synced += 1;
                            // Update checksum
                            let checksum = Self::compute_resource_checksum(&resource);
                            checksums.write().await.insert(iri.clone(), checksum);
                        }
                    } else if let Ok(Some(prop)) = repo.get_owl_property(iri).await {
                        let resource = Self::owl_property_to_resource(&prop);

                        if let Err(e) = Self::sync_single_resource(config, http_client, &resource).await {
                            warn!("Failed to sync resource {}: {}", iri, e);
                        } else {
                            synced += 1;
                            let checksum = Self::compute_resource_checksum(&resource);
                            checksums.write().await.insert(iri.clone(), checksum);
                        }
                    } else {
                        debug!("Resource {} not found in Neo4j, skipping", iri);
                    }
                }
            }
        }

        let duration = start_time.elapsed();

        // Update metrics
        metrics.total_incremental_syncs.fetch_add(1, Ordering::Relaxed);
        metrics.total_resources_synced.fetch_add(synced, Ordering::Relaxed);
        metrics.total_sync_duration_ms.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
        metrics.sync_count_for_avg.fetch_add(1, Ordering::Relaxed);

        // Decrement pending changes
        {
            let mut status = sync_status.write().await;
            status.pending_changes = status.pending_changes.saturating_sub(event.affected_iris.len() as u64);
            status.last_incremental_sync = Some(Utc::now());
            status.resources_synced += synced;
        }

        info!(
            "Incremental sync completed: {} synced, {} deleted in {}ms",
            synced,
            deleted,
            duration.as_millis()
        );

        Ok(())
    }

    /// Sync a batch of resources to JSS
    async fn sync_resources_to_jss(&self, resources: &[OntologyResource]) -> Result<u64> {
        let mut synced_count = 0u64;

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

        Ok(synced_count)
    }

    /// Sync a single resource (static method for use in workers)
    async fn sync_single_resource(
        config: &JssSyncConfig,
        http_client: &Client,
        resource: &OntologyResource,
    ) -> Result<()> {
        let slug = Self::iri_to_slug(&resource.iri);
        let resource_path = format!("{}/{}.jsonld", config.public_ontology_path, slug);
        let url = format!("{}{}", config.jss_config.base_url, resource_path);

        // Serialize resource to JSON-LD
        let jsonld = Self::resource_to_jsonld_static(resource)?;

        // PUT to JSS with retries
        let mut last_error = None;
        for attempt in 0..config.max_retries {
            let response = http_client
                .put(&url)
                .header("Content-Type", "application/ld+json")
                .body(serde_json::to_string(&jsonld).map_err(|e| JssSyncError::SerializationError(e.to_string()))?)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    return Ok(());
                }
                Ok(resp) => {
                    last_error = Some(JssSyncError::JssServerError {
                        status: resp.status().as_u16(),
                        message: format!("Failed to sync resource: {}", resource.iri),
                    });
                }
                Err(e) => {
                    last_error = Some(JssSyncError::HttpError(e));
                }
            }

            if attempt < config.max_retries - 1 {
                tokio::time::sleep(Duration::from_millis(config.retry_delay_ms * (attempt as u64 + 1))).await;
            }
        }

        Err(last_error.unwrap_or(JssSyncError::SerializationError("Unknown error after retries".to_string())))
    }

    /// Delete a resource from JSS
    async fn delete_resource_from_jss(
        config: &JssSyncConfig,
        http_client: &Client,
        iri: &str,
    ) -> Result<()> {
        let slug = Self::iri_to_slug(iri);
        let resource_path = format!("{}/{}.jsonld", config.public_ontology_path, slug);
        let url = format!("{}{}", config.jss_config.base_url, resource_path);

        let response = http_client.delete(&url).send().await?;

        if response.status().is_success() || response.status().as_u16() == 404 {
            Ok(())
        } else {
            Err(JssSyncError::JssServerError {
                status: response.status().as_u16(),
                message: format!("Failed to delete resource: {}", iri),
            })
        }
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
        Self::sync_single_resource(&self.config, &self.http_client, resource).await
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

    // ========================================================================
    // Conversion Helpers
    // ========================================================================

    fn owl_class_to_resource(class: &OwlClass) -> OntologyResource {
        OntologyResource {
            iri: class.iri.clone(),
            label: class.label.clone(),
            description: class.description.clone(),
            resource_type: OntologyResourceType::Class,
            parent_iris: class.parent_classes.clone(),
            properties: class.properties.clone(),
            source_domain: class.source_domain.clone(),
            quality_score: class.quality_score,
            last_modified: class.last_synced,
        }
    }

    fn owl_property_to_resource(prop: &OwlProperty) -> OntologyResource {
        OntologyResource {
            iri: prop.iri.clone(),
            label: prop.label.clone(),
            description: None,
            resource_type: OntologyResourceType::Property,
            parent_iris: Vec::new(),
            properties: HashMap::new(),
            source_domain: None,
            quality_score: prop.quality_score,
            last_modified: None,
        }
    }

    fn compute_resource_checksum(resource: &OntologyResource) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        resource.iri.hash(&mut hasher);
        resource.label.hash(&mut hasher);
        resource.description.hash(&mut hasher);
        for parent in &resource.parent_iris {
            parent.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
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
        Self::resource_to_jsonld_static(resource)
    }

    fn resource_to_jsonld_static(resource: &OntologyResource) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "@context": {
                "@vocab": "http://visionflow.io/ontology#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "vf": "http://visionflow.io/ontology#"
            },
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
            "vf:lastModified": resource.last_modified.map(|dt| dt.to_rfc3339()),
            "vf:syncedAt": Utc::now().to_rfc3339(),
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

// ============================================================================
// Tests
// ============================================================================

// TODO: Add integration tests for error paths:
// - Network timeout during sync (test with mock server that delays response > http_timeout_secs)
// - Invalid response format (test with mock server returning malformed JSON/non-JSON)
// - Partial sync failure (test batch where some resources fail, verify others succeed)
// - Retry exhaustion (test max_retries reached, verify proper error propagation)
// - JSS server errors (test 4xx/5xx responses, verify JssSyncError::JssServerError)
// - Neo4j connection failures (test OntologyRepository errors)
// - Change event channel overflow (test with > 1000 pending events)
// See: tests/integration/jss_sync_tests.rs (to be created)

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
        assert!(status.last_full_sync.is_none());
        assert!(status.last_incremental_sync.is_none());
    }

    #[tokio::test]
    async fn test_metrics_initialization() {
        let service = JssSyncService::from_env();
        let metrics = service.get_metrics().await;

        assert_eq!(metrics.total_full_syncs, 0);
        assert_eq!(metrics.total_incremental_syncs, 0);
        assert_eq!(metrics.total_resources_synced, 0);
        assert_eq!(metrics.total_errors, 0);
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

        let jsonld = JssSyncService::resource_to_jsonld_static(&resource).unwrap();

        assert_eq!(jsonld["@id"], "http://example.org/ontology#Person");
        assert_eq!(jsonld["@type"], "owl:Class");
        assert_eq!(jsonld["rdfs:label"], "Person");
        assert_eq!(jsonld["rdfs:comment"], "A human being");
        // Compare as f64 with tolerance for floating-point precision
        let score = jsonld["vf:qualityScore"].as_f64().unwrap();
        assert!((score - 0.95).abs() < 0.001, "Expected ~0.95, got {}", score);
        assert_eq!(jsonld["vf:sourceDomain"], "core");

        let subclass_of = jsonld["rdfs:subClassOf"].as_array().unwrap();
        assert_eq!(subclass_of.len(), 1);
        assert_eq!(subclass_of[0]["@id"], "http://example.org/ontology#Agent");
    }

    #[test]
    fn test_change_event_types() {
        let event = Neo4jChangeEvent {
            change_type: Neo4jChangeType::ClassCreated,
            affected_iris: vec!["http://example.org/Test".to_string()],
            timestamp: Utc::now(),
            source: ChangeSource::GitHubSync,
        };

        assert_eq!(event.change_type, Neo4jChangeType::ClassCreated);
        assert_eq!(event.source, ChangeSource::GitHubSync);
        assert_eq!(event.affected_iris.len(), 1);
    }

    #[test]
    fn test_compute_resource_checksum() {
        let resource1 = OntologyResource {
            iri: "http://example.org/ontology#Person".to_string(),
            label: Some("Person".to_string()),
            description: None,
            resource_type: OntologyResourceType::Class,
            parent_iris: vec![],
            properties: HashMap::new(),
            source_domain: None,
            quality_score: None,
            last_modified: None,
        };

        let resource2 = OntologyResource {
            iri: "http://example.org/ontology#Person".to_string(),
            label: Some("Person Updated".to_string()), // Different label
            description: None,
            resource_type: OntologyResourceType::Class,
            parent_iris: vec![],
            properties: HashMap::new(),
            source_domain: None,
            quality_score: None,
            last_modified: None,
        };

        let checksum1 = JssSyncService::compute_resource_checksum(&resource1);
        let checksum2 = JssSyncService::compute_resource_checksum(&resource2);

        // Same resource should produce same checksum
        assert_eq!(checksum1, JssSyncService::compute_resource_checksum(&resource1));

        // Different content should produce different checksum
        assert_ne!(checksum1, checksum2);
    }

    #[test]
    fn test_config_defaults() {
        let config = JssSyncConfig::default();

        assert_eq!(config.sync_interval_secs, 300);
        assert_eq!(config.batch_size, 100);
        assert!(config.enable_auto_sync);
        assert!(config.enable_startup_sync);
        assert_eq!(config.max_retries, 3);
    }
}

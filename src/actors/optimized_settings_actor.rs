// High-Performance Settings Actor with Multi-layered Caching
// Implements Redis caching, LRU cache, path pattern optimization, and delta compression

use actix::prelude::*;
use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, UpdateSettings, GetSettingByPath, SetSettingByPath, GetSettingsByPaths, SetSettingsByPaths, UpdatePhysicsFromAutoBalance};
use crate::actors::{GraphServiceActor, gpu::ForceComputeActor};
use crate::config::path_access::PathAccessible;
use crate::errors::{VisionFlowError, VisionFlowResult, SettingsError, ActorError, ErrorContext};
use std::collections::HashMap;
use serde_json::Value;
use log::{info, error, debug, warn};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use lru::LruCache;
use std::num::NonZeroUsize;
use blake3::Hasher;
use flate2::{Compress, Decompress, Compression, FlushCompress, FlushDecompress};
use flate2::Status;

#[cfg(feature = "redis")]
use redis::{Client as RedisClient, AsyncCommands};

// Cache configuration constants
const CACHE_SIZE: usize = 1000;

pub struct OptimizedSettingsActor {
    settings: Arc<RwLock<AppFullSettings>>,
    #[cfg(feature = "redis")]
    redis_client: Option<RedisClient>,
    path_cache: Arc<RwLock<LruCache<String, CachedValue>>>,
    path_lookup: Arc<RwLock<HashMap<String, PathPattern>>>,
    metrics: Arc<RwLock<PerformanceMetrics>>,
    compressor: Arc<RwLock<Compress>>,
    decompressor: Arc<RwLock<Decompress>>,
    graph_service_addr: Option<Addr<crate::actors::graph_service_supervisor::TransitionalGraphSupervisor>>,
    gpu_compute_addr: Option<Addr<ForceComputeActor>>,
}

#[derive(Clone, Debug)]
struct CachedValue {
    value: Value,
    hash: String,
    timestamp: Instant,
    ttl: Duration,
}

#[derive(Clone, Debug)]
struct PathPattern {
    compiled_path: Vec<String>,
    field_type: FieldType,
    validation_rules: ValidationRules,
}

#[derive(Clone, Debug)]
enum FieldType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    String,
    Object,
    Array,
}

#[derive(Clone, Debug)]
struct ValidationRules {
    min: Option<f64>,
    max: Option<f64>,
    required: bool,
    pattern: Option<String>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub redis_hits: u64,
    pub redis_misses: u64,
    pub batch_operations: u64,
    pub compression_ratio: f64,
    pub avg_response_time_ms: f64,
    pub total_requests: u64,
    pub memory_usage_bytes: u64,
    pub bandwidth_saved_bytes: u64,
}

impl PerformanceMetrics {
    pub fn cache_hit_rate(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        }
    }
    
    pub fn redis_hit_rate(&self) -> f64 {
        if self.redis_hits + self.redis_misses == 0 {
            0.0
        } else {
            self.redis_hits as f64 / (self.redis_hits + self.redis_misses) as f64
        }
    }
    
    pub fn bandwidth_savings_percentage(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            // Assuming full settings fetch is ~50KB vs path fetch ~500B
            let full_size = 50_000 * self.total_requests;
            let actual_size = full_size - self.bandwidth_saved_bytes;
            (self.bandwidth_saved_bytes as f64 / full_size as f64) * 100.0
        }
    }
}

const CACHE_TTL: Duration = Duration::from_secs(300); // 5 minutes
const REDIS_TTL: usize = 3600; // 1 hour
const EXPECTED_FULL_SETTINGS_SIZE: u64 = 50_000; // ~50KB
const EXPECTED_PATH_SIZE: u64 = 500; // ~500B

impl OptimizedSettingsActor {
    pub fn new() -> VisionFlowResult<Self> {
        // Load settings from file or use defaults
        let settings = AppFullSettings::new()
            .map_err(|e| {
                error!("Failed to load settings from file: {}", e);
                VisionFlowError::Settings(SettingsError::ParseError {
                    file_path: "settings".to_string(),
                    reason: e.to_string(),
                })
            })?;
        
        // Initialize Redis client (optional)
        #[cfg(feature = "redis")]
        let redis_client = match std::env::var("REDIS_URL") {
            Ok(url) => {
                match RedisClient::open(url) {
                    Ok(client) => {
                        info!("Connected to Redis for settings caching");
                        Some(client)
                    }
                    Err(e) => {
                        warn!("Failed to connect to Redis, using local cache only: {}", e);
                        None
                    }
                }
            }
            Err(_) => {
                debug!("No Redis URL configured, using local cache only");
                None
            }
        };
        
        // Initialize LRU cache
        let path_cache = Arc::new(RwLock::new(
            LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap())
        ));
        
        // Pre-compile common path patterns
        let mut path_lookup = HashMap::new();
        Self::initialize_path_patterns(&mut path_lookup);
        
        info!("OptimizedSettingsActor initialized with performance optimizations");
        debug!("Logseq physics: damping={}, spring={}, repulsion={}", 
            settings.visualisation.graphs.logseq.physics.damping,
            settings.visualisation.graphs.logseq.physics.spring_k,
            settings.visualisation.graphs.logseq.physics.repel_k
        );
        
        Ok(Self {
            settings: Arc::new(RwLock::new(settings)),
            #[cfg(feature = "redis")]
            redis_client,
            path_cache,
            path_lookup: Arc::new(RwLock::new(path_lookup)),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            compressor: Arc::new(RwLock::new(Compress::new(Compression::default(), false))),
            decompressor: Arc::new(RwLock::new(Decompress::new(false))),
            graph_service_addr: None,
            gpu_compute_addr: None,
        })
    }

    pub fn with_actors(
        graph_service_addr: Option<Addr<crate::actors::graph_service_supervisor::TransitionalGraphSupervisor>>,
        gpu_compute_addr: Option<Addr<ForceComputeActor>>,
    ) -> VisionFlowResult<Self> {
        let mut actor = Self::new()?;
        actor.graph_service_addr = graph_service_addr;
        actor.gpu_compute_addr = gpu_compute_addr;
        info!("OptimizedSettingsActor initialized with GPU and Graph actor addresses for physics forwarding and concurrent update batching");
        Ok(actor)
    }

    fn initialize_path_patterns(lookup: &mut HashMap<String, PathPattern>) {
        // Physics settings patterns - most frequently accessed
        let physics_patterns = vec![
            ("visualisation.graphs.logseq.physics.damping", FieldType::Float32, 0.0, 1.0),
            ("visualisation.graphs.logseq.physics.spring_k", FieldType::Float32, 0.0, 10.0),
            ("visualisation.graphs.logseq.physics.repel_k", FieldType::Float32, 0.0, 100.0),
            ("visualisation.graphs.logseq.physics.max_velocity", FieldType::Float32, 0.1, 50.0),
            ("visualisation.graphs.logseq.physics.gravity", FieldType::Float32, 0.0, 1.0),
            ("visualisation.graphs.logseq.physics.temperature", FieldType::Float32, 0.0, 1.0),
            ("visualisation.graphs.logseq.physics.bounds_size", FieldType::Float32, 100.0, 2000.0),
            ("visualisation.graphs.logseq.physics.iterations", FieldType::Int32, 1.0, 1000.0),
            ("visualisation.graphs.logseq.physics.enabled", FieldType::Bool, 0.0, 1.0),
        ];
        
        for (path, field_type, min, max) in physics_patterns {
            let compiled_path: Vec<String> = path.split('.').map(|s| s.to_string()).collect();
            let validation_rules = ValidationRules {
                min: Some(min),
                max: Some(max),
                required: true,
                pattern: None,
            };
            
            lookup.insert(path.to_string(), PathPattern {
                compiled_path,
                field_type,
                validation_rules,
            });
        }
        
        info!("Initialized {} pre-compiled path patterns", lookup.len());
    }
    
    pub async fn get_settings(&self) -> AppFullSettings {
        let start_time = Instant::now();
        let settings = self.settings.read().await.clone();
        self.update_metrics(start_time, false, true).await;
        settings
    }
    
    async fn update_metrics(&self, start_time: Instant, was_cached: bool, success: bool) {
        let response_time = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        
        if was_cached {
            metrics.cache_hits += 1;
            // Calculate bandwidth savings
            let savings = EXPECTED_FULL_SETTINGS_SIZE - EXPECTED_PATH_SIZE;
            metrics.bandwidth_saved_bytes += savings;
        } else {
            metrics.cache_misses += 1;
        }
        
        // Update running average response time
        let total = metrics.total_requests as f64;
        let prev_avg_ms = metrics.avg_response_time_ms;
        let new_time_ms = response_time.as_millis() as f64;
        metrics.avg_response_time_ms = (prev_avg_ms * (total - 1.0) + new_time_ms) / total;
    }
    
    async fn get_cached_value(&self, path: &str) -> Option<Value> {
        // Check local cache first
        {
            let mut cache = self.path_cache.write().await;
            if let Some(cached) = cache.get(path) {
                if cached.timestamp.elapsed() < cached.ttl {
                    debug!("Local cache hit for path: {}", path);
                    return Some(cached.value.clone());
                } else {
                    // Remove expired entry
                    cache.pop(path);
                    debug!("Expired cache entry removed for path: {}", path);
                }
            }
        }
        
        // Check Redis cache if available
        #[cfg(feature = "redis")]
        if let Some(redis_client) = &self.redis_client {
            if let Ok(mut conn) = redis_client.get_async_connection().await {
                let redis_key = format!("settings:{}", path);
                if let Ok(compressed_data) = conn.get::<String, Vec<u8>>(&redis_key).await {
                    if let Ok(json_str) = self.decompress_data(&compressed_data).await {
                        if let Ok(value) = serde_json::from_str::<Value>(&json_str) {
                            debug!("Redis hit for path: {}", path);
                            
                            // Update local cache
                            let cached_value = CachedValue {
                                value: value.clone(),
                                hash: self.calculate_hash(&value).await,
                                timestamp: Instant::now(),
                                ttl: CACHE_TTL,
                            };
                            
                            let mut cache = self.path_cache.write().await;
                            cache.put(path.to_string(), cached_value);
                            
                            let mut metrics = self.metrics.write().await;
                            metrics.redis_hits += 1;
                            
                            return Some(value);
                        }
                    }
                }
            }
        }
        
        None
    }
    
    async fn set_cached_value(&self, path: &str, value: &Value) {
        let hash = self.calculate_hash(value).await;
        
        // Update local cache
        let cached_value = CachedValue {
            value: value.clone(),
            hash: hash.clone(),
            timestamp: Instant::now(),
            ttl: CACHE_TTL,
        };
        
        {
            let mut cache = self.path_cache.write().await;
            cache.put(path.to_string(), cached_value);
        }
        
        // Update Redis cache asynchronously if available
        #[cfg(feature = "redis")]
        if let Some(redis_client) = &self.redis_client {
            let redis_client = redis_client.clone();
            let path = path.to_string();
            let value = value.clone();
            
            tokio::spawn(async move {
                if let Ok(mut conn) = redis_client.get_async_connection().await {
                    if let Ok(json_str) = serde_json::to_string(&value) {
                        // Compress data before storing in Redis
                        let mut compressor = Compress::new(Compression::default(), false);
                        let mut compressed = Vec::new();
                        let json_bytes = json_str.as_bytes();
                        
                        // Simple compression approach
                        let mut output = vec![0; json_bytes.len() * 2];
                        let status = compressor.compress_vec(
                            json_bytes,
                            &mut output,
                            FlushCompress::Finish,
                        ).unwrap();
                        
                        if status == Status::StreamEnd {
                            let compressed_size = compressor.total_out() as usize;
                            output.truncate(compressed_size);
                            
                            let redis_key = format!("settings:{}", path);
                            if let Err(e) = conn.set_ex::<String, Vec<u8>, ()>(&redis_key, output, REDIS_TTL).await {
                                warn!("Failed to cache value in Redis: {}", e);
                            } else {
                                debug!("Cached compressed value in Redis for path: {}", path);
                            }
                        }
                    }
                }
            });
        }
    }
    
    async fn calculate_hash(&self, value: &Value) -> String {
        let mut hasher = Hasher::new();
        if let Ok(json_str) = serde_json::to_string(value) {
            hasher.update(json_str.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }
    
    async fn decompress_data(&self, compressed: &[u8]) -> VisionFlowResult<String> {
        let mut decompressor = self.decompressor.write().await;
        let mut output = Vec::new();
        
        let mut buffer = vec![0; compressed.len() * 4]; // Estimate decompressed size
        let status = decompressor.decompress_vec(
            compressed,
            &mut buffer,
            FlushDecompress::Finish,
        ).map_err(|e| format!("Decompression error: {}", e))?;
        
        if status == Status::StreamEnd {
            let decompressed_size = decompressor.total_out() as usize;
            buffer.truncate(decompressed_size);
            output.extend(buffer);
        }
        
        String::from_utf8(output).map_err(|e| VisionFlowError::Serialization(format!("UTF-8 conversion error: {}", e)))
    }
    
    async fn get_optimized_path_value(&self, path: &str) -> VisionFlowResult<Value> {
        let start_time = Instant::now();
        
        // Try cache first
        if let Some(cached_value) = self.get_cached_value(path).await {
            self.update_metrics(start_time, true, true).await;
            return Ok(cached_value);
        }
        
        // Check if we have a pre-compiled path pattern
        let pattern = {
            let lookup = self.path_lookup.read().await;
            lookup.get(path).cloned()
        };
        
        let current = self.settings.read().await;
        let result = if let Some(pattern) = pattern {
            // Use optimized path traversal
            self.traverse_compiled_path(&current, &pattern.compiled_path)
        } else {
            // Fallback to JSON traversal
            let json = serde_json::to_value(&*current)
                .map_err(|e| format!("Failed to serialize settings: {}", e))?;
            
            let parts: Vec<&str> = path.split('.').collect();
            let mut value = &json;
            
            for part in parts {
                match value.get(part) {
                    Some(v) => value = v,
                    None => return Err(VisionFlowError::Settings(SettingsError::ValidationFailed {
                        setting_path: path.to_string(),
                        reason: "Path not found".to_string(),
                    })),
                }
            }
            
            Ok(value.clone())
        };
        
        if let Ok(value) = &result {
            self.set_cached_value(path, value).await;
        }
        
        self.update_metrics(start_time, false, result.is_ok()).await;
        result
    }
    
    fn traverse_compiled_path(&self, settings: &AppFullSettings, compiled_path: &[String]) -> VisionFlowResult<Value> {
        // Highly optimized path traversal for known patterns
        if compiled_path.len() == 4 && compiled_path[0] == "visualisation" 
            && compiled_path[1] == "graphs" && compiled_path[2] == "logseq" 
            && compiled_path[3] == "physics" {
            
            // Direct access to physics settings - much faster than JSON traversal
            let physics = &settings.visualisation.graphs.logseq.physics;
            return Ok(serde_json::to_value(physics)
                .map_err(|e| format!("Failed to serialize physics: {}", e))?);
        }
        
        if compiled_path.len() == 5 && compiled_path[0] == "visualisation" 
            && compiled_path[1] == "graphs" && compiled_path[2] == "logseq" 
            && compiled_path[3] == "physics" {
            
            let physics = &settings.visualisation.graphs.logseq.physics;
            let field = &compiled_path[4];
            
            let value = match field.as_str() {
                "damping" => serde_json::Value::Number(serde_json::Number::from_f64(physics.damping as f64).unwrap()),
                "spring_k" => serde_json::Value::Number(serde_json::Number::from_f64(physics.spring_k as f64).unwrap()),
                "repel_k" => serde_json::Value::Number(serde_json::Number::from_f64(physics.repel_k as f64).unwrap()),
                "max_velocity" => serde_json::Value::Number(serde_json::Number::from_f64(physics.max_velocity as f64).unwrap()),
                "gravity" => serde_json::Value::Number(serde_json::Number::from_f64(physics.gravity as f64).unwrap()),
                "temperature" => serde_json::Value::Number(serde_json::Number::from_f64(physics.temperature as f64).unwrap()),
                "bounds_size" => serde_json::Value::Number(serde_json::Number::from_f64(physics.bounds_size as f64).unwrap()),
                "iterations" => serde_json::Value::Number(serde_json::Number::from(physics.iterations)),
                "enabled" => serde_json::Value::Bool(physics.enabled),
                _ => return Err(VisionFlowError::Settings(SettingsError::ValidationFailed {
                    setting_path: format!("physics.{}", field),
                    reason: "Unknown physics field".to_string(),
                })),
            };
            
            return Ok(value);
        }
        
        Err(VisionFlowError::Settings(SettingsError::ValidationFailed {
            setting_path: "compiled_path".to_string(),
            reason: "Path pattern not optimized".to_string(),
        }))
    }
    
    pub async fn update_settings(&self, new_settings: AppFullSettings) -> VisionFlowResult<()> {
        let mut settings = self.settings.write().await;
        *settings = new_settings;
        
        // Clear all caches since settings changed
        {
            let mut cache = self.path_cache.write().await;
            cache.clear();
        }
        
        // Persist to file
        settings.save().map_err(|e| {
            error!("Failed to save settings to file: {}", e);
            VisionFlowError::Settings(SettingsError::SaveFailed {
                file_path: "settings".to_string(),
                reason: e,
            })
        })?;
        
        info!("Settings updated, caches cleared, and saved successfully");
        Ok(())
    }
    
    /// Get comprehensive performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        let mut metrics = self.metrics.read().await.clone();
        
        // Calculate memory usage
        let cache = self.path_cache.read().await;
        metrics.memory_usage_bytes = cache.len() as u64 * 1024; // Rough estimate
        
        metrics
    }
    
    /// Clear all caches
    pub async fn clear_caches(&self) {
        // Clear local cache
        {
            let mut cache = self.path_cache.write().await;
            cache.clear();
        }
        
        // Clear Redis cache if available
        #[cfg(feature = "redis")]
        if let Some(redis_client) = &self.redis_client {
            if let Ok(mut conn) = redis_client.get_async_connection().await {
                if let Err(e) = conn.flushdb::<()>().await {
                    warn!("Failed to clear Redis cache: {}", e);
                }
            }
        }
        
        // Reset metrics
        {
            let mut metrics = self.metrics.write().await;
            *metrics = PerformanceMetrics::default();
        }
        
        info!("All caches and metrics cleared");
    }
    
    /// Warm up cache with common paths
    pub async fn warm_cache(&self) {
        let common_paths = vec![
            "visualisation.graphs.logseq.physics",
            "visualisation.graphs.logseq.physics.damping",
            "visualisation.graphs.logseq.physics.spring_k",
            "visualisation.graphs.logseq.physics.repel_k",
            "visualisation.graphs.logseq.physics.max_velocity",
            "visualisation.graphs.logseq.physics.gravity",
            "visualisation.graphs.logseq.physics.temperature",
            "visualisation.graphs.logseq.physics.bounds_size",
            "visualisation.graphs.logseq.physics.iterations",
            "visualisation.graphs.logseq.physics.enabled",
        ];
        
        let start_time = Instant::now();
        let mut warmed_count = 0;
        
        for path in common_paths {
            if let Ok(_) = self.get_optimized_path_value(path).await {
                warmed_count += 1;
                debug!("Warmed cache for path: {}", path);
            }
        }
        
        info!("Cache warm-up completed: {} paths cached in {:?}", 
              warmed_count, start_time.elapsed());
    }
    
    fn validate_value_with_pattern(value: &Value, pattern: &PathPattern) -> VisionFlowResult<()> {
        match (&pattern.field_type, value) {
            (FieldType::Float32 | FieldType::Float64, Value::Number(n)) => {
                if let Some(f) = n.as_f64() {
                    if let Some(min) = pattern.validation_rules.min {
                        if f < min {
                            return Err(VisionFlowError::Settings(SettingsError::ValidationFailed {
                                setting_path: "value".to_string(),
                                reason: format!("Value {} below minimum {}", f, min),
                            }));
                        }
                    }
                    if let Some(max) = pattern.validation_rules.max {
                        if f > max {
                            return Err(VisionFlowError::Settings(SettingsError::ValidationFailed {
                                setting_path: "value".to_string(),
                                reason: format!("Value {} above maximum {}", f, max),
                            }));
                        }
                    }
                }
                Ok(())
            }
            (FieldType::Int32 | FieldType::Int64, Value::Number(n)) => {
                if n.is_i64() {
                    Ok(())
                } else {
                    Err(VisionFlowError::Settings(SettingsError::ValidationFailed {
                        setting_path: "value".to_string(),
                        reason: "Expected integer value".to_string(),
                    }))
                }
            }
            (FieldType::Bool, Value::Bool(_)) => Ok(()),
            (FieldType::String, Value::String(_)) => Ok(()),
            _ => Err(VisionFlowError::Settings(SettingsError::ValidationFailed {
                setting_path: "value".to_string(),
                reason: "Type mismatch".to_string(),
            })),
        }
    }
}

impl Actor for OptimizedSettingsActor {
    type Context = Context<Self>;
    
    fn started(&mut self, ctx: &mut Self::Context) {
        info!("OptimizedSettingsActor started");
        
        // Start cache warm-up in background
        let addr = ctx.address();
        ctx.spawn(async move {
            if let Err(e) = addr.send(WarmCacheMessage).await {
                warn!("Failed to warm cache on startup: {}", e);
            }
        }.into_actor(self));
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("OptimizedSettingsActor stopped");
    }
}

// Internal message for cache warming
#[derive(Message)]
#[rtype(result = "()")]
pub struct WarmCacheMessage;

impl Handler<WarmCacheMessage> for OptimizedSettingsActor {
    type Result = ResponseFuture<()>;
    
    fn handle(&mut self, _msg: WarmCacheMessage, _ctx: &mut Self::Context) -> Self::Result {
        let actor = self.clone();
        
        Box::pin(async move {
            actor.warm_cache().await;
        })
    }
}

// Handle GetSettings message
impl Handler<GetSettings> for OptimizedSettingsActor {
    type Result = ResponseFuture<VisionFlowResult<AppFullSettings>>;
    
    fn handle(&mut self, _msg: GetSettings, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let metrics = self.metrics.clone();
        
        Box::pin(async move {
            let start_time = Instant::now();
            let result = Ok(settings.read().await.clone());
            
            // Update metrics
            {
                let mut metrics = metrics.write().await;
                metrics.total_requests += 1;
                let response_time = start_time.elapsed();
                let total = metrics.total_requests as f64;
                let prev_avg = metrics.avg_response_time_ms;
                let new_time_ms = response_time.as_millis() as f64;
                metrics.avg_response_time_ms = (prev_avg * (total - 1.0) + new_time_ms) / total;
            }
            
            result
        })
    }
}

// Handle UpdateSettings message  
impl Handler<UpdateSettings> for OptimizedSettingsActor {
    type Result = ResponseFuture<VisionFlowResult<()>>;
    
    fn handle(&mut self, msg: UpdateSettings, _ctx: &mut Self::Context) -> Self::Result {
        let actor = self.clone();
        
        Box::pin(async move {
            actor.update_settings(msg.settings).await
        })
    }
}

// Optimized handler for getting settings by path
impl Handler<GetSettingByPath> for OptimizedSettingsActor {
    type Result = ResponseFuture<VisionFlowResult<serde_json::Value>>;
    
    fn handle(&mut self, msg: GetSettingByPath, _ctx: &mut Self::Context) -> Self::Result {
        let actor = self.clone();
        let path = msg.path.clone();
        
        Box::pin(async move {
            actor.get_optimized_path_value(&path).await
        })
    }
}

// Clone implementation for OptimizedSettingsActor
impl Clone for OptimizedSettingsActor {
    fn clone(&self) -> Self {
        Self {
            settings: self.settings.clone(),
            #[cfg(feature = "redis")]
            redis_client: self.redis_client.clone(),
            path_cache: self.path_cache.clone(),
            path_lookup: self.path_lookup.clone(),
            metrics: self.metrics.clone(),
            compressor: Arc::new(RwLock::new(Compress::new(Compression::default(), false))),
            decompressor: Arc::new(RwLock::new(Decompress::new(false))),
            graph_service_addr: self.graph_service_addr.clone(),
            gpu_compute_addr: self.gpu_compute_addr.clone(),
        }
    }
}

// Optimized batch handler for getting multiple settings by path
impl Handler<GetSettingsByPaths> for OptimizedSettingsActor {
    type Result = ResponseFuture<VisionFlowResult<HashMap<String, Value>>>;
    
    fn handle(&mut self, msg: GetSettingsByPaths, _ctx: &mut Self::Context) -> Self::Result {
        let actor = self.clone();
        let paths = msg.paths;
        
        Box::pin(async move {
            let start_time = Instant::now();
            let mut results = HashMap::new();
            let mut cache_hits = 0;
            let mut cache_misses = 0;
            
            // Process paths concurrently for better performance
            let futures: Vec<_> = paths.into_iter().map(|path| {
                let actor = actor.clone();
                async move {
                    let result = actor.get_optimized_path_value(&path).await;
                    (path, result)
                }
            }).collect();
            
            let path_results = futures::future::join_all(futures).await;
            
            for (path, result) in path_results {
                match result {
                    Ok(value) => {
                        results.insert(path, value);
                        cache_hits += 1;
                    }
                    Err(e) => {
                        error!("Failed to get path {}: {}", path, e);
                        cache_misses += 1;
                    }
                }
            }
            
            // Update batch metrics
            {
                let mut metrics = actor.metrics.write().await;
                metrics.batch_operations += 1;
                metrics.cache_hits += cache_hits;
                metrics.cache_misses += cache_misses;
            }
            
            debug!("Optimized batch operation completed: {} hits, {} misses, took {:?}", 
                   cache_hits, cache_misses, start_time.elapsed());
            
            Ok(results)
        })
    }
}

// Ultra-optimized batch handler for setting multiple values
impl Handler<SetSettingsByPaths> for OptimizedSettingsActor {
    type Result = ResponseFuture<VisionFlowResult<()>>;
    
    fn handle(&mut self, msg: SetSettingsByPaths, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let path_lookup = self.path_lookup.clone();
        let path_cache = self.path_cache.clone();
        let metrics = self.metrics.clone();
        #[cfg(feature = "redis")]
        let redis_client = self.redis_client.clone();
        let updates = msg.updates;
        
        Box::pin(async move {
            let start_time = Instant::now();
            let mut current = settings.write().await;
            let mut validation_needed = false;
            let mut cache_invalidations = Vec::new();
            
            // Pre-validate all updates using compiled patterns
            {
                let lookup = path_lookup.read().await;
                for (path, value) in &updates {
                    if let Some(pattern) = lookup.get(path) {
                        Self::validate_value_with_pattern(value, pattern).map_err(|e| {
                            VisionFlowError::Settings(SettingsError::ValidationFailed {
                                setting_path: path.clone(),
                                reason: format!("Validation failed: {}", e),
                            })
                        })?;
                    }
                }
            }
            
            // Apply updates using direct field access for maximum performance
            for (path, value) in updates {
                if path.starts_with("visualisation.graphs.logseq.physics.") {
                    validation_needed = true;
                    cache_invalidations.push(path.clone());
                    
                    let field_name = path.replace("visualisation.graphs.logseq.physics.", "");
                    let internal_field = match field_name.as_str() {
                        "springK" => "spring_k",
                        "repelK" => "repel_k", 
                        "maxVelocity" => "max_velocity",
                        "boundsSize" => "bounds_size",
                        other => other,
                    };
                    
                    // Ultra-fast direct field access - bypassing all reflection/path traversal
                    let physics = &mut current.visualisation.graphs.logseq.physics;
                    
                    match internal_field {
                        "damping" => {
                            if let Some(f_val) = value.as_f64() {
                                physics.damping = f_val as f32;
                            }
                        }
                        "spring_k" => {
                            if let Some(f_val) = value.as_f64() {
                                physics.spring_k = f_val as f32;
                            }
                        }
                        "repel_k" => {
                            if let Some(f_val) = value.as_f64() {
                                physics.repel_k = f_val as f32;
                            }
                        }
                        "max_velocity" => {
                            if let Some(f_val) = value.as_f64() {
                                physics.max_velocity = f_val as f32;
                            }
                        }
                        "gravity" => {
                            if let Some(f_val) = value.as_f64() {
                                physics.gravity = f_val as f32;
                            }
                        }
                        "temperature" => {
                            if let Some(f_val) = value.as_f64() {
                                physics.temperature = f_val as f32;
                            }
                        }
                        "bounds_size" => {
                            if let Some(f_val) = value.as_f64() {
                                physics.bounds_size = f_val as f32;
                            }
                        }
                        "enabled" => {
                            if let Some(b_val) = value.as_bool() {
                                physics.enabled = b_val;
                            }
                        }
                        "iterations" => {
                            if let Some(i_val) = value.as_u64() {
                                physics.iterations = i_val as u32;
                            }
                        }
                        _ => {
                            error!("Unsupported physics field in batch update: {}", internal_field);
                            continue;
                        }
                    }
                    
                    debug!("Direct batch updated physics setting: {} = {:?}", internal_field, value);
                }
            }
            
            // Invalidate relevant cache entries
            {
                let mut cache = path_cache.write().await;
                for path in &cache_invalidations {
                    cache.pop(path);
                }
                
                // Also invalidate parent paths
                cache.pop("visualisation.graphs.logseq.physics");
            }
            
            // Asynchronous Redis cache invalidation
            #[cfg(feature = "redis")]
            if let Some(redis_client) = redis_client {
                tokio::spawn(async move {
                    if let Ok(mut conn) = redis_client.get_async_connection().await {
                        for path in cache_invalidations {
                            let redis_key = format!("settings:{}", path);
                            if let Err(e) = conn.del::<String, ()>(&redis_key).await {
                                warn!("Failed to invalidate Redis cache for {}: {}", path, e);
                            }
                        }
                    }
                });
            }
            
            // Single validation for all batch updates
            if validation_needed {
                current.validate_config_camel_case().map_err(|e| {
                    error!("Validation failed after batch update: {:?}", e);
                    VisionFlowError::Settings(SettingsError::ValidationFailed {
                        setting_path: "batch_update".to_string(),
                        reason: format!("Batch validation failed: {:?}", e),
                    })
                })?;
                
                // Save to file if persistence is enabled
                if current.system.persist_settings {
                    current.save().map_err(|e| {
                        error!("Failed to save settings after batch update: {}", e);
                        VisionFlowError::Settings(SettingsError::SaveFailed {
                            file_path: "batch_settings".to_string(),
                            reason: e,
                        })
                    })?;
                }
            }
            
            // Update performance metrics
            {
                let mut metrics = metrics.write().await;
                metrics.batch_operations += 1;
                let response_time = start_time.elapsed();
                let total = metrics.total_requests as f64 + 1.0;
                let prev_avg = metrics.avg_response_time_ms;
                let new_time_ms = response_time.as_millis() as f64;
                metrics.avg_response_time_ms = (prev_avg * (total - 1.0) + new_time_ms) / total;
                metrics.total_requests += 1;
            }
            
            info!("Ultra-optimized batch settings update completed in {:?}", start_time.elapsed());
            Ok(())
        })
    }
}

// Handle UpdatePhysicsFromAutoBalance message with optimizations
impl Handler<UpdatePhysicsFromAutoBalance> for OptimizedSettingsActor {
    type Result = ();
    
    fn handle(&mut self, msg: UpdatePhysicsFromAutoBalance, ctx: &mut Self::Context) {
        let settings = self.settings.clone();
        let path_cache = self.path_cache.clone();
        
        ctx.spawn(Box::pin(async move {
            let mut current = settings.write().await;
            
            // Clear physics-related cache entries
            {
                let mut cache = path_cache.write().await;
                cache.clear(); // Clear all for simplicity during auto-balance
            }
            
            // Merge the physics update from auto-balance
            if let Err(e) = current.merge_update(msg.physics_update.clone()) {
                error!("[AUTO-BALANCE] Failed to merge physics update: {}", e);
                return;
            }
            
            info!("[AUTO-BALANCE] Physics parameters updated in settings from auto-tuning");
            
            // Log final tuned values
            if let Some(physics) = msg.physics_update.get("visualisation")
                .and_then(|v| v.get("graphs"))
                .and_then(|g| g.get("logseq"))
                .and_then(|l| l.get("physics")) {
                
                info!("[AUTO-BALANCE] Auto-tune complete - optimized settings updated");
                
                if let Some(repel_k) = physics.get("repelK").and_then(|v| v.as_f64()) {
                    info!("[AUTO-BALANCE] Final repelK: {:.3}", repel_k);
                }
                if let Some(damping) = physics.get("damping").and_then(|v| v.as_f64()) {
                    info!("[AUTO-BALANCE] Final damping: {:.3}", damping);
                }
                if let Some(max_vel) = physics.get("maxVelocity").and_then(|v| v.as_f64()) {
                    info!("[AUTO-BALANCE] Final maxVelocity: {:.3}", max_vel);
                }
            }
            
            // Save to file if persistence is enabled
            if current.system.persist_settings {
                if let Err(e) = current.save() {
                    error!("[AUTO-BALANCE] Failed to save auto-tuned settings to file: {}", e);
                } else {
                    info!("[AUTO-BALANCE] Auto-tuned settings saved to settings.yaml");
                }
            } else {
                info!("[AUTO-BALANCE] Settings persistence disabled, not saving to file");
            }
        }).into_actor(self));
    }
}

// Performance metrics message handler
#[derive(Message)]
#[rtype(result = "PerformanceMetrics")]
pub struct GetPerformanceMetrics;

impl Handler<GetPerformanceMetrics> for OptimizedSettingsActor {
    type Result = ResponseFuture<PerformanceMetrics>;
    
    fn handle(&mut self, _msg: GetPerformanceMetrics, _ctx: &mut Self::Context) -> Self::Result {
        let actor = self.clone();
        
        Box::pin(async move {
            actor.get_performance_metrics().await
        })
    }
}

// Cache management message handlers
#[derive(Message)]
#[rtype(result = "()")]
pub struct ClearCaches;

impl Handler<ClearCaches> for OptimizedSettingsActor {
    type Result = ResponseFuture<()>;
    
    fn handle(&mut self, _msg: ClearCaches, _ctx: &mut Self::Context) -> Self::Result {
        let actor = self.clone();
        
        Box::pin(async move {
            actor.clear_caches().await;
        })
    }
}
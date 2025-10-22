//! Test utilities and helper functions for VisionFlow settings tests
//!
//! Provides common mocks, factories, and testing utilities used across
//! all test modules in the settings system.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::RwLock as AsyncRwLock;

// Mock configuration structures (would import from actual config in real implementation)
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestAppSettings {
    pub visualisation: TestVisualisationSettings,
    pub system: TestSystemSettings,
    pub xr: TestXRSettings,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestVisualisationSettings {
    pub glow: TestGlowSettings,
    pub graphs: TestGraphSettings,
    #[serde(rename = "colorSchemes")]
    pub color_schemes: Vec<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestGlowSettings {
    #[serde(rename = "nodeGlowStrength")]
    pub node_glow_strength: f32,
    #[serde(rename = "edgeGlowStrength")]
    pub edge_glow_strength: f32,
    #[serde(rename = "environmentGlowStrength")]
    pub environment_glow_strength: f32,
    #[serde(rename = "baseColor")]
    pub base_color: String,
    #[serde(rename = "emissionColor")]
    pub emission_color: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestGraphSettings {
    pub logseq: TestLogseqGraphSettings,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestLogseqGraphSettings {
    pub physics: TestPhysicsSettings,
    #[serde(rename = "nodeRadius")]
    pub node_radius: f32,
    #[serde(rename = "edgeThickness")]
    pub edge_thickness: f32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestPhysicsSettings {
    #[serde(rename = "springK")]
    pub spring_k: f32,
    #[serde(rename = "repelK")]
    pub repel_k: f32,
    #[serde(rename = "attractionK")]
    pub attraction_k: f32,
    #[serde(rename = "maxVelocity")]
    pub max_velocity: f32,
    #[serde(rename = "boundsSize")]
    pub bounds_size: f32,
    #[serde(rename = "separationRadius")]
    pub separation_radius: f32,
    #[serde(rename = "centerGravityK")]
    pub center_gravity_k: f32,
    #[serde(rename = "coolingRate")]
    pub cooling_rate: f32,
    #[serde(rename = "boundaryDamping")]
    pub boundary_damping: f32,
    #[serde(rename = "updateThreshold")]
    pub update_threshold: f32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestSystemSettings {
    #[serde(rename = "debugMode")]
    pub debug_mode: bool,
    #[serde(rename = "maxConnections")]
    pub max_connections: u32,
    #[serde(rename = "connectionTimeout")]
    pub connection_timeout: u32,
    #[serde(rename = "autoSave")]
    pub auto_save: bool,
    #[serde(rename = "logLevel")]
    pub log_level: String,
    pub websocket: TestWebSocketSettings,
    pub audit: TestAuditSettings,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestWebSocketSettings {
    #[serde(rename = "heartbeatInterval")]
    pub heartbeat_interval: u32,
    #[serde(rename = "reconnectDelay")]
    pub reconnect_delay: u32,
    #[serde(rename = "maxRetries")]
    pub max_retries: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestAuditSettings {
    #[serde(rename = "auditLogPath")]
    pub audit_log_path: String,
    #[serde(rename = "maxLogSize")]
    pub max_log_size: u64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TestXRSettings {
    #[serde(rename = "handMeshColor")]
    pub hand_mesh_color: String,
    #[serde(rename = "handRayColor")]
    pub hand_ray_color: String,
    #[serde(rename = "teleportRayColor")]
    pub teleport_ray_color: String,
    #[serde(rename = "controllerRayColor")]
    pub controller_ray_color: String,
    #[serde(rename = "planeColor")]
    pub plane_color: String,
    #[serde(rename = "portalEdgeColor")]
    pub portal_edge_color: String,
    #[serde(rename = "spaceType")]
    pub space_type: String,
    #[serde(rename = "locomotionMethod")]
    pub locomotion_method: String,
}

// Test factories
impl TestAppSettings {
    pub fn new() -> Self {
        Self {
            visualisation: TestVisualisationSettings {
                glow: TestGlowSettings {
                    node_glow_strength: 1.5,
                    edge_glow_strength: 2.0,
                    environment_glow_strength: 1.0,
                    base_color: "#00ffff".to_string(),
                    emission_color: "#ffffff".to_string(),
                    enabled: true,
                },
                graphs: TestGraphSettings {
                    logseq: TestLogseqGraphSettings {
                        physics: TestPhysicsSettings {
                            spring_k: 0.1,
                            repel_k: 100.0,
                            attraction_k: 0.02,
                            max_velocity: 5.0,
                            bounds_size: 1000.0,
                            separation_radius: 50.0,
                            center_gravity_k: 0.1,
                            cooling_rate: 0.95,
                            boundary_damping: 0.1,
                            update_threshold: 0.01,
                        },
                        node_radius: 10.0,
                        edge_thickness: 2.0,
                    },
                },
                color_schemes: vec!["default".to_string(), "dark".to_string()],
            },
            system: TestSystemSettings {
                debug_mode: false,
                max_connections: 100,
                connection_timeout: 5000,
                auto_save: true,
                log_level: "info".to_string(),
                websocket: TestWebSocketSettings {
                    heartbeat_interval: 30000,
                    reconnect_delay: 1000,
                    max_retries: 5,
                },
                audit: TestAuditSettings {
                    audit_log_path: "/var/log/audit.log".to_string(),
                    max_log_size: 10485760,
                },
            },
            xr: TestXRSettings {
                hand_mesh_color: "#ffffff".to_string(),
                hand_ray_color: "#0099ff".to_string(),
                teleport_ray_color: "#00ff00".to_string(),
                controller_ray_color: "#ff0000".to_string(),
                plane_color: "#333333".to_string(),
                portal_edge_color: "#ffff00".to_string(),
                space_type: "room-scale".to_string(),
                locomotion_method: "teleport".to_string(),
            },
        }
    }

    pub fn with_custom_values(mut self, overrides: Value) -> Self {
        if let Some(vis) = overrides.get("visualisation") {
            if let Some(glow) = vis.get("glow") {
                if let Some(strength) = glow.get("nodeGlowStrength") {
                    if let Some(val) = strength.as_f64() {
                        self.visualisation.glow.node_glow_strength = val as f32;
                    }
                }
                if let Some(color) = glow.get("baseColor") {
                    if let Some(val) = color.as_str() {
                        self.visualisation.glow.base_color = val.to_string();
                    }
                }
            }
        }

        if let Some(system) = overrides.get("system") {
            if let Some(debug) = system.get("debugMode") {
                if let Some(val) = debug.as_bool() {
                    self.system.debug_mode = val;
                }
            }
            if let Some(connections) = system.get("maxConnections") {
                if let Some(val) = connections.as_u64() {
                    self.system.max_connections = val as u32;
                }
            }
        }

        self
    }
}

// Validation error types for testing
#[derive(Debug, PartialEq)]
pub enum TestValidationError {
    TypeMismatch,
    OutOfRange,
    InvalidFormat,
    SecurityViolation,
    PathNotFound,
    InvalidValue(String),
}

impl std::fmt::Display for TestValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestValidationError::TypeMismatch => write!(f, "Type mismatch"),
            TestValidationError::OutOfRange => write!(f, "Value out of range"),
            TestValidationError::InvalidFormat => write!(f, "Invalid format"),
            TestValidationError::SecurityViolation => write!(f, "Security violation"),
            TestValidationError::PathNotFound => write!(f, "Path not found"),
            TestValidationError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
        }
    }
}

impl std::error::Error for TestValidationError {}

// Path validation utilities
pub fn validate_path_update(
    settings: &mut TestAppSettings,
    path: &str,
    value: &Value,
) -> Result<(), TestValidationError> {
    match path {
        p if p.contains("nodeGlowStrength") => {
            if let Some(num) = value.as_f64() {
                if num < 0.0 || num > 10.0 {
                    return Err(TestValidationError::OutOfRange);
                }
                settings.visualisation.glow.node_glow_strength = num as f32;
            } else {
                return Err(TestValidationError::TypeMismatch);
            }
        }
        p if p.contains("maxConnections") => {
            if let Some(num) = value.as_u64() {
                if num < 1 || num > 10000 {
                    return Err(TestValidationError::OutOfRange);
                }
                settings.system.max_connections = num as u32;
            } else {
                return Err(TestValidationError::TypeMismatch);
            }
        }
        p if p.contains("baseColor") => {
            if let Some(color_str) = value.as_str() {
                if !is_valid_hex_color(color_str) {
                    return Err(TestValidationError::InvalidFormat);
                }
                settings.visualisation.glow.base_color = color_str.to_string();
            } else {
                return Err(TestValidationError::TypeMismatch);
            }
        }
        p if p.contains("debugMode") => {
            if let Some(bool_val) = value.as_bool() {
                settings.system.debug_mode = bool_val;
            } else {
                return Err(TestValidationError::TypeMismatch);
            }
        }
        p if p.contains("springK") => {
            if let Some(num) = value.as_f64() {
                if num <= 0.0 {
                    return Err(TestValidationError::OutOfRange);
                }
                settings.visualisation.graphs.logseq.physics.spring_k = num as f32;
            } else {
                return Err(TestValidationError::TypeMismatch);
            }
        }
        p if p.contains("locomotionMethod") => {
            if let Some(method) = value.as_str() {
                if !["teleport", "smooth", "dash"].contains(&method) {
                    return Err(TestValidationError::InvalidValue(
                        "Invalid locomotion method".to_string(),
                    ));
                }
                settings.xr.locomotion_method = method.to_string();
            } else {
                return Err(TestValidationError::TypeMismatch);
            }
        }
        _ => {
            // Security check for unknown paths
            if let Some(s) = value.as_str() {
                if contains_dangerous_content(s) {
                    return Err(TestValidationError::SecurityViolation);
                }
            }
            return Err(TestValidationError::PathNotFound);
        }
    }

    Ok(())
}

// Utility functions
pub fn is_valid_hex_color(color: &str) -> bool {
    if !color.starts_with('#') || color.len() != 7 {
        return false;
    }
    color[1..].chars().all(|c| c.is_ascii_hexdigit())
}

pub fn contains_dangerous_content(content: &str) -> bool {
    let dangerous_patterns = [
        "<script",
        "DROP TABLE",
        "javascript:",
        "../",
        "..\\",
        "'; --",
        "\"; --",
        "<?php",
        "<%",
        "%3Cscript",
    ];

    let content_lower = content.to_lowercase();
    dangerous_patterns
        .iter()
        .any(|pattern| content_lower.contains(&pattern.to_lowercase()))
}

pub fn parse_dot_notation_path(path: &str) -> Vec<&str> {
    if path.is_empty() || path == "." {
        return vec![];
    }
    path.split('.').filter(|s| !s.is_empty()).collect()
}

// Performance measurement utilities
pub struct PerformanceTimer {
    start_time: Instant,
}

impl PerformanceTimer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.elapsed().as_millis()
    }

    pub fn elapsed_micros(&self) -> u128 {
        self.elapsed().as_micros()
    }
}

// Concurrent testing utilities
pub struct ConcurrentTestHarness<T> {
    data: Arc<RwLock<T>>,
    thread_count: usize,
}

impl<T: Send + Sync + 'static> ConcurrentTestHarness<T> {
    pub fn new(data: T, thread_count: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            thread_count,
        }
    }

    pub fn run_concurrent_readers<F, R>(&self, reader_fn: F) -> Vec<std::thread::JoinHandle<R>>
    where
        F: Fn(&T) -> R + Send + 'static + Clone,
        R: Send + 'static,
    {
        let mut handles = Vec::new();

        for i in 0..self.thread_count {
            let data_clone = self.data.clone();
            let reader_fn_clone = reader_fn.clone();

            let handle = std::thread::spawn(move || {
                let guard = data_clone.read().unwrap();
                reader_fn_clone(&*guard)
            });

            handles.push(handle);
        }

        handles
    }

    pub fn run_concurrent_writers<F, R>(&self, writer_fn: F) -> Vec<std::thread::JoinHandle<R>>
    where
        F: Fn(&mut T, usize) -> R + Send + 'static + Clone,
        R: Send + 'static,
    {
        let mut handles = Vec::new();

        for i in 0..self.thread_count {
            let data_clone = self.data.clone();
            let writer_fn_clone = writer_fn.clone();

            let handle = std::thread::spawn(move || {
                let mut guard = data_clone.write().unwrap();
                writer_fn_clone(&mut *guard, i)
            });

            handles.push(handle);
        }

        handles
    }
}

// Mock HTTP response utilities
pub struct MockHttpResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl MockHttpResponse {
    pub fn ok(body: Value) -> Self {
        Self {
            status_code: 200,
            headers: {
                let mut h = HashMap::new();
                h.insert("content-type".to_string(), "application/json".to_string());
                h
            },
            body: serde_json::to_string(&body).unwrap_or_default(),
        }
    }

    pub fn bad_request(error: &str) -> Self {
        Self {
            status_code: 400,
            headers: {
                let mut h = HashMap::new();
                h.insert("content-type".to_string(), "application/json".to_string());
                h
            },
            body: json!({
                "success": false,
                "error": error
            })
            .to_string(),
        }
    }

    pub fn internal_error(error: &str) -> Self {
        Self {
            status_code: 500,
            headers: {
                let mut h = HashMap::new();
                h.insert("content-type".to_string(), "application/json".to_string());
                h
            },
            body: json!({
                "success": false,
                "error": error
            })
            .to_string(),
        }
    }
}

// Async testing utilities
pub async fn run_concurrent_async_tasks<F, Fut, R>(task_count: usize, task_fn: F) -> Vec<R>
where
    F: Fn(usize) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = R> + Send,
    R: Send,
{
    use futures::future::join_all;

    let tasks = (0..task_count).map(|i| task_fn(i)).collect::<Vec<_>>();
    join_all(tasks).await
}

// Memory usage tracking for performance tests
pub struct MemoryTracker {
    initial_usage: Option<usize>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            initial_usage: Self::get_memory_usage(),
        }
    }

    pub fn get_memory_usage() -> Option<usize> {
        // In a real implementation, this would use system APIs
        // For testing purposes, we'll simulate memory usage
        None
    }

    pub fn memory_increase(&self) -> Option<usize> {
        if let (Some(initial), Some(current)) = (self.initial_usage, Self::get_memory_usage()) {
            Some(current.saturating_sub(initial))
        } else {
            None
        }
    }
}

// Test data generators
pub fn generate_large_settings_object(size: usize) -> TestAppSettings {
    let mut settings = TestAppSettings::new();

    // Generate large color schemes array
    settings.visualisation.color_schemes = (0..size).map(|i| format!("scheme_{}", i)).collect();

    settings
}

pub fn generate_complex_nested_value(depth: usize) -> Value {
    if depth == 0 {
        return json!("leaf_value");
    }

    json!({
        "level": depth,
        "data": format!("data_at_level_{}", depth),
        "nested": generate_complex_nested_value(depth - 1),
        "array": [1, 2, 3, depth]
    })
}

// Assertion helpers
pub fn assert_camel_case_keys(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            for (key, val) in map {
                assert!(
                    is_camel_case(key),
                    "Key '{}' at path '{}' is not camelCase",
                    key,
                    path
                );

                let new_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path, key)
                };
                assert_camel_case_keys(val, &new_path);
            }
        }
        Value::Array(arr) => {
            for (idx, item) in arr.iter().enumerate() {
                let new_path = format!("{}[{}]", path, idx);
                assert_camel_case_keys(item, &new_path);
            }
        }
        _ => {} // Primitive values don't need checking
    }
}

pub fn is_camel_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    // First character should be lowercase
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_lowercase() {
        return false;
    }

    // Rest should be alphanumeric (can include uppercase for camelCase)
    chars.all(|c| c.is_ascii_alphanumeric())
}

#[cfg(test)]
mod test_utils_tests {
    use super::*;

    #[test]
    fn test_camel_case_validation() {
        assert!(is_camel_case("camelCase"));
        assert!(is_camel_case("simple"));
        assert!(is_camel_case("nodeGlowStrength"));

        assert!(!is_camel_case("PascalCase"));
        assert!(!is_camel_case("snake_case"));
        assert!(!is_camel_case("kebab-case"));
        assert!(!is_camel_case(""));
        assert!(!is_camel_case("123invalid"));
    }

    #[test]
    fn test_hex_color_validation() {
        assert!(is_valid_hex_color("#ff0000"));
        assert!(is_valid_hex_color("#00FF00"));
        assert!(is_valid_hex_color("#123abc"));

        assert!(!is_valid_hex_color("ff0000"));
        assert!(!is_valid_hex_color("#gg0000"));
        assert!(!is_valid_hex_color("#ff00"));
        assert!(!is_valid_hex_color("#ff00000"));
        assert!(!is_valid_hex_color(""));
    }

    #[test]
    fn test_dangerous_content_detection() {
        assert!(contains_dangerous_content("<script>alert('xss')</script>"));
        assert!(contains_dangerous_content("'; DROP TABLE users; --"));
        assert!(contains_dangerous_content("../../../etc/passwd"));
        assert!(contains_dangerous_content("javascript:alert(1)"));

        assert!(!contains_dangerous_content("normal text"));
        assert!(!contains_dangerous_content("email@domain.com"));
        assert!(!contains_dangerous_content("file.txt"));
    }

    #[test]
    fn test_dot_notation_parsing() {
        assert_eq!(parse_dot_notation_path("simple"), vec!["simple"]);
        assert_eq!(
            parse_dot_notation_path("nested.field"),
            vec!["nested", "field"]
        );
        assert_eq!(
            parse_dot_notation_path("deep.nested.field.value"),
            vec!["deep", "nested", "field", "value"]
        );
        assert_eq!(parse_dot_notation_path(""), Vec::<&str>::new());
        assert_eq!(parse_dot_notation_path("."), Vec::<&str>::new());
    }
}

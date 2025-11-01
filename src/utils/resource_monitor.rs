use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/
/
/
/
/
/
/
/

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    
    pub max_file_descriptors: usize,
    
    pub fd_warning_threshold: f32,
    
    pub fd_error_threshold: f32,
    
    pub max_memory_mb: u64,
    
    pub max_tcp_connections: usize,
    
    pub max_zombie_age_secs: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_file_descriptors: 1000, 
            fd_warning_threshold: 0.7,  
            fd_error_threshold: 0.9,    
            max_memory_mb: 1024,        
            max_tcp_connections: 100,
            max_zombie_age_secs: 300, 
        }
    }
}

impl ResourceLimits {
    
    pub fn high_performance() -> Self {
        Self {
            max_file_descriptors: 2000,
            fd_warning_threshold: 0.8,
            fd_error_threshold: 0.95,
            max_memory_mb: 4096, 
            max_tcp_connections: 500,
            max_zombie_age_secs: 60, 
        }
    }

    
    pub fn constrained() -> Self {
        Self {
            max_file_descriptors: 500,
            fd_warning_threshold: 0.6,
            fd_error_threshold: 0.8,
            max_memory_mb: 512,
            max_tcp_connections: 50,
            max_zombie_age_secs: 600, 
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub timestamp: std::time::SystemTime,
    pub file_descriptors: usize,
    pub memory_usage_mb: u64,
    pub tcp_connections: usize,
    pub zombie_processes: usize,
    pub active_connections: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlert {
    pub alert_type: ResourceAlertType,
    pub message: String,
    pub current_value: u64,
    pub threshold: u64,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceAlertType {
    FileDescriptorWarning,
    FileDescriptorError,
    MemoryWarning,
    MemoryError,
    ConnectionWarning,
    ConnectionError,
    ZombieProcessWarning,
    ResourceExhaustion,
}

/
pub struct ResourceMonitor {
    limits: ResourceLimits,
    current_usage: Arc<RwLock<ResourceUsage>>,
    connection_registry: Arc<RwLock<HashMap<String, Vec<String>>>>,
    monitoring_active: Arc<std::sync::atomic::AtomicBool>,
    cleanup_callbacks: Arc<
        RwLock<
            Vec<
                Box<
                    dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
                        + Send
                        + Sync,
                >,
            >,
        >,
    >,
    alert_history: Arc<RwLock<Vec<ResourceAlert>>>,
}

impl ResourceMonitor {
    
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            limits,
            current_usage: Arc::new(RwLock::new(ResourceUsage {
                timestamp: std::time::SystemTime::now(),
                file_descriptors: 0,
                memory_usage_mb: 0,
                tcp_connections: 0,
                zombie_processes: 0,
                active_connections: HashMap::new(),
            })),
            connection_registry: Arc::new(RwLock::new(HashMap::new())),
            monitoring_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            cleanup_callbacks: Arc::new(RwLock::new(Vec::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    
    pub async fn start_monitoring(&self, interval: Duration) -> Result<(), String> {
        if self
            .monitoring_active
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err("Resource monitoring is already active".to_string());
        }

        info!("Starting resource monitoring with interval: {:?}", interval);
        self.monitoring_active
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let current_usage = self.current_usage.clone();
        let limits = self.limits.clone();
        let monitoring_active = self.monitoring_active.clone();
        let cleanup_callbacks = self.cleanup_callbacks.clone();
        let alert_history = self.alert_history.clone();

        tokio::spawn(async move {
            let mut monitoring_interval = tokio::time::interval(interval);

            while monitoring_active.load(std::sync::atomic::Ordering::Relaxed) {
                monitoring_interval.tick().await;

                
                let usage = match Self::collect_resource_usage().await {
                    Ok(usage) => usage,
                    Err(e) => {
                        error!("Failed to collect resource usage: {}", e);
                        continue;
                    }
                };

                
                let alerts = Self::check_resource_limits(&usage, &limits);

                
                {
                    let mut current_usage_guard = current_usage.write().await;
                    *current_usage_guard = usage.clone();
                }

                if !alerts.is_empty() {
                    let mut alert_history_guard = alert_history.write().await;
                    alert_history_guard.extend(alerts.clone());

                    
                    if alert_history_guard.len() > 100 {
                        let drain_count = alert_history_guard.len() - 100;
                        alert_history_guard.drain(..drain_count);
                    }
                }

                
                for alert in &alerts {
                    match alert.alert_type {
                        ResourceAlertType::FileDescriptorError
                        | ResourceAlertType::MemoryError
                        | ResourceAlertType::ResourceExhaustion => {
                            warn!("Critical resource alert: {}", alert.message);
                            Self::trigger_cleanup(&cleanup_callbacks).await;
                        }
                        _ => {
                            debug!("Resource alert: {}", alert.message);
                        }
                    }
                }

                
                if usage.file_descriptors > 0 {
                    debug!(
                        "Resource usage - FDs: {}/{}, Memory: {}MB, TCP: {}, Zombies: {}",
                        usage.file_descriptors,
                        limits.max_file_descriptors,
                        usage.memory_usage_mb,
                        usage.tcp_connections,
                        usage.zombie_processes
                    );
                }
            }

            info!("Resource monitoring stopped");
        });

        Ok(())
    }

    
    pub fn stop_monitoring(&self) {
        info!("Stopping resource monitoring");
        self.monitoring_active
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    
    pub async fn register_connection(&self, service: String, connection_id: String) {
        let mut registry = self.connection_registry.write().await;
        let service_clone = service.clone();
        registry
            .entry(service)
            .or_insert_with(Vec::new)
            .push(connection_id.clone());
        debug!(
            "Registered connection {} for service {}",
            connection_id, service_clone
        );
    }

    
    pub async fn unregister_connection(&self, service: &str, connection_id: &str) {
        let mut registry = self.connection_registry.write().await;
        if let Some(connections) = registry.get_mut(service) {
            connections.retain(|id| id != connection_id);
            if connections.is_empty() {
                registry.remove(service);
            }
        }
        debug!(
            "Unregistered connection {} for service {}",
            connection_id, service
        );
    }

    
    pub async fn add_cleanup_callback<F, Fut>(&self, callback: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut callbacks = self.cleanup_callbacks.write().await;
        callbacks.push(Box::new(move || Box::pin(callback())));
    }

    
    pub async fn get_current_usage(&self) -> ResourceUsage {
        self.current_usage.read().await.clone()
    }

    
    pub async fn get_recent_alerts(&self, limit: usize) -> Vec<ResourceAlert> {
        let alerts = self.alert_history.read().await;
        alerts.iter().rev().take(limit).cloned().collect()
    }

    
    pub async fn check_resource_availability(&self) -> Result<(), String> {
        let usage = self.current_usage.read().await;

        
        let fd_threshold =
            (self.limits.max_file_descriptors as f32 * self.limits.fd_error_threshold) as usize;
        if usage.file_descriptors >= fd_threshold {
            return Err(format!(
                "File descriptor limit exceeded: {} >= {} ({}% of max {})",
                usage.file_descriptors,
                fd_threshold,
                (self.limits.fd_error_threshold * 100.0) as u8,
                self.limits.max_file_descriptors
            ));
        }

        
        if usage.memory_usage_mb >= self.limits.max_memory_mb {
            return Err(format!(
                "Memory limit exceeded: {}MB >= {}MB",
                usage.memory_usage_mb, self.limits.max_memory_mb
            ));
        }

        
        if usage.tcp_connections >= self.limits.max_tcp_connections {
            return Err(format!(
                "Connection limit exceeded: {} >= {}",
                usage.tcp_connections, self.limits.max_tcp_connections
            ));
        }

        Ok(())
    }

    
    pub async fn force_cleanup(&self) {
        info!("Forcing resource cleanup");
        Self::trigger_cleanup(&self.cleanup_callbacks).await;

        
        Self::cleanup_zombie_processes().await;
        Self::trigger_garbage_collection().await;
    }

    

    
    async fn collect_resource_usage() -> Result<ResourceUsage, String> {
        let timestamp = std::time::SystemTime::now();

        
        let file_descriptors = Self::count_file_descriptors().await?;

        
        let memory_usage_mb = Self::get_memory_usage().await?;

        
        let tcp_connections = Self::count_tcp_connections().await?;

        
        let zombie_processes = Self::count_zombie_processes().await?;

        Ok(ResourceUsage {
            timestamp,
            file_descriptors,
            memory_usage_mb,
            tcp_connections,
            zombie_processes,
            active_connections: Self::get_active_connections().await.unwrap_or_default(),
        })
    }

    
    async fn count_file_descriptors() -> Result<usize, String> {
        #[cfg(target_os = "linux")]
        {
            match tokio::fs::read_dir("/proc/self/fd").await {
                Ok(mut entries) => {
                    let mut count: usize = 0;
                    while let Ok(Some(_)) = entries.next_entry().await {
                        count += 1;
                    }
                    Ok(count.saturating_sub(1)) 
                }
                Err(e) => Err(format!("Failed to read /proc/self/fd: {}", e)),
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            
            match tokio::process::Command::new("lsof")
                .args(["-p", &std::process::id().to_string()])
                .output()
                .await
            {
                Ok(output) => {
                    let count = String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .skip(1) 
                        .count();
                    Ok(count)
                }
                Err(_) => Ok(0), 
            }
        }
    }

    
    async fn get_memory_usage() -> Result<u64, String> {
        #[cfg(target_os = "linux")]
        {
            match tokio::fs::read_to_string("/proc/self/status").await {
                Ok(status) => {
                    for line in status.lines() {
                        if line.starts_with("VmRSS:") {
                            if let Some(kb_str) = line.split_whitespace().nth(1) {
                                if let Ok(kb) = kb_str.parse::<u64>() {
                                    return Ok(kb / 1024); 
                                }
                            }
                        }
                    }
                    Err("VmRSS not found in /proc/self/status".to_string())
                }
                Err(e) => Err(format!("Failed to read /proc/self/status: {}", e)),
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            
            Ok(100) 
        }
    }

    
    async fn count_tcp_connections() -> Result<usize, String> {
        #[cfg(target_os = "linux")]
        {
            match tokio::fs::read_to_string("/proc/net/tcp").await {
                Ok(tcp_data) => {
                    let count = tcp_data.lines().skip(1).count(); 
                    Ok(count)
                }
                Err(_) => Ok(0), 
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Ok(0) 
        }
    }

    
    async fn count_zombie_processes() -> Result<usize, String> {
        match tokio::process::Command::new("ps")
            .args(["aux"])
            .output()
            .await
        {
            Ok(output) => {
                let count = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .filter(|line| line.contains("<defunct>") || line.contains("Z"))
                    .count();
                Ok(count)
            }
            Err(_) => Ok(0), 
        }
    }

    
    fn check_resource_limits(usage: &ResourceUsage, limits: &ResourceLimits) -> Vec<ResourceAlert> {
        let mut alerts = Vec::new();
        let timestamp = std::time::SystemTime::now();

        
        let fd_warning_threshold =
            (limits.max_file_descriptors as f32 * limits.fd_warning_threshold) as usize;
        let fd_error_threshold =
            (limits.max_file_descriptors as f32 * limits.fd_error_threshold) as usize;

        if usage.file_descriptors >= fd_error_threshold {
            alerts.push(ResourceAlert {
                alert_type: ResourceAlertType::FileDescriptorError,
                message: format!(
                    "Critical: {} file descriptors open ({}% of limit {})",
                    usage.file_descriptors,
                    (usage.file_descriptors as f32 / limits.max_file_descriptors as f32 * 100.0)
                        as u8,
                    limits.max_file_descriptors
                ),
                current_value: usage.file_descriptors as u64,
                threshold: fd_error_threshold as u64,
                timestamp,
            });
        } else if usage.file_descriptors >= fd_warning_threshold {
            alerts.push(ResourceAlert {
                alert_type: ResourceAlertType::FileDescriptorWarning,
                message: format!(
                    "Warning: {} file descriptors open ({}% of limit {})",
                    usage.file_descriptors,
                    (usage.file_descriptors as f32 / limits.max_file_descriptors as f32 * 100.0)
                        as u8,
                    limits.max_file_descriptors
                ),
                current_value: usage.file_descriptors as u64,
                threshold: fd_warning_threshold as u64,
                timestamp,
            });
        }

        
        if usage.memory_usage_mb >= limits.max_memory_mb {
            alerts.push(ResourceAlert {
                alert_type: ResourceAlertType::MemoryError,
                message: format!(
                    "Memory limit exceeded: {}MB >= {}MB",
                    usage.memory_usage_mb, limits.max_memory_mb
                ),
                current_value: usage.memory_usage_mb,
                threshold: limits.max_memory_mb,
                timestamp,
            });
        } else if usage.memory_usage_mb >= limits.max_memory_mb * 8 / 10 {
            
            alerts.push(ResourceAlert {
                alert_type: ResourceAlertType::MemoryWarning,
                message: format!(
                    "High memory usage: {}MB ({}% of limit {}MB)",
                    usage.memory_usage_mb,
                    (usage.memory_usage_mb * 100 / limits.max_memory_mb),
                    limits.max_memory_mb
                ),
                current_value: usage.memory_usage_mb,
                threshold: limits.max_memory_mb * 8 / 10,
                timestamp,
            });
        }

        
        if usage.tcp_connections >= limits.max_tcp_connections {
            alerts.push(ResourceAlert {
                alert_type: ResourceAlertType::ConnectionError,
                message: format!(
                    "Connection limit exceeded: {} >= {}",
                    usage.tcp_connections, limits.max_tcp_connections
                ),
                current_value: usage.tcp_connections as u64,
                threshold: limits.max_tcp_connections as u64,
                timestamp,
            });
        }

        
        if usage.zombie_processes > 0 {
            alerts.push(ResourceAlert {
                alert_type: ResourceAlertType::ZombieProcessWarning,
                message: format!("Zombie processes detected: {}", usage.zombie_processes),
                current_value: usage.zombie_processes as u64,
                threshold: 0,
                timestamp,
            });
        }

        alerts
    }

    
    async fn trigger_cleanup(
        cleanup_callbacks: &Arc<
            RwLock<
                Vec<
                    Box<
                        dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
                            + Send
                            + Sync,
                    >,
                >,
            >,
        >,
    ) {
        let callbacks = cleanup_callbacks.read().await;
        for callback in callbacks.iter() {
            callback().await;
        }
    }

    
    async fn cleanup_zombie_processes() {
        match tokio::process::Command::new("pkill")
            .args(["-9", "defunct"])
            .output()
            .await
        {
            Ok(_) => debug!("Attempted zombie process cleanup"),
            Err(e) => debug!("Failed to clean zombie processes: {}", e),
        }
    }

    
    async fn trigger_garbage_collection() {
        
        
        debug!("Triggering garbage collection");

        
        
        
        
    }

    
    async fn get_active_connections() -> Result<HashMap<String, usize>, String> {
        let mut connections = HashMap::new();

        #[cfg(target_os = "linux")]
        {
            
            if let Ok(tcp_data) = tokio::fs::read_to_string("/proc/net/tcp").await {
                let tcp_count = tcp_data.lines().skip(1).count();
                connections.insert("tcp".to_string(), tcp_count);
            }

            
            if let Ok(tcp6_data) = tokio::fs::read_to_string("/proc/net/tcp6").await {
                let tcp6_count = tcp6_data.lines().skip(1).count();
                connections.insert("tcp6".to_string(), tcp6_count);
            }

            
            if let Ok(udp_data) = tokio::fs::read_to_string("/proc/net/udp").await {
                let udp_count = udp_data.lines().skip(1).count();
                connections.insert("udp".to_string(), udp_count);
            }

            
            if let Ok(unix_data) = tokio::fs::read_to_string("/proc/net/unix").await {
                let unix_count = unix_data.lines().skip(1).count();
                connections.insert("unix".to_string(), unix_count);
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            
            connections.insert("total".to_string(), 0);
        }

        Ok(connections)
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        self.stop_monitoring();
        info!("ResourceMonitor dropped - monitoring stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_resource_monitor_creation() {
        let limits = ResourceLimits::default();
        let monitor = ResourceMonitor::new(limits);

        let usage = monitor.get_current_usage().await;
        assert_eq!(usage.file_descriptors, 0); 
    }

    #[tokio::test]
    async fn test_connection_registration() {
        let monitor = ResourceMonitor::new(ResourceLimits::default());

        monitor
            .register_connection("test-service".to_string(), "conn-1".to_string())
            .await;
        monitor
            .register_connection("test-service".to_string(), "conn-2".to_string())
            .await;

        let registry = monitor.connection_registry.read().await;
        let connections = registry.get("test-service").unwrap();
        assert_eq!(connections.len(), 2);

        drop(registry);

        monitor
            .unregister_connection("test-service", "conn-1")
            .await;

        let registry = monitor.connection_registry.read().await;
        let connections = registry.get("test-service").unwrap();
        assert_eq!(connections.len(), 1);
    }

    #[tokio::test]
    async fn test_resource_limits() {
        let usage = ResourceUsage {
            timestamp: std::time::SystemTime::now(),
            file_descriptors: 950,
            memory_usage_mb: 1500,
            tcp_connections: 150,
            zombie_processes: 2,
            active_connections: HashMap::new(),
        };

        let limits = ResourceLimits {
            max_file_descriptors: 1000,
            fd_warning_threshold: 0.7,
            fd_error_threshold: 0.9,
            max_memory_mb: 1024,
            max_tcp_connections: 100,
            max_zombie_age_secs: 300,
        };

        let alerts = ResourceMonitor::check_resource_limits(&usage, &limits);

        
        assert!(!alerts.is_empty());

        
        assert!(alerts
            .iter()
            .any(|a| matches!(a.alert_type, ResourceAlertType::FileDescriptorError)));

        
        assert!(alerts
            .iter()
            .any(|a| matches!(a.alert_type, ResourceAlertType::MemoryError)));

        
        assert!(alerts
            .iter()
            .any(|a| matches!(a.alert_type, ResourceAlertType::ConnectionError)));
    }
}

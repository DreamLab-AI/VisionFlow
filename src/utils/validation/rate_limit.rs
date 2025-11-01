use actix_web::{dev::ServiceRequest, HttpRequest, HttpResponse, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub burst_size: u32,
    pub cleanup_interval: Duration,
    pub ban_duration: Duration,
    pub max_violations: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,                    
            burst_size: 10,                             
            cleanup_interval: Duration::from_secs(300), 
            ban_duration: Duration::from_secs(3600),    
            max_violations: 5,                          
        }
    }
}

///
#[derive(Debug, Clone)]
struct RateLimitEntry {
    tokens: u32,
    last_refill: Instant,
    violation_count: u32,
    banned_until: Option<Instant>,
}

impl RateLimitEntry {
    fn new(config: &RateLimitConfig) -> Self {
        Self {
            tokens: config.burst_size,
            last_refill: Instant::now(),
            violation_count: 0,
            banned_until: None,
        }
    }

    fn is_banned(&self) -> bool {
        if let Some(banned_until) = self.banned_until {
            Instant::now() < banned_until
        } else {
            false
        }
    }

    fn refill_tokens(&mut self, config: &RateLimitConfig) {
        let now = Instant::now();
        let time_passed = now.duration_since(self.last_refill);
        let tokens_to_add = (time_passed.as_secs() * config.requests_per_minute as u64 / 60) as u32;

        if tokens_to_add > 0 {
            self.tokens = (self.tokens + tokens_to_add).min(config.burst_size);
            self.last_refill = now;
        }
    }

    fn try_consume_token(&mut self, config: &RateLimitConfig) -> bool {
        if self.is_banned() {
            return false;
        }

        self.refill_tokens(config);

        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            self.violation_count += 1;

            if self.violation_count >= config.max_violations {
                self.banned_until = Some(Instant::now() + config.ban_duration);
                warn!("Client banned due to rate limit violations");
            }

            false
        }
    }
}

///
pub struct RateLimiter {
    clients: Arc<RwLock<HashMap<String, RateLimitEntry>>>,
    config: RateLimitConfig,
    last_cleanup: Arc<RwLock<Instant>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        let limiter = Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            config,
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        };

        
        limiter.start_cleanup_task();
        limiter
    }

    
    pub fn is_allowed(&self, client_id: &str) -> bool {
        self.cleanup_if_needed();

        let mut clients = self.clients.write().unwrap();
        let entry = clients
            .entry(client_id.to_string())
            .or_insert_with(|| RateLimitEntry::new(&self.config));

        let allowed = entry.try_consume_token(&self.config);

        if !allowed {
            warn!("Rate limit exceeded for client: {}", client_id);
        }

        allowed
    }

    
    pub fn remaining_tokens(&self, client_id: &str) -> u32 {
        let mut clients = self.clients.write().unwrap();
        let entry = clients
            .entry(client_id.to_string())
            .or_insert_with(|| RateLimitEntry::new(&self.config));

        entry.refill_tokens(&self.config);
        entry.tokens
    }

    
    pub fn reset_time(&self, client_id: &str) -> Duration {
        let clients = self.clients.read().unwrap();
        if let Some(entry) = clients.get(client_id) {
            let time_since_refill = Instant::now().duration_since(entry.last_refill);
            let time_to_next_token =
                Duration::from_secs(60 / self.config.requests_per_minute as u64);
            time_to_next_token.saturating_sub(time_since_refill)
        } else {
            Duration::from_secs(0)
        }
    }

    
    pub fn is_banned(&self, client_id: &str) -> bool {
        let clients = self.clients.read().unwrap();
        clients
            .get(client_id)
            .map(|entry| entry.is_banned())
            .unwrap_or(false)
    }

    
    fn cleanup_if_needed(&self) {
        let mut last_cleanup = self.last_cleanup.write().unwrap();
        let now = Instant::now();

        if now.duration_since(*last_cleanup) < self.config.cleanup_interval {
            return;
        }

        *last_cleanup = now;
        drop(last_cleanup);

        let mut clients = self.clients.write().unwrap();
        let before_count = clients.len();

        clients.retain(|_, entry| {
            
            !entry.is_banned()
                || entry
                    .banned_until
                    .map(|until| now < until + Duration::from_secs(3600))
                    .unwrap_or(true)
        });

        let after_count = clients.len();
        if before_count != after_count {
            debug!(
                "Rate limiter cleanup: removed {} expired entries",
                before_count - after_count
            );
        }
    }

    
    fn start_cleanup_task(&self) {
        let clients = self.clients.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.cleanup_interval);

            loop {
                interval.tick().await;

                let mut clients_guard = clients.write().unwrap();
                let before_count = clients_guard.len();
                let now = Instant::now();

                clients_guard.retain(|_, entry| {
                    
                    let keep_banned = entry
                        .banned_until
                        .map(|until| now < until + Duration::from_secs(3600))
                        .unwrap_or(false);

                    let keep_active =
                        now.duration_since(entry.last_refill) < Duration::from_secs(1800); 

                    keep_banned || keep_active
                });

                let after_count = clients_guard.len();
                if before_count != after_count {
                    info!(
                        "Rate limiter background cleanup: removed {} expired entries",
                        before_count - after_count
                    );
                }
            }
        });
    }

    
    pub fn get_stats(&self) -> RateLimitStats {
        let clients = self.clients.read().unwrap();
        let now = Instant::now();

        let total_clients = clients.len();
        let banned_clients = clients.values().filter(|entry| entry.is_banned()).count();
        let active_clients = clients
            .values()
            .filter(|entry| now.duration_since(entry.last_refill) < Duration::from_secs(300))
            .count();

        RateLimitStats {
            total_clients,
            banned_clients,
            active_clients,
            config: self.config.clone(),
        }
    }
}

///
#[derive(Debug, Serialize)]
pub struct RateLimitStats {
    pub total_clients: usize,
    pub banned_clients: usize,
    pub active_clients: usize,
    pub config: RateLimitConfig,
}

///
pub fn extract_client_id(req: &HttpRequest) -> String {
    
    let real_ip = req
        .headers()
        .get("X-Real-IP")
        .or_else(|| req.headers().get("X-Forwarded-For"))
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.trim().parse::<IpAddr>().ok());

    
    let ip = real_ip.or_else(|| req.peer_addr().map(|addr| addr.ip()));

    match ip {
        Some(addr) => addr.to_string(),
        None => "unknown".to_string(),
    }
}

///
pub fn extract_client_id_from_service_request(req: &ServiceRequest) -> String {
    let real_ip = req
        .headers()
        .get("X-Real-IP")
        .or_else(|| req.headers().get("X-Forwarded-For"))
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.trim().parse::<IpAddr>().ok());

    let ip = real_ip.or_else(|| req.peer_addr().map(|addr| addr.ip()));

    match ip {
        Some(addr) => addr.to_string(),
        None => "unknown".to_string(),
    }
}

///
pub fn create_rate_limit_response(client_id: &str, limiter: &RateLimiter) -> Result<HttpResponse> {
    let remaining = limiter.remaining_tokens(client_id);
    let reset_time = limiter.reset_time(client_id);
    let retry_after = reset_time.as_secs();

    let response =
        if limiter.is_banned(client_id) {
            HttpResponse::TooManyRequests()
            .insert_header(("X-RateLimit-Limit", limiter.config.requests_per_minute.to_string()))
            .insert_header(("X-RateLimit-Remaining", "0"))
            .insert_header(("X-RateLimit-Reset", reset_time.as_secs().to_string()))
            .insert_header(("Retry-After", retry_after.to_string()))
            .json(serde_json::json!({
                "error": "rate_limit_exceeded",
                "message": "Client is temporarily banned due to excessive rate limit violations",
                "retry_after": retry_after
            }))
        } else {
            HttpResponse::TooManyRequests()
                .insert_header((
                    "X-RateLimit-Limit",
                    limiter.config.requests_per_minute.to_string(),
                ))
                .insert_header(("X-RateLimit-Remaining", remaining.to_string()))
                .insert_header(("X-RateLimit-Reset", reset_time.as_secs().to_string()))
                .insert_header(("Retry-After", retry_after.to_string()))
                .json(serde_json::json!({
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests, please slow down",
                    "retry_after": retry_after
                }))
        };

    Ok(response)
}

///
pub struct EndpointRateLimits;

impl EndpointRateLimits {
    
    pub fn settings_update() -> RateLimitConfig {
        RateLimitConfig {
            requests_per_minute: 30,
            burst_size: 5,
            ..Default::default()
        }
    }

    
    pub fn ragflow_chat() -> RateLimitConfig {
        RateLimitConfig {
            requests_per_minute: 20,
            burst_size: 3,
            ..Default::default()
        }
    }

    
    pub fn bots_operations() -> RateLimitConfig {
        RateLimitConfig {
            requests_per_minute: 40,
            burst_size: 8,
            ..Default::default()
        }
    }

    
    pub fn health_check() -> RateLimitConfig {
        RateLimitConfig {
            requests_per_minute: 120,
            burst_size: 20,
            ..Default::default()
        }
    }

    
    pub fn socket_flow_updates() -> RateLimitConfig {
        RateLimitConfig {
            requests_per_minute: 300,                   
            burst_size: 50,                             
            cleanup_interval: Duration::from_secs(600), 
            ban_duration: Duration::from_secs(600),     
            max_violations: 10,                         
        }
    }

    
    pub fn default() -> RateLimitConfig {
        RateLimitConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_rate_limiter_basic() {
        let config = RateLimitConfig {
            requests_per_minute: 60,
            burst_size: 5,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);
        let client_id = "test_client";

        
        for _ in 0..5 {
            assert!(limiter.is_allowed(client_id));
        }

        
        assert!(!limiter.is_allowed(client_id));
    }

    #[test]
    fn test_rate_limiter_refill() {
        let config = RateLimitConfig {
            requests_per_minute: 60,
            burst_size: 1,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);
        let client_id = "test_client_refill";

        
        assert!(limiter.is_allowed(client_id));
        assert!(!limiter.is_allowed(client_id));

        
        thread::sleep(Duration::from_secs(2));

        
        assert!(limiter.is_allowed(client_id));
    }

    #[test]
    fn test_ban_after_violations() {
        let config = RateLimitConfig {
            requests_per_minute: 60,
            burst_size: 1,
            max_violations: 2,
            ban_duration: Duration::from_secs(1),
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);
        let client_id = "test_client_ban";

        
        assert!(limiter.is_allowed(client_id)); 
        assert!(!limiter.is_allowed(client_id)); 
        assert!(!limiter.is_allowed(client_id)); 

        
        assert!(limiter.is_banned(client_id));
        assert!(!limiter.is_allowed(client_id));
    }
}

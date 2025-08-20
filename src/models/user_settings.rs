use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use log::{info, error, debug, warn};
use once_cell::sync::Lazy;

use crate::models::UISettings;

// Global cache for user settings
static USER_SETTINGS_CACHE: Lazy<Arc<RwLock<HashMap<String, CachedUserSettings>>>> = 
    Lazy::new(|| Arc::new(RwLock::new(HashMap::new())));

// Cache expiration time (10 minutes)
const CACHE_EXPIRATION: Duration = Duration::from_secs(10 * 60);

// Cache entry with timestamp
struct CachedUserSettings {
    settings: UserSettings,
    timestamp: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    pub pubkey: String,
    pub settings: UISettings,
    pub last_modified: i64,
}

impl UserSettings {
    pub fn new(pubkey: &str, settings: UISettings) -> Self {
        Self {
            pubkey: pubkey.to_string(),
            settings,
            last_modified: chrono::Utc::now().timestamp(),
        }
    }

    pub fn load(pubkey: &str) -> Option<Self> {
        // First check the cache
        {
            let cache = match USER_SETTINGS_CACHE.read() {
                Ok(cache) => cache,
                Err(e) => {
                    error!("Failed to read user settings cache: {}", e);
                    return None;
                }
            };
            if let Some(cached) = cache.get(pubkey) {
                // Check if cache is still valid
                if cached.timestamp.elapsed() < CACHE_EXPIRATION {
                    debug!("Using cached settings for user {}", pubkey);
                    return Some(cached.settings.clone());
                }
                // Cache expired, will reload from disk
                debug!("Cache expired for user {}, reloading from disk", pubkey);
            }
        }
        
        // Not in cache or expired, load from disk
        let path = Self::get_settings_path(pubkey);
        match fs::read_to_string(&path) {
            Ok(content) => {
                match serde_yaml::from_str::<UserSettings>(&content) {
                    Ok(settings) => {
                        // Add to cache
                        let settings_clone = settings.clone();
                        {
                            let mut cache = match USER_SETTINGS_CACHE.write() {
                                Ok(cache) => cache,
                                Err(e) => {
                                    error!("Failed to write to user settings cache: {}", e);
                                    // Continue without caching
                                    return Some(settings);
                                }
                            };
                            cache.insert(pubkey.to_string(), CachedUserSettings {
                                settings: settings_clone,
                                timestamp: Instant::now(),
                            });
                        }
                        info!("Loaded settings for user {} and added to cache", pubkey);
                        Some(settings)
                    }
                    Err(e) => {
                        error!("Failed to parse settings for user {}: {}", pubkey, e);
                        None
                    }
                }
            }
            Err(e) => {
                debug!("No settings file found for user {}: {}", pubkey, e);
                None
            },
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::get_settings_path(&self.pubkey);
        
        // Update cache first (this is fast and ensures immediate availability)
        {
            let mut cache = match USER_SETTINGS_CACHE.write() {
                Ok(cache) => cache,
                Err(e) => {
                    warn!("Failed to write to user settings cache during save: {}", e);
                    // Continue with save operation even if caching fails
                    return self.save_to_disk();
                }
            };
            cache.insert(self.pubkey.clone(), CachedUserSettings {
                settings: self.clone(),
                timestamp: Instant::now(),
            });
            debug!("Updated cache for user {}", self.pubkey);
        }
        
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                warn!("Failed to create settings directory: {}", e);
                return Err(format!("Failed to create settings directory: {}", e));
            }
        }

        // Save settings to disk asynchronously to avoid blocking
        // For now we'll use a simple thread, but this could be improved with a proper async task
        let pubkey = self.pubkey.clone();
        let settings_clone = self.clone();
        
        std::thread::spawn(move || {
            debug!("Background thread saving settings for user {}", pubkey);
            match serde_yaml::to_string(&settings_clone) {
                Ok(yaml) => {
                    match fs::write(&path, yaml) {
                        Ok(_) => info!("Saved settings for user {} to disk", pubkey),
                        Err(e) => error!("Failed to write settings file for {}: {}", pubkey, e)
                    }
                }
                Err(e) => error!("Failed to serialize settings for {}: {}", pubkey, e),
            }
        });
        
        // Return success immediately since we've updated the cache
        Ok(())
    }
    
    fn save_to_disk(&self) -> Result<(), String> {
        let path = Self::get_settings_path(&self.pubkey);
        
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return Err(format!("Failed to create settings directory: {}", e));
            }
        }
        
        // Serialize and save settings
        match serde_yaml::to_string(self) {
            Ok(content) => {
                match std::fs::write(&path, content) {
                    Ok(_) => {
                        debug!("Saved settings to disk for user {}", self.pubkey);
                        Ok(())
                    }
                    Err(e) => Err(format!("Failed to write settings file: {}", e))
                }
            }
            Err(e) => Err(format!("Failed to serialize settings: {}", e))
        }
    }

    fn get_settings_path(pubkey: &str) -> PathBuf {
        PathBuf::from("/app/user_settings").join(format!("{}.yaml", pubkey))
    }
    
    // Clear the cache entry for a specific user
    pub fn clear_cache(pubkey: &str) {
        let mut cache = match USER_SETTINGS_CACHE.write() {
            Ok(cache) => cache,
            Err(e) => {
                error!("Failed to write to cache for clearing user {}: {}", pubkey, e);
                return;
            }
        };
        if cache.remove(pubkey).is_some() {
            debug!("Cleared cache for user {}", pubkey);
        }
    }
    
    // Clear all cached settings
    pub fn clear_all_cache() {
        let mut cache = match USER_SETTINGS_CACHE.write() {
            Ok(cache) => cache,
            Err(e) => {
                error!("Failed to write to cache for clearing all settings: {}", e);
                return;
            }
        };
        let count = cache.len();
        cache.clear();
        debug!("Cleared all cached settings ({} entries)", count);
    }
}
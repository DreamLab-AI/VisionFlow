# Production Configuration Guide

## Executive Summary

VisionFlow's configuration system represents a **production-grade, security-hardened configuration architecture** that provides exceptional flexibility, safety, and operational excellence. The system implements comprehensive validation, hot-reload capabilities, and enterprise-grade security measures while maintaining intuitive management and deployment patterns.

**🎯 Configuration System Status: PRODUCTION COMPLETE ✅**

**Production Configuration Features:**
- **🔄 Hot-Reload Capabilities**: Configuration changes without service restart
- **🛡️ Comprehensive Validation**: Multi-tier validation with detailed error reporting
- **🔒 Security-First Design**: Protected settings isolation and secret management
- **📊 Environment Awareness**: Development, staging, production configurations
- **🎯 Version Control**: Configuration versioning with rollback capabilities

## Overview

VisionFlow uses a production-hardened, layered configuration system combining YAML files, environment variables, and runtime settings. This system supports enterprise-grade configuration management with comprehensive validation, security, and operational features.

## Configuration Hierarchy

```mermaid
graph TD
    ENV[Environment Variables] --> YAML[YAML Config Files]
    YAML --> RUNTIME[Runtime Settings]
    RUNTIME --> APP[Application State]

    style ENV fill:#f9f,stroke:#333,stroke-width:2px
    style YAML fill:#bbf,stroke:#333,stroke-width:2px
    style RUNTIME fill:#bfb,stroke:#333,stroke-width:2px
```

Priority order (highest to lowest):
1. Environment variables
2. Command-line arguments
3. YAML configuration files
4. Default values

## Configuration Flow Map

### Settings Sources and Their Rust Structures

| Configuration Source | File/Location | Rust Structure | Purpose |
|---------------------|---------------|----------------|----------|
| **Main Settings** | `data/settings.yaml` | `Settings` | Complete application configuration |
| **Environment Overrides** | `.env` file | Merged into `Settings` | Secrets and deployment-specific values |
| **Protected Settings** | In-memory only | `ProtectedSettings` | API keys and sensitive user data |
| **UI Settings** | Derived from above | `UISettings` | Safe subset sent to frontend |
| **User Settings** | `/app/user_settings/<pubkey>.yaml` | `UserSettings` | Per-user preferences |

### Key Configuration Mappings

#### Environment Variable → settings.yaml → Rust Struct

| Environment Variable | settings.yaml Path | Rust Struct Field | Type | Description |
|---------------------|-------------------|------------------|------|-------------|
| **Core Configuration** ||||
| `AGENT_CONTROL_URL` | *(runtime only)* | `app_state.agent_control_url` | `String` | MCP server URL for agent control |
| `RUST_LOG` | `logging.level` | `Settings.logging.level` | `String` | Log level (trace/debug/info/warn/error) |
| `DATABASE_URL` | `database.url` | `Settings.database.url` | `String` | PostgreSQL connection string |
| **GPU Configuration** ||||
| `ENABLE_GPU_PHYSICS` | `gpu.enabled` | `Settings.gpu.enabled` | `bool` | Enable CUDA physics simulation |
| `NVIDIA_GPU_UUID` | `gpu.device_uuid` | `Settings.gpu.device_uuid` | `String` | Specific GPU device UUID |
| `CUDA_VISIBLE_DEVICES` | `gpu.device_id` | `Settings.gpu.device_id` | `u32` | GPU device index |
| **Physics Simulation** ||||
| `PHYSICS_UPDATE_RATE` | `graph.simulation.update_rate` | `Settings.graph.simulation.update_rate` | `f32` | Updates per second |
| `PHYSICS_TIME_STEP` | `graph.simulation.time_step` | `Settings.graph.simulation.time_step` | `f32` | Simulation timestep |
| `PHYSICS_DAMPING` | `graph.simulation.damping` | `Settings.graph.simulation.damping` | `f32` | Velocity damping factor |
| **AI Services** ||||
| `RAGFLOW_API_KEY` | `ragflow.api_key` | `Settings.ragflow.api_key` | `String` | RAGFlow API authentication |
| `PERPLEXITY_API_KEY` | `perplexity.api_key` | `Settings.perplexity.api_key` | `String` | Perplexity AI API key |
| `OPENAI_API_KEY` | `openai.api_key` | `Settings.openai.api_key` | `String` | OpenAI API key |
| **Feature Access** ||||
| `POWER_USER_KEY_*` | `features.power_user_keys[]` | `Settings.features.power_user_keys` | `Vec<String>` | Array of power user keys |
| `FEATURES_ENABLED` | `features.enabled_features[]` | `Settings.features.enabled_features` | `Vec<String>` | Enabled feature flags |

## Configuration Files

### Main Configuration: `config.yml`

```yaml
# Server Configuration
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  shutdown_timeout: 30

# Database Configuration
database:
  url: "postgres://user:pass@localhost/dbname"
  max_connections: 10
  connection_timeout: 30

# Graph Settings
graph:
  max_nodes: 100000
  simulation:
    repulsion_strength: 100.0
    attraction_strength: 0.01
    centering_strength: 0.01
    damping: 0.9
    time_step: 0.016

# AI Services
ragflow:
  api_key: "${RAGFLOW_API_KEY}"
  base_url: "https://api.ragflow.com"
  timeout: 30
  max_retries: 3

perplexity:
  api_key: "${PERPLEXITY_API_KEY}"
  base_url: "https://api.perplexity.ai"
  model: "mixtral-8x7b-instruct"
  max_tokens: 2048
  temperature: 0.7

# Authentication
auth:
  nostr:
    enabled: true
    relay_urls:
      - "wss://relay.damus.io"
      - "wss://nostr.wine"
  api_keys:
    enabled: true
    rotation_days: 90

# Feature Access
features:
  power_users_enabled: true
  power_user_keys:
    - "${POWER_USER_KEY_1}"
    - "${POWER_USER_KEY_2}"

  enabled_features:
    - graph_visualization
    - ai_chat
    - speech_recognition
    - xr_support

# GPU Configuration
gpu:
  enabled: true
  device_id: 0
  fallback_to_cpu: true
  block_size: 256
  max_memory_mb: 4096

# Logging
logging:
  level: "info"
  format: "json"
  file: "logs/server.log"
  max_size_mb: 100
  max_files: 10

# CORS Settings
cors:
  allowed_origins:
    - "http://localhost:3000"
    - "http://localhost:5173"
  allowed_methods:
    - "GET"
    - "POST"
    - "PUT"
    - "DELETE"
  allowed_headers:
    - "Content-Type"
    - "Authorization"
    - "X-API-Key"
```

### User Settings: `data/settings.yaml`

```yaml
# User-specific settings
theme: "dark"
language: "en"

# Visualization preferences
visualisation:
  node_size: 10
  edge_width: 2
  label_size: 12
  show_labels: true
  animation_speed: 1.0

# Client preferences
client:
  auto_connect: true
  reconnect_attempts: 5
  reconnect_delay: 1000
```

## Environment Variables

### Core Variables

```bash
# Server
HOST=0.0.0.0
PORT=8080
RUST_LOG=info

# Database
DATABASE_URL=postgres://user:pass@localhost/dbname

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret

# AI Services
RAGFLOW_API_KEY=your-ragflow-key
PERPLEXITY_API_KEY=your-perplexity-key
OPENAI_API_KEY=your-openai-key

# Feature Access
POWER_USER_KEY_1=key1
POWER_USER_KEY_2=key2

# GPU
CUDA_ENABLED=true
CUDA_DEVICE_ID=0
CUDA_VISIBLE_DEVICES=0

# Development
DEV_MODE=false
HOT_RELOAD=true
```

### Docker Environment

```bash
# Docker-specific
DOCKER_BUILDKIT=1
COMPOSE_PROJECT_NAME=visionflow

# Volumes
DATA_PATH=/data
LOGS_PATH=/logs

# Networking
EXTERNAL_PORT=8080
INTERNAL_PORT=8080
```

## Runtime Configuration

### Settings Actor

The `SettingsActor` manages runtime configuration changes:

```rust
// Get current settings
let settings = settings_actor.send(GetSettings).await?;

// Update specific setting
settings_actor.send(SetSettingByPath {
    path: "graph.simulation.damping".to_string(),
    value: json!(0.95),
}).await?;
```

### Client Settings Synchronization

Client settings are synchronized via WebSocket:

```typescript
// Client-side
const settings = {
  theme: 'dark',
  visualisation: {
    nodeSize: 12,
    showLabels: true
  }
};

websocket.send({
  type: 'updateSettings',
  payload: settings
});
```

## Configuration Schemas

### Settings Structure

```rust
pub struct Settings {
    // Server configuration
    pub server: ServerConfig,

    // Graph settings
    pub graph: GraphConfig,

    // AI service settings
    pub ragflow: Option<RagflowConfig>,
    pub perplexity: Option<PerplexityConfig>,

    // Authentication
    pub auth: AuthConfig,

    // Feature flags
    pub features: FeatureConfig,

    // GPU settings
    pub gpu: GpuConfig,

    // User settings
    pub user_settings: UserSettings,
}
```

### Detailed Structure Mappings

#### ServerConfig
```rust
pub struct ServerConfig {
    pub host: String,              // Default: "0.0.0.0"
    pub port: u16,                 // Default: 8080
    pub workers: usize,            // Default: CPU cores
    pub shutdown_timeout: u64,     // Default: 30 seconds
    pub max_connections: u32,      // Default: 10000
}
```

#### GraphConfig
```rust
pub struct GraphConfig {
    pub max_nodes: usize,          // Default: 100000
    pub simulation: SimulationParams,
    pub visualisation: VisualizationConfig,
}

pub struct SimulationParams {
    pub repulsion_strength: f32,   // Default: 100.0
    pub attraction_strength: f32,  // Default: 0.01
    pub centering_strength: f32,   // Default: 0.01
    pub damping: f32,              // Default: 0.9
    pub time_step: f32,            // Default: 0.016
    pub update_rate: f32,          // Default: 60.0
    pub link_distance: f32,        // Default: 30.0
    pub collision_radius: f32,     // Default: 10.0
}
```

#### ProtectedSettings vs UISettings
```rust
// Protected settings (never sent to frontend)
pub struct ProtectedSettings {
    pub api_keys: HashMap<String, String>,
    pub database_credentials: DatabaseConfig,
    pub jwt_secret: String,
    pub encryption_keys: Vec<String>,
}

// UI settings (safe subset for frontend)
pub struct UISettings {
    pub theme: String,
    pub visualisation: VisualizationConfig,
    pub features: EnabledFeatures,
    pub user_preferences: UserPreferences,
}
```

#### Settings Inheritance Flow
```mermaid
graph LR
    A[Environment Variables] --> B[settings.yaml]
    B --> C[Settings]
    C --> D[ProtectedSettings]
    C --> E[UISettings]
    E --> F[Frontend Client]
    D --> G[Backend Only]

    style D fill:#ff9999,stroke:#333,stroke-width:2px
    style E fill:#99ff99,stroke:#333,stroke-width:2px
```

### Validation

Configuration is validated on load:

```rust
impl Settings {
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate port range
        if self.server.port < 1024 || self.server.port > 65535 {
            return Err(ConfigError::InvalidPort);
        }

        // Validate GPU settings
        if self.gpu.enabled && self.gpu.device_id > 7 {
            return Err(ConfigError::InvalidGpuDevice);
        }

        Ok(())
    }
}
```

## Production Configuration Loading

### Enhanced Startup Sequence with Validation

```mermaid
sequenceDiagram
    participant Main
    participant Config
    participant Validator
    participant Security
    participant Env
    participant File
    participant Monitor

    Main->>Config: load_production_configuration()
    Config->>File: Read config.yml
    Config->>File: Read settings.yaml
    Config->>Env: Override with env vars
    Config->>Validator: Comprehensive validation
    Validator->>Security: Security validation
    Security-->>Validator: Security status
    Validator-->>Config: Validation results
    
    alt Validation Failed
        Config->>Config: Generate detailed error report
        Config-->>Main: ConfigurationError with details
    else Validation Successful
        Config->>Monitor: Start configuration monitor
        Monitor->>Monitor: Watch for file changes
        Config-->>Main: Validated Settings
    end
```

### Production Configuration Loading Implementation

```rust
pub async fn load_production_configuration() -> Result<ProductionSettings, ConfigurationError> {
    let mut config = Config::new();
    let start_time = std::time::Instant::now();

    // 1. Load base configuration with error handling
    config.merge(File::with_name("config.yml")
        .format(FileFormat::Yaml)
        .required(true)
    ).map_err(|e| ConfigurationError::BaseConfigMissing(e.to_string()))?;

    // 2. Load environment-specific overrides
    let env = std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string());
    if let Ok(env_config) = File::with_name(&format!("config.{}.yml", env)) {
        config.merge(env_config)?;
    }

    // 3. Load user settings with validation
    config.merge(File::with_name("data/settings.yaml")
        .format(FileFormat::Yaml)
        .required(false)
    )?;

    // 4. Override with environment variables (secure)
    config.merge(Environment::with_prefix("VISIONFLOW")
        .separator("__")
        .try_parsing(true)
        .ignore_empty(true)
    )?;

    // 5. Deserialize with comprehensive error reporting
    let raw_settings: RawSettings = config.try_into()
        .map_err(|e| ConfigurationError::DeserializationError {
            message: e.to_string(),
            context: "Failed to deserialize configuration".to_string(),
        })?;

    // 6. Comprehensive validation
    let validation_result = validate_configuration(&raw_settings).await?;
    
    // 7. Security validation
    let security_validation = validate_security_settings(&raw_settings).await?;

    // 8. Build production settings
    let settings = ProductionSettings::from_raw(raw_settings, validation_result, security_validation)?;

    let load_time = start_time.elapsed();
    info!("Configuration loaded successfully in {:?}", load_time);
    
    if load_time > Duration::from_millis(100) {
        warn!("Configuration loading took longer than expected: {:?}", load_time);
    }

    Ok(settings)
}

pub async fn validate_configuration(settings: &RawSettings) -> Result<ValidationResult, ConfigurationError> {
    let mut validator = ConfigurationValidator::new();
    
    // Validate server configuration
    validator.validate_server_config(&settings.server)?;
    
    // Validate GPU configuration
    if settings.gpu.enabled {
        validator.validate_gpu_config(&settings.gpu).await?;
    }
    
    // Validate network configuration
    validator.validate_network_config(&settings.network)?;
    
    // Validate feature flags
    validator.validate_feature_config(&settings.features)?;
    
    // Validate resource limits
    validator.validate_resource_limits(&settings.limits)?;
    
    Ok(validator.into_result())
}

pub async fn validate_security_settings(settings: &RawSettings) -> Result<SecurityValidation, ConfigurationError> {
    let mut security_validator = SecurityValidator::new();
    
    // Check for hardcoded secrets
    security_validator.check_hardcoded_secrets(&settings).await?;
    
    // Validate API key formats
    security_validator.validate_api_keys(&settings.api_keys).await?;
    
    // Validate CORS settings
    security_validator.validate_cors_config(&settings.cors)?;
    
    // Check security headers configuration
    security_validator.validate_security_headers(&settings.security)?;
    
    Ok(security_validator.into_result())
}
```

## Production Hot-Reload System

### Advanced Hot Reloading with Safety

VisionFlow implements a production-grade hot-reload system with comprehensive safety measures:

```rust
pub struct ProductionConfigurationManager {
    current_config: Arc<RwLock<ProductionSettings>>,
    file_watcher: RecommendedWatcher,
    validation_service: ValidationService,
    rollback_service: RollbackService,
    notification_service: NotificationService,
}

impl ProductionConfigurationManager {
    pub async fn start_hot_reload_monitoring(&mut self) -> Result<(), ConfigError> {
        let (tx, rx) = mpsc::channel();
        
        // Watch multiple configuration sources
        let paths_to_watch = vec![
            "config.yml",
            "data/settings.yaml",
            "config.prod.yml",
            "config.staging.yml",
        ];
        
        for path in paths_to_watch {
            if Path::new(path).exists() {
                self.file_watcher.watch(Path::new(path), RecursiveMode::NonRecursive)
                    .map_err(|e| ConfigError::WatcherSetup(e.to_string()))?;
            }
        }
        
        // Start event processing loop
        tokio::spawn(async move {
            while let Ok(event) = rx.recv() {
                if let Err(e) = Self::handle_config_change(event).await {
                    error!("Configuration hot-reload failed: {}", e);
                }
            }
        });
        
        info!("Configuration hot-reload monitoring started");
        Ok(())
    }
    
    async fn handle_config_change(event: notify::Event) -> Result<(), ConfigError> {
        match event.kind {
            EventKind::Modify(ModifyKind::Data(_)) => {
                info!("Configuration file change detected, initiating hot-reload");
                
                // 1. Create configuration backup
                let current_backup = Self::create_backup().await?;
                
                // 2. Load new configuration with validation
                match Self::load_and_validate_new_config().await {
                    Ok(new_config) => {
                        // 3. Test new configuration
                        if Self::test_configuration(&new_config).await.is_ok() {
                            // 4. Apply new configuration atomically
                            Self::apply_configuration_atomically(new_config).await?;
                            
                            // 5. Notify systems of configuration change
                            Self::broadcast_configuration_update().await?;
                            
                            info!("Configuration hot-reload completed successfully");
                            
                            // 6. Clean up old backup
                            Self::cleanup_backup(current_backup).await?;
                        } else {
                            warn!("New configuration failed testing, rolling back");
                            return Err(ConfigError::TestingFailed);
                        }
                    }
                    Err(e) => {
                        error!("Failed to load new configuration: {}", e);
                        
                        // Automatic rollback on validation failure
                        Self::rollback_to_backup(current_backup).await?;
                        return Err(ConfigError::ValidationFailed(e.to_string()));
                    }
                }
            }
            _ => {} // Ignore other events
        }
        
        Ok(())
    }
    
    async fn test_configuration(config: &ProductionSettings) -> Result<(), ConfigError> {
        // 1. Test database connections
        if let Some(db_config) = &config.database {
            Self::test_database_connection(db_config).await?;
        }
        
        // 2. Test external service endpoints
        for service in &config.external_services {
            Self::test_service_endpoint(service).await?;
        }
        
        // 3. Test GPU configuration if enabled
        if config.gpu.enabled {
            Self::test_gpu_configuration(&config.gpu).await?;
        }
        
        // 4. Validate resource limits don't exceed system capacity
        Self::validate_resource_availability(config).await?;
        
        Ok(())
    }
    
    async fn apply_configuration_atomically(new_config: ProductionSettings) -> Result<(), ConfigError> {
        // Use atomic swap to update configuration
        let config_lock = GLOBAL_CONFIG.write().await;
        let old_config = config_lock.clone();
        
        // Apply the new configuration
        *config_lock = Arc::new(new_config);
        drop(config_lock);
        
        // Notify all registered configuration watchers
        CONFIG_UPDATE_NOTIFIER.notify_waiters();
        
        info!("Configuration applied atomically");
        Ok(())
    }
}
```

### Configuration Change Notification System

```rust
pub struct ConfigurationChangeNotification {
    pub timestamp: SystemTime,
    pub change_type: ChangeType,
    pub affected_sections: Vec<String>,
    pub validation_results: ValidationResult,
    pub rollback_available: bool,
}

pub enum ChangeType {
    HotReload,
    EnvironmentVariableUpdate,
    RuntimeUpdate,
    Rollback,
    InitialLoad,
}

impl ConfigurationManager {
    pub async fn subscribe_to_config_changes(&self) -> ConfigChangeReceiver {
        self.notification_service.subscribe().await
    }
    
    pub async fn broadcast_config_change(&self, notification: ConfigurationChangeNotification) {
        self.notification_service.broadcast(notification).await;
        
        // Log configuration changes for audit
        info!("Configuration change: {:?} affecting sections: {:?}", 
              notification.change_type, notification.affected_sections);
    }
}
```

### Feature Flags

Dynamic feature toggles:

```rust
// Check if feature is enabled
if settings.features.is_enabled("ai_chat") {
    // Enable AI chat functionality
}

// Toggle feature at runtime
settings_actor.send(SetSettingByPath {
    path: "features.enabled_features".to_string(),
    value: json!(["graph_visualization", "ai_chat"]),
}).await?;
```

## Security Considerations

### Sensitive Data

1. **Never commit secrets**: Use environment variables
   ```yaml
   api_key: "${API_KEY}"  # Good
   api_key: "sk-12345"    # Bad
   ```

2. **Rotate keys regularly**
   ```bash
   # Generate new API key
   openssl rand -hex 32
   ```

3. **Use secrets management**
   ```bash
   # Docker secrets
   docker secret create api_key ./api_key.txt
   ```

### Access Control

```yaml
# Restrict feature access
features:
  power_users_only:
    - admin_panel
    - gpu_compute
    - bulk_operations
```

## Configuration Complexity Guide

### Understanding the Settings System

The VisionFlow configuration system has multiple layers that can seem complex at first. Here's a guide to understanding how they work together:

#### 1. Configuration Sources Priority
```
Highest Priority (Overrides everything below)
↓ Environment Variables (e.g., ENABLE_GPU_PHYSICS=true)
↓ Command-line arguments (e.g., --port 8080)
↓ User-specific settings (/app/user_settings/<pubkey>.yaml)
↓ Main settings file (data/settings.yaml)
↓ Default configuration (config.yml)
Lowest Priority (Default values in code)
```

#### 2. Common Configuration Scenarios

**Scenario: Enable GPU Physics**
```bash
# Method 1: Environment variable (temporary)
export ENABLE_GPU_PHYSICS=true

# Method 2: settings.yaml (persistent)
# Edit data/settings.yaml:
gpu:
  enabled: true
  device_id: 0

# Method 3: Docker compose (deployment)
# In docker-compose.yml:
environment:
  - ENABLE_GPU_PHYSICS=true
  - NVIDIA_GPU_UUID=auto  # Auto-detect GPU
```

**Scenario: Configure AI Services**
```yaml
# In data/settings.yaml:
ragflow:
  api_key: "${RAGFLOW_API_KEY}"  # Reference env var
  base_url: "https://api.ragflow.com"
  timeout: 30

perplexity:
  api_key: "${PERPLEXITY_API_KEY}"
  model: "mixtral-8x7b-instruct"
```

#### 3. Settings Actor Communication
```rust
// Frontend requests settings
WebSocket → SettingsActor::GetUISettings → UISettings → Frontend

// Backend updates settings
API → SettingsActor::SetSettingByPath → Validation → Persistence → Broadcast
```

## Production Best Practices

### 1. Environment-Specific Configuration Management

**Multi-Environment Setup:**
```bash
# Production-ready environment configuration
configs/
├── config.base.yml           # Base configuration (common settings)
├── config.development.yml    # Development overrides
├── config.staging.yml        # Staging environment
├── config.production.yml     # Production environment
├── secrets.development.yml   # Development secrets (encrypted)
├── secrets.staging.yml       # Staging secrets (encrypted)
└── secrets.production.yml    # Production secrets (encrypted)

# Load configuration based on environment
export ENVIRONMENT=production
export CONFIG_ENCRYPTION_KEY=your-encryption-key
./visionflow-server
```

**Configuration Inheritance:**
```yaml
# config.base.yml
server:
  host: "0.0.0.0"
  port: 4000
  timeout: 30

# config.production.yml (extends base)
extends: config.base.yml
server:
  workers: 8  # Override for production
  timeout: 60 # Longer timeout for production

security:
  enable_audit_logging: true
  strict_validation: true

monitoring:
  metrics_enabled: true
  health_check_interval: 10
```

### 2. Comprehensive Configuration Validation

**Production Validation Tests:**
```rust
#[cfg(test)]
mod production_config_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_production_config_validation() {
        let config = load_production_config().await.expect("Config should load");
        
        // Test all critical validations
        assert!(config.validate_production_readiness().await.is_ok());
        assert!(config.validate_security_requirements().await.is_ok());
        assert!(config.validate_performance_limits().await.is_ok());
        assert!(config.validate_resource_availability().await.is_ok());
    }
    
    #[tokio::test] 
    async fn test_configuration_hot_reload() {
        let mut manager = ConfigurationManager::new().await;
        let initial_config = manager.get_current_config().await;
        
        // Modify configuration file
        let test_config = create_test_config_modification();
        write_test_config_file(&test_config).await;
        
        // Wait for hot reload
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        let updated_config = manager.get_current_config().await;
        assert_ne!(initial_config.hash(), updated_config.hash());
        
        // Verify configuration was applied correctly
        assert_eq!(updated_config.test_field, test_config.test_field);
    }
    
    #[tokio::test]
    async fn test_configuration_rollback() {
        let mut manager = ConfigurationManager::new().await;
        
        // Create invalid configuration
        let invalid_config = create_invalid_test_config();
        write_test_config_file(&invalid_config).await;
        
        // Wait for hot reload attempt
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        // Verify configuration was rolled back
        let current_config = manager.get_current_config().await;
        assert!(current_config.is_valid());
        
        // Check rollback notification was sent
        let notifications = manager.get_recent_notifications().await;
        assert!(notifications.iter().any(|n| matches!(n.change_type, ChangeType::Rollback)));
    }
}
```

### 3. Security-First Configuration

**Secret Management:**
```yaml
# config.production.yml - No secrets in plain text
database:
  url: "${DATABASE_URL}"  # Reference to environment variable
  
security:
  jwt_secret: "${JWT_SECRET}"
  encryption_key: "${ENCRYPTION_KEY}"
  
external_services:
  ragflow:
    api_key: "${RAGFLOW_API_KEY}"
  perplexity:
    api_key: "${PERPLEXITY_API_KEY}"
```

**Configuration Encryption:**
```rust
pub struct EncryptedConfigLoader {
    encryption_key: String,
    cache: Arc<RwLock<HashMap<String, EncryptedConfig>>>,
}

impl EncryptedConfigLoader {
    pub async fn load_encrypted_config(&self, path: &str) -> Result<DecryptedConfig, ConfigError> {
        // 1. Load encrypted configuration file
        let encrypted_content = tokio::fs::read_to_string(path).await
            .map_err(|e| ConfigError::FileRead(e.to_string()))?;
        
        // 2. Decrypt configuration
        let decrypted_content = self.decrypt_config(&encrypted_content)?;
        
        // 3. Parse and validate
        let config: DecryptedConfig = serde_yaml::from_str(&decrypted_content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        
        // 4. Validate decrypted configuration
        self.validate_decrypted_config(&config).await?;
        
        // 5. Cache for performance (with TTL)
        self.cache_config(path.to_string(), config.clone()).await;
        
        Ok(config)
    }
}
```

### 4. Production Documentation Standards

**Comprehensive Configuration Documentation:**
```yaml
# config.production.yml with extensive documentation
server:
  host: "0.0.0.0"                    # Bind address (use 0.0.0.0 for containers)
  port: 4000                         # Server port (ensure firewall allows)
  workers: 8                         # Worker threads (2x CPU cores recommended)
  shutdown_timeout: 30               # Graceful shutdown timeout (seconds)
  max_connections: 10000             # Maximum concurrent connections
  request_timeout: 60                # Request timeout (seconds)
  
  # Security settings
  enable_tls: true                   # Enable TLS/HTTPS (required for production)
  tls_cert_path: "/etc/ssl/cert.pem" # TLS certificate path
  tls_key_path: "/etc/ssl/key.pem"   # TLS private key path

graph:
  max_nodes: 100000                  # Maximum graph nodes (affects memory: ~100MB per 10K nodes)
  max_edges: 500000                  # Maximum graph edges (affects memory: ~50MB per 100K edges)
  
  simulation:
    update_rate: 60.0                # Physics update rate (FPS) - higher uses more CPU
    repulsion_strength: 100.0        # Node repulsion force (1-1000, higher = more spread)
    attraction_strength: 0.01        # Edge attraction force (0.001-1, higher = tighter)
    damping: 0.9                     # Velocity damping (0.1-1, higher = more stable)
    time_step: 0.016                 # Simulation timestep (0.001-0.1, lower = more accurate)

gpu:
  enabled: true                      # Enable GPU acceleration (requires NVIDIA GPU + CUDA)
  device_id: 0                       # GPU device ID (use nvidia-smi to list)
  memory_limit_mb: 4096             # GPU memory limit (MB) - leave headroom for system
  fallback_to_cpu: true             # Enable CPU fallback on GPU failure
  safety_checks: true               # Enable comprehensive GPU safety checks
  temperature_limit: 85             # GPU temperature limit (°C) before throttling
```

### 5. Production Monitoring Integration

**Configuration Health Monitoring:**
```rust
pub struct ConfigurationHealthMonitor {
    health_checker: HealthChecker,
    metrics_collector: MetricsCollector,
    alert_manager: AlertManager,
}

impl ConfigurationHealthMonitor {
    pub async fn monitor_configuration_health(&self) -> ConfigHealthStatus {
        let mut status = ConfigHealthStatus::new();
        
        // Check configuration file integrity
        status.file_integrity = self.check_config_file_integrity().await;
        
        // Validate current configuration
        status.validation_status = self.validate_current_config().await;
        
        // Check configuration performance impact
        status.performance_impact = self.measure_config_performance().await;
        
        // Verify security settings
        status.security_status = self.audit_security_config().await;
        
        // Check resource usage against limits
        status.resource_status = self.check_resource_utilization().await;
        
        // Send alerts if issues detected
        if status.has_issues() {
            self.alert_manager.send_config_alert(&status).await;
        }
        
        status
    }
}
```

### 6. Configuration Versioning and Rollback

**Version Control Integration:**
```rust
pub struct ConfigurationVersionManager {
    git_repo: git2::Repository,
    version_history: Vec<ConfigVersion>,
    rollback_stack: VecDeque<ConfigSnapshot>,
}

impl ConfigurationVersionManager {
    pub async fn save_configuration_version(&mut self, config: &ProductionSettings, message: &str) -> Result<String, VersionError> {
        let version_id = generate_version_id();
        
        // Create configuration snapshot
        let snapshot = ConfigSnapshot {
            id: version_id.clone(),
            timestamp: SystemTime::now(),
            config: config.clone(),
            message: message.to_string(),
            hash: config.compute_hash(),
        };
        
        // Save to version control
        self.save_to_git(&snapshot).await?;
        
        // Add to rollback stack (keep last 10 versions)
        self.rollback_stack.push_front(snapshot);
        if self.rollback_stack.len() > 10 {
            self.rollback_stack.pop_back();
        }
        
        info!("Configuration version {} saved: {}", version_id, message);
        Ok(version_id)
    }
    
    pub async fn rollback_to_version(&mut self, version_id: &str) -> Result<ProductionSettings, VersionError> {
        let snapshot = self.find_version(version_id)?;
        
        // Validate that rollback version is compatible
        self.validate_rollback_compatibility(&snapshot.config).await?;
        
        // Apply rollback configuration
        let rollback_config = snapshot.config.clone();
        
        // Save current configuration before rollback
        let current_config = GLOBAL_CONFIG.read().await;
        self.save_configuration_version(&current_config, &format!("Pre-rollback snapshot before reverting to {}", version_id)).await?;
        
        info!("Rolling back configuration to version: {} ({})", version_id, snapshot.message);
        Ok(rollback_config)
    }
}
```

## Troubleshooting

### Common Issues

1. **Configuration not loading**
   ```bash
   # Check file permissions
   ls -la config.yml

   # Validate YAML syntax
   yamllint config.yml
   ```

2. **Environment variables not working**
   ```bash
   # Debug environment
   env | grep APP_

   # Check variable expansion
   echo $DATABASE_URL
   ```

3. **Type mismatches**
   ```yaml
   # Ensure correct types
   port: 8080        # Number, not "8080"
   enabled: true     # Boolean, not "true"
   ```

### Debug Configuration

```rust
// Enable config debugging
std::env::set_var("CONFIG_DEBUG", "true");

// Log loaded configuration
info!("Loaded config: {:?}", settings);
```

### Configuration Validation Errors

1. **Missing Required Fields**
   ```
   Error: missing field `api_key` at line 15 column 3
   Solution: Ensure all required fields are present in settings.yaml
   ```

2. **Type Mismatches**
   ```
   Error: invalid type: string "true", expected a boolean
   Solution: Use boolean values without quotes (true, not "true")
   ```

3. **Environment Variable Not Found**
   ```
   Error: environment variable `RAGFLOW_API_KEY` not found
   Solution: Set the variable or provide a default value
   ```

### Settings Actor Issues

1. **Settings Not Updating**
   ```rust
   // Check if settings actor is running
   let settings = settings_actor.send(GetSettings).await?;
   log::info!("Current settings: {:?}", settings);
   ```

2. **WebSocket Settings Sync Failed**
   ```
   Error: Failed to broadcast settings update
   Solution: Check WebSocket connection and client handlers
   ```

### Docker Configuration Issues

1. **MCP Connection Failed**
   ```bash
   # Verify network exists
   docker network ls | grep mcp-visionflow-net

   # Check container connectivity
   docker exec visionflow-server ping multi-agent-container
   ```

2. **GPU Not Detected**
   ```bash
   # Verify GPU runtime
   docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi

   # Check GPU UUID
   nvidia-smi --query-gpu=uuid --format=csv,noheader
   ```

## Common Configuration Patterns

### Docker Network Configuration
```yaml
# Agent control URL for MCP connection
AGENT_CONTROL_URL: "tcp://multi-agent-container:9500"

# Network settings
networks:
  visionflow-net:
    name: "mcp-visionflow-net"
    external: true
```

### GPU Configuration
```bash
# Auto-detect GPU
NVIDIA_GPU_UUID=auto

# Or specify exact GPU
NVIDIA_GPU_UUID="GPU-553dc306-dab3-32e2-c69b-28175a6f4da6"

# Fallback to CPU if GPU fails
ENABLE_GPU_PHYSICS=true
GPU_FALLBACK_TO_CPU=true
```

### Feature Flags
```yaml
# Enable specific features
features:
  enabled_features:
    - graph_visualization
    - ai_chat
    - visionflow_multi-agent
    - gpu_physics

  # Power user access
  power_user_keys:
    - "${POWER_USER_KEY_1}"
    - "${POWER_USER_KEY_2}"
```

## Production Migration and Upgrade Guide

### Migration Strategy for Production Systems

**Zero-Downtime Migration Process:**
```bash
#!/bin/bash
# production-migration.sh

set -e

echo "Starting VisionFlow production migration..."

# 1. Backup current configuration
echo "Creating configuration backup..."
cp -r /etc/visionflow/config /etc/visionflow/config.backup.$(date +%Y%m%d_%H%M%S)

# 2. Validate new configuration
echo "Validating new configuration..."
./visionflow-server --validate-config --config-path=./config.new.yml

# 3. Test new configuration in staging
echo "Testing configuration in staging environment..."
./test-config-staging.sh

# 4. Blue-green deployment with configuration migration
echo "Starting blue-green deployment..."
./scripts/blue-green-deploy.sh --config-migration

# 5. Health check after migration
echo "Performing post-migration health checks..."
./scripts/health-check.sh --comprehensive

# 6. Rollback capability check
echo "Verifying rollback capability..."
./scripts/test-rollback.sh --dry-run

echo "Migration completed successfully!"
```

### Configuration Schema Evolution

**From LogseqXR to VisionFlow (Production):**
```yaml
# Legacy LogseqXR configuration (deprecated)
logseq_xr:
  visualisation:
    type: "spring"
    physics:
      repulsion: 100
      attraction: 0.01
  
  server:
    port: 3000
    host: "localhost"

# VisionFlow Production Configuration (current)
visionflow:
  # Enhanced graph configuration with production features
  graph:
    dual_mode: true
    knowledge_graph:
      enabled: true
      source: "logseq"
      sync_interval: 30
    agent_graph:
      enabled: true
      mcp_integration: true
    
    # Production physics configuration
    simulation:
      repulsion_strength: 100.0
      attraction_strength: 0.01
      adaptive_balancing: true
      stability_monitoring: true
  
  # Production server configuration
  server:
    host: "0.0.0.0"
    port: 4000
    workers: 8
    production_mode: true
    health_checks: true
    metrics: true
  
  # New production features
  security:
    zero_trust: true
    input_validation: comprehensive
    audit_logging: true
  
  monitoring:
    performance_tracking: true
    resource_monitoring: true
    alert_thresholds:
      cpu_usage: 80
      memory_usage: 85
      gpu_temperature: 80
```

**From v1 to v2 (Production Upgrade):**
```yaml
# v1 Configuration (legacy)
ai_service:
  type: "ragflow"
  key: "xxx"
  timeout: 30

database:
  type: "sqlite"
  path: "./data.db"

server:
  port: 3000

# v2 Production Configuration (current)
external_services:
  ragflow:
    api_key: "${RAGFLOW_API_KEY}"
    base_url: "https://api.ragflow.com"
    timeout: 30
    retry_attempts: 3
    circuit_breaker: true
    health_check: true
  
  perplexity:
    api_key: "${PERPLEXITY_API_KEY}"
    base_url: "https://api.perplexity.ai"
    model: "mixtral-8x7b-instruct"

database:
  # Production database configuration
  primary:
    type: "postgresql"
    url: "${DATABASE_URL}"
    max_connections: 20
    connection_timeout: 30
    health_check: true
  
  # Read replicas for scaling
  read_replicas:
    - url: "${DATABASE_READ_REPLICA_1_URL}"
    - url: "${DATABASE_READ_REPLICA_2_URL}"
  
  # Backup configuration
  backup:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention_days: 30

server:
  port: 4000
  production_mode: true
  
  # Production server features
  clustering:
    enabled: true
    nodes: 3
  
  load_balancing:
    enabled: true
    strategy: "round_robin"
  
  ssl:
    enabled: true
    cert_path: "/etc/ssl/cert.pem"
    key_path: "/etc/ssl/key.pem"
```

### Automated Migration Tools

**Configuration Migration Script:**
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct LegacyConfig {
    pub logseq_xr: Option<LegacyLogseqConfig>,
    pub ai_service: Option<LegacyAIService>,
    pub server: Option<LegacyServerConfig>,
}

#[derive(Debug, Serialize)]
pub struct ProductionConfig {
    pub visionflow: VisionFlowConfig,
    pub external_services: ExternalServicesConfig,
    pub server: ProductionServerConfig,
    pub monitoring: MonitoringConfig,
    pub security: SecurityConfig,
}

pub struct ConfigurationMigrator {
    migration_rules: HashMap<String, Box<dyn MigrationRule>>,
    validation_service: ValidationService,
}

impl ConfigurationMigrator {
    pub async fn migrate_to_production(&self, legacy_config: LegacyConfig) -> Result<ProductionConfig, MigrationError> {
        let mut production_config = ProductionConfig::default();
        
        // 1. Migrate server configuration
        if let Some(legacy_server) = legacy_config.server {
            production_config.server = self.migrate_server_config(legacy_server)?;
        }
        
        // 2. Migrate AI service configuration
        if let Some(legacy_ai) = legacy_config.ai_service {
            production_config.external_services = self.migrate_ai_services(legacy_ai)?;
        }
        
        // 3. Migrate visualization configuration
        if let Some(legacy_viz) = legacy_config.logseq_xr {
            production_config.visionflow = self.migrate_visualization_config(legacy_viz)?;
        }
        
        // 4. Add production-specific configurations
        production_config.monitoring = self.create_default_monitoring_config();
        production_config.security = self.create_default_security_config();
        
        // 5. Validate migrated configuration
        self.validation_service.validate_production_config(&production_config).await?;
        
        // 6. Generate migration report
        let report = self.generate_migration_report(&legacy_config, &production_config);
        info!("Migration completed successfully: {}", report);
        
        Ok(production_config)
    }
    
    pub fn generate_migration_report(&self, legacy: &LegacyConfig, production: &ProductionConfig) -> MigrationReport {
        MigrationReport {
            migrated_sections: vec!["server", "ai_services", "visualization"],
            new_features_added: vec!["monitoring", "security", "health_checks", "circuit_breakers"],
            deprecated_features: vec!["sqlite_database", "single_instance_mode"],
            manual_review_required: vec!["api_keys", "ssl_certificates", "database_migration"],
            estimated_downtime: Duration::from_secs(60), // 1 minute
        }
    }
}

#[derive(Debug)]
pub struct MigrationReport {
    pub migrated_sections: Vec<&'static str>,
    pub new_features_added: Vec<&'static str>,
    pub deprecated_features: Vec<&'static str>,
    pub manual_review_required: Vec<&'static str>,
    pub estimated_downtime: Duration,
}
```

## Related Documentation

- [Quick Reference](./quick-reference.md) - Essential configuration cheatsheet
- [Server Configuration](../server/config.md) - Server-specific settings
- [Feature Access](../server/feature-access.md) - Feature flag system
- [Environment Setup](../development/setup.md) - Development environment
- [Deployment](../deployment/index.md) - Production configuration
- [MCP Architecture](../mcp-architecture.md) - Understanding agent connections
// High-Performance WebSocket Settings Handler with Delta Compression
// Implements binary protocol, delta synchronization, and bandwidth optimization

use actix::prelude::*;
use actix_web_actors::ws;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use log::{info, error, debug, warn};
use flate2::{Compress, Decompress, Compression, FlushCompress, FlushDecompress};
use flate2::Status;
use blake3::Hasher;
use crate::actors::messages::{GetSettingsByPaths, SetSettingsByPaths};
use crate::app_state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketSettingsMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub data: Value,
    pub timestamp: u64,
    pub compression: Option<String>,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaUpdate {
    pub path: String,
    pub value: Value,
    pub old_value: Option<Value>,
    pub operation: DeltaOperation,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaOperation {
    Set,
    Delete,
    Batch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequest {
    pub last_sync: u64,
    pub client_id: String,
    pub compression_supported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDelta {
    pub bandwidth_saved: u64,
    pub compression_ratio: f64,
    pub message_count: u64,
}

pub struct WebSocketSettingsHandler {
    client_id: String,
    app_state: actix_web::web::Data<AppState>,
    last_sync: u64,
    settings_cache: HashMap<String, CachedSetting>,
    compression_enabled: bool,
    compressor: Compress,
    decompressor: Decompress,
    metrics: WebSocketMetrics,
    heartbeat: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedSetting {
    value: Value,
    hash: String,
    timestamp: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct WebSocketMetrics {
    messages_sent: u64,
    messages_received: u64,
    bytes_sent: u64,
    bytes_received: u64,
    compression_ratio: f64,
    delta_messages: u64,
    full_sync_messages: u64,
}

impl WebSocketSettingsHandler {
    pub fn new(app_state: actix_web::web::Data<AppState>) -> Self {
        let client_id = uuid::Uuid::new_v4().to_string();
        
        Self {
            client_id,
            app_state,
            last_sync: Self::current_timestamp(),
            settings_cache: HashMap::new(),
            compression_enabled: true,
            compressor: Compress::new(Compression::default(), false),
            decompressor: Decompress::new(false),
            metrics: WebSocketMetrics::default(),
            heartbeat: Instant::now(),
        }
    }
    
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    fn calculate_hash(value: &Value) -> String {
        let mut hasher = Hasher::new();
        if let Ok(json_str) = serde_json::to_string(value) {
            hasher.update(json_str.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }
    
    fn compress_data(&mut self, data: &[u8]) -> Result<Vec<u8>, String> {
        let mut compressed = Vec::new();
        let mut output = vec![0; data.len() * 2];
        
        let status = self.compressor.compress_vec(
            data,
            &mut output,
            FlushCompress::Finish,
        ).map_err(|e| format!("Compression error: {}", e))?;
        
        if status == Status::StreamEnd {
            let compressed_size = self.compressor.total_out() as usize;
            output.truncate(compressed_size);
            compressed.extend(output);
            
            // Update compression metrics
            let original_size = data.len();
            let compressed_size = compressed.len();
            let ratio = 1.0 - (compressed_size as f64 / original_size as f64);
            self.metrics.compression_ratio = 
                (self.metrics.compression_ratio + ratio) / 2.0; // Running average
            
            debug!("Compressed {} bytes to {} bytes (ratio: {:.2}%)",
                   original_size, compressed_size, ratio * 100.0);
        }
        
        Ok(compressed)
    }
    
    fn decompress_data(&mut self, compressed: &[u8]) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();
        let mut buffer = vec![0; compressed.len() * 4];
        
        let status = self.decompressor.decompress_vec(
            compressed,
            &mut buffer,
            FlushDecompress::Finish,
        ).map_err(|e| format!("Decompression error: {}", e))?;
        
        if status == Status::StreamEnd {
            let decompressed_size = self.decompressor.total_out() as usize;
            buffer.truncate(decompressed_size);
            output.extend(buffer);
        }
        
        Ok(output)
    }
    
    fn send_compressed_message(&mut self, ctx: &mut ws::WebsocketContext<Self>, message: &WebSocketSettingsMessage) {
        let json_str = match serde_json::to_string(message) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to serialize WebSocket message: {}", e);
                return;
            }
        };
        
        let message_bytes = json_str.as_bytes();
        let original_size = message_bytes.len();
        
        if self.compression_enabled && original_size > 1024 { // Only compress larger messages
            match self.compress_data(message_bytes) {
                Ok(compressed) => {
                    let mut compressed_message = message.clone();
                    compressed_message.compression = Some("gzip".to_string());
                    compressed_message.checksum = Some(Self::calculate_hash(&message.data));
                    
                    // Send as binary message
                    let compressed_len = compressed.len();
                    ctx.binary(compressed);
                    self.metrics.bytes_sent += compressed_len as u64;
                    
                    debug!("Sent compressed message: {} -> {} bytes", original_size, compressed_len);
                }
                Err(e) => {
                    warn!("Compression failed, sending uncompressed: {}", e);
                    ctx.text(json_str);
                    self.metrics.bytes_sent += original_size as u64;
                }
            }
        } else {
            // Send uncompressed
            ctx.text(json_str);
            self.metrics.bytes_sent += original_size as u64;
        }
        
        self.metrics.messages_sent += 1;
    }
    
    fn handle_setting_change(&mut self, ctx: &mut ws::WebsocketContext<Self>, path: String, new_value: Value) {
        let timestamp = Self::current_timestamp();
        let hash = Self::calculate_hash(&new_value);
        
        // Check if this is actually a change
        let old_value = if let Some(cached) = self.settings_cache.get(&path) {
            if cached.hash == hash {
                // No change, skip
                return;
            }
            Some(cached.value.clone())
        } else {
            None
        };
        
        // Update cache
        self.settings_cache.insert(path.clone(), CachedSetting {
            value: new_value.clone(),
            hash,
            timestamp,
        });
        
        // Send delta update
        let delta = DeltaUpdate {
            path: path.clone(),
            value: new_value,
            old_value,
            operation: DeltaOperation::Set,
            timestamp,
        };
        
        let message = WebSocketSettingsMessage {
            msg_type: "settingsDelta".to_string(),
            data: serde_json::to_value(&delta).unwrap_or_default(),
            timestamp,
            compression: None,
            checksum: None,
        };
        
        self.send_compressed_message(ctx, &message);
        self.metrics.delta_messages += 1;
        
        info!("Sent delta update for setting: {}", path);
    }
    
    fn handle_batch_setting_changes(&mut self, ctx: &mut ws::WebsocketContext<Self>, updates: Vec<(String, Value)>) {
        let timestamp = Self::current_timestamp();
        let mut deltas = Vec::new();
        
        for (path, new_value) in updates {
            let hash = Self::calculate_hash(&new_value);
            
            // Check for actual changes
            let old_value = if let Some(cached) = self.settings_cache.get(&path) {
                if cached.hash == hash {
                    continue; // No change
                }
                Some(cached.value.clone())
            } else {
                None
            };
            
            // Update cache
            self.settings_cache.insert(path.clone(), CachedSetting {
                value: new_value.clone(),
                hash,
                timestamp,
            });
            
            deltas.push(DeltaUpdate {
                path,
                value: new_value,
                old_value,
                operation: DeltaOperation::Set,
                timestamp,
            });
        }
        
        if deltas.is_empty() {
            return; // No actual changes
        }
        
        // Send batch delta
        let message = WebSocketSettingsMessage {
            msg_type: "settingsBatchDelta".to_string(),
            data: serde_json::to_value(&deltas).unwrap_or_default(),
            timestamp,
            compression: None,
            checksum: None,
        };
        
        self.send_compressed_message(ctx, &message);
        self.metrics.delta_messages += 1;
        
        info!("Sent batch delta update for {} settings", deltas.len());
    }
    
    fn handle_sync_request(&mut self, ctx: &mut ws::WebsocketContext<Self>, request: SyncRequest) {
        info!("Handling sync request from client {} (last_sync: {})", 
              request.client_id, request.last_sync);
        
        self.compression_enabled = request.compression_supported;
        
        // For now, send full sync (in production, implement delta sync)
        let message = WebSocketSettingsMessage {
            msg_type: "fullSync".to_string(),
            data: serde_json::to_value(&self.settings_cache).unwrap_or_default(),
            timestamp: Self::current_timestamp(),
            compression: None,
            checksum: None,
        };
        
        self.send_compressed_message(ctx, &message);
        self.metrics.full_sync_messages += 1;
        
        // Update client's last sync timestamp
        self.last_sync = Self::current_timestamp();
    }
    
    fn handle_heartbeat(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_secs(30), |act, ctx| {
            if Instant::now().duration_since(act.heartbeat) > Duration::from_secs(60) {
                info!("WebSocket client heartbeat timeout, closing connection");
                ctx.stop();
                return;
            }
            
            // Send ping
            let message = WebSocketSettingsMessage {
                msg_type: "ping".to_string(),
                data: serde_json::to_value(&act.metrics).unwrap_or_default(),
                timestamp: Self::current_timestamp(),
                compression: None,
                checksum: None,
            };
            
            act.send_compressed_message(ctx, &message);
        });
    }
    
    fn handle_performance_request(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
        let bandwidth_saved = if self.metrics.messages_sent > 0 {
            // Estimate bandwidth savings from compression and delta updates
            let estimated_full_size = self.metrics.messages_sent * 50000; // ~50KB per full sync
            let actual_size = self.metrics.bytes_sent;
            estimated_full_size.saturating_sub(actual_size)
        } else {
            0
        };
        
        let performance_delta = PerformanceDelta {
            bandwidth_saved,
            compression_ratio: self.metrics.compression_ratio,
            message_count: self.metrics.messages_sent,
        };
        
        let message = WebSocketSettingsMessage {
            msg_type: "performanceMetrics".to_string(),
            data: serde_json::to_value(&performance_delta).unwrap_or_default(),
            timestamp: Self::current_timestamp(),
            compression: None,
            checksum: None,
        };
        
        self.send_compressed_message(ctx, &message);
        
        info!("Performance metrics - Messages: {}, Bandwidth saved: {} bytes, Compression: {:.1}%",
              self.metrics.messages_sent, bandwidth_saved, self.metrics.compression_ratio * 100.0);
    }
}

impl Actor for WebSocketSettingsHandler {
    type Context = ws::WebsocketContext<Self>;
    
    fn started(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket settings handler started for client: {}", self.client_id);
        
        // Start heartbeat
        self.handle_heartbeat(ctx);
        
        // Send initial welcome message
        let welcome_message = WebSocketSettingsMessage {
            msg_type: "connected".to_string(),
            data: serde_json::json!({
                "clientId": self.client_id,
                "compressionEnabled": self.compression_enabled,
                "features": ["delta-sync", "compression", "batch-updates"]
            }),
            timestamp: Self::current_timestamp(),
            compression: None,
            checksum: None,
        };
        
        self.send_compressed_message(ctx, &welcome_message);
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("WebSocket settings handler stopped for client: {}", self.client_id);
        
        // Log final metrics
        info!("Final metrics - Messages sent: {}, received: {}, bytes sent: {}, compression ratio: {:.1}%",
              self.metrics.messages_sent, self.metrics.messages_received, 
              self.metrics.bytes_sent, self.metrics.compression_ratio * 100.0);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSettingsHandler {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                self.heartbeat = Instant::now();
                self.metrics.messages_received += 1;
                self.metrics.bytes_received += text.len() as u64;
                
                // Parse message
                match serde_json::from_str::<WebSocketSettingsMessage>(&text) {
                    Ok(ws_message) => {
                        match ws_message.msg_type.as_str() {
                            "syncRequest" => {
                                if let Ok(request) = serde_json::from_value::<SyncRequest>(ws_message.data) {
                                    self.handle_sync_request(ctx, request);
                                }
                            }
                            "settingUpdate" => {
                                if let Ok(update) = serde_json::from_value::<HashMap<String, Value>>(ws_message.data) {
                                    for (path, value) in update {
                                        self.handle_setting_change(ctx, path, value);
                                    }
                                }
                            }
                            "batchUpdate" => {
                                if let Ok(updates) = serde_json::from_value::<Vec<(String, Value)>>(ws_message.data) {
                                    self.handle_batch_setting_changes(ctx, updates);
                                }
                            }
                            "performanceRequest" => {
                                self.handle_performance_request(ctx);
                            }
                            "pong" => {
                                debug!("Received pong from client {}", self.client_id);
                            }
                            _ => {
                                warn!("Unknown WebSocket message type: {}", ws_message.msg_type);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to parse WebSocket message: {}", e);
                    }
                }
            }
            
            Ok(ws::Message::Binary(bytes)) => {
                self.heartbeat = Instant::now();
                self.metrics.messages_received += 1;
                self.metrics.bytes_received += bytes.len() as u64;
                
                // Try to decompress
                match self.decompress_data(&bytes) {
                    Ok(decompressed) => {
                        if let Ok(text) = String::from_utf8(decompressed) {
                            // Recursively handle as text message
                            let text_msg = Ok(ws::Message::Text(text.into()));
                            <Self as StreamHandler<Result<ws::Message, ws::ProtocolError>>>::handle(self, text_msg, ctx);
                        } else {
                            error!("Failed to convert decompressed data to UTF-8");
                        }
                    }
                    Err(e) => {
                        error!("Failed to decompress binary message: {}", e);
                    }
                }
            }
            
            Ok(ws::Message::Ping(msg)) => {
                self.heartbeat = Instant::now();
                ctx.pong(&msg);
            }
            
            Ok(ws::Message::Pong(_)) => {
                self.heartbeat = Instant::now();
            }
            
            Ok(ws::Message::Close(reason)) => {
                info!("WebSocket closing: {:?}", reason);
                ctx.stop();
            }
            
            Err(e) => {
                error!("WebSocket protocol error: {}", e);
                ctx.stop();
            }
            
            _ => {
                warn!("Unhandled WebSocket message type");
            }
        }
    }
}

// Message handlers for integration with settings actor

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastSettingChange {
    pub path: String,
    pub value: Value,
    pub client_id: Option<String>,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastBatchChange {
    pub updates: Vec<(String, Value)>,
    pub client_id: Option<String>,
}

impl Handler<BroadcastSettingChange> for WebSocketSettingsHandler {
    type Result = ();
    
    fn handle(&mut self, msg: BroadcastSettingChange, ctx: &mut Self::Context) {
        // Don't echo back to the sender
        if let Some(sender_id) = &msg.client_id {
            if sender_id == &self.client_id {
                return;
            }
        }
        
        self.handle_setting_change(ctx, msg.path, msg.value);
    }
}

impl Handler<BroadcastBatchChange> for WebSocketSettingsHandler {
    type Result = ();
    
    fn handle(&mut self, msg: BroadcastBatchChange, ctx: &mut Self::Context) {
        // Don't echo back to the sender
        if let Some(sender_id) = &msg.client_id {
            if sender_id == &self.client_id {
                return;
            }
        }
        
        self.handle_batch_setting_changes(ctx, msg.updates);
    }
}

// WebSocket route handler
pub async fn websocket_settings(
    req: actix_web::HttpRequest,
    stream: actix_web::web::Payload,
    app_state: actix_web::web::Data<AppState>,
) -> Result<actix_web::HttpResponse, actix_web::Error> {
    let resp = ws::start(
        WebSocketSettingsHandler::new(app_state),
        &req,
        stream,
    );
    
    info!("New WebSocket settings connection established");
    resp
}

impl WebSocketSettingsHandler {
    fn send_reliable_message(&mut self, ctx: &mut ws::WebsocketContext<Self>, message: &WebSocketSettingsMessage) {
        // Direct send without circuit breaker for critical messages like heartbeats
        let json_str = match serde_json::to_string(message) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to serialize reliable WebSocket message: {}", e);
                return;
            }
        };

        ctx.text(json_str);
        self.metrics.messages_sent += 1;
    }

    fn send_error_response(&mut self, ctx: &mut ws::WebsocketContext<Self>, error_message: &str) {
        let error_response = WebSocketSettingsMessage {
            msg_type: "error".to_string(),
            data: serde_json::json!({
                "error": error_message,
                "clientId": self.client_id,
                "timestamp": Self::current_timestamp()
            }),
            timestamp: Self::current_timestamp(),
            compression: None,
            checksum: None,
        };

        self.send_reliable_message(ctx, &error_response);
    }
}
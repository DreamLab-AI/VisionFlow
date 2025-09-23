//! Voice command tag management system
//!
//! This module implements a tagging system that allows voice commands to be tracked
//! through the hive mind/agent system and responses to be routed back to TTS.
//!
//! The system works as follows:
//! 1. User speaks command → STT → Generate unique tag
//! 2. Command + tag → Sent to hive mind/agents
//! 3. Agents process and respond with the same tag
//! 4. Tagged response → Routed back to TTS → User hears response

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use serde::{Deserialize, Serialize};
use log::{info, debug, warn, error};
use uuid::Uuid;

use crate::types::speech::SpeechOptions;
use crate::actors::voice_commands::{VoiceCommand, SwarmVoiceResponse};

/// Unique tag for tracking voice commands through the system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoiceTag {
    /// Unique identifier for this voice command
    pub tag_id: String,
    /// Session ID for context
    pub session_id: String,
    /// Timestamp when tag was created
    pub created_at: DateTime<Utc>,
}

impl VoiceTag {
    /// Generate a new unique voice tag
    pub fn new(session_id: String) -> Self {
        Self {
            tag_id: format!("voice_tag_{}", Uuid::new_v4()),
            session_id,
            created_at: Utc::now(),
        }
    }

    /// Get a short representation for logging
    pub fn short_id(&self) -> String {
        self.tag_id.chars().take(12).collect()
    }
}

/// Tagged voice command with routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggedVoiceCommand {
    /// The voice command
    pub command: VoiceCommand,
    /// Unique tag for tracking
    pub tag: VoiceTag,
    /// Whether response should be spoken via TTS
    pub expect_voice_response: bool,
    /// TTS options for the response
    pub tts_options: SpeechOptions,
    /// Optional timeout for the command
    pub timeout: Option<DateTime<Utc>>,
}

/// Tagged response from agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggedVoiceResponse {
    /// The response from the agent/hive mind
    pub response: SwarmVoiceResponse,
    /// Tag that matches the original command
    pub tag: VoiceTag,
    /// Whether this is the final response
    pub is_final: bool,
    /// Timestamp of the response
    pub responded_at: DateTime<Utc>,
}

/// Status of a tagged voice command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaggedCommandStatus {
    /// Command sent to agents, waiting for response
    Pending,
    /// Response received, processing for TTS
    Processing,
    /// Response sent to TTS
    Completed,
    /// Command timed out without response
    TimedOut,
    /// Error occurred during processing
    Failed(String),
}

/// Active voice command being tracked
#[derive(Debug, Clone)]
struct ActiveVoiceCommand {
    /// The tagged command
    command: TaggedVoiceCommand,
    /// Current status
    status: TaggedCommandStatus,
    /// Number of partial responses received
    response_count: usize,
    /// Last activity timestamp
    last_activity: DateTime<Utc>,
}

/// Voice tag manager for tracking commands and routing responses
pub struct VoiceTagManager {
    /// Active commands being tracked by tag
    active_commands: Arc<RwLock<HashMap<String, ActiveVoiceCommand>>>,
    /// Channel for sending tagged responses to TTS
    tts_sender: Option<mpsc::Sender<TaggedVoiceResponse>>,
    /// Default timeout for voice commands
    default_timeout: ChronoDuration,
    /// Maximum number of active commands to track
    max_active_commands: usize,
}

impl VoiceTagManager {
    /// Create a new voice tag manager
    pub fn new() -> Self {
        Self {
            active_commands: Arc::new(RwLock::new(HashMap::new())),
            tts_sender: None,
            default_timeout: ChronoDuration::minutes(5), // 5 minute timeout
            max_active_commands: 100, // Track up to 100 concurrent commands
        }
    }

    /// Set the TTS response channel
    pub fn set_tts_sender(&mut self, sender: mpsc::Sender<TaggedVoiceResponse>) {
        self.tts_sender = Some(sender);
        info!("Voice tag manager TTS sender configured");
    }

    /// Generate a tagged voice command with unique tracking
    pub async fn create_tagged_command(
        &self,
        mut command: VoiceCommand,
        expect_voice_response: bool,
        tts_options: SpeechOptions,
        timeout: Option<ChronoDuration>,
    ) -> Result<TaggedVoiceCommand, String> {
        // Generate unique tag
        let tag = VoiceTag::new(command.session_id.clone());

        // Calculate timeout
        let timeout_time = timeout
            .unwrap_or(self.default_timeout)
            .to_std()
            .map(|d| Utc::now() + ChronoDuration::from_std(d).unwrap_or(self.default_timeout))
            .unwrap_or_else(|_| Utc::now() + self.default_timeout);

        // Update command with tag for tracking
        command.session_id = tag.tag_id.clone(); // Use tag as session for tracking

        let tagged_command = TaggedVoiceCommand {
            command,
            tag: tag.clone(),
            expect_voice_response,
            tts_options,
            timeout: Some(timeout_time),
        };

        // Track the command
        let mut active_commands = self.active_commands.write().await;

        // Clean up old commands if at capacity
        if active_commands.len() >= self.max_active_commands {
            self.cleanup_expired_commands_internal(&mut active_commands).await;
        }

        active_commands.insert(tag.tag_id.clone(), ActiveVoiceCommand {
            command: tagged_command.clone(),
            status: TaggedCommandStatus::Pending,
            response_count: 0,
            last_activity: Utc::now(),
        });

        info!("Created tagged voice command {} for session {}",
              tag.short_id(), tag.session_id);

        Ok(tagged_command)
    }

    /// Process a tagged response from agents
    pub async fn process_tagged_response(
        &self,
        mut response: TaggedVoiceResponse,
    ) -> Result<(), String> {
        let tag_id = response.tag.tag_id.clone();
        let mut active_commands = self.active_commands.write().await;

        if let Some(active_cmd) = active_commands.get_mut(&tag_id) {
            // Update activity
            active_cmd.last_activity = Utc::now();
            active_cmd.response_count += 1;

            // Update status
            active_cmd.status = if response.is_final {
                TaggedCommandStatus::Completed
            } else {
                TaggedCommandStatus::Processing
            };

            debug!("Processing tagged response {} (count: {}, final: {})",
                   response.tag.short_id(), active_cmd.response_count, response.is_final);

            // Check if this command expects voice response
            if active_cmd.command.expect_voice_response {
                // Update response timestamp
                response.responded_at = Utc::now();

                // Send to TTS if channel is available
                if let Some(sender) = &self.tts_sender {
                    match sender.try_send(response.clone()) {
                        Ok(_) => {
                            info!("Sent tagged response {} to TTS", response.tag.short_id());
                        }
                        Err(e) => {
                            error!("Failed to send tagged response to TTS: {}", e);
                            active_cmd.status = TaggedCommandStatus::Failed(
                                format!("TTS routing failed: {}", e)
                            );
                            return Err(format!("Failed to route response to TTS: {}", e));
                        }
                    }
                } else {
                    warn!("No TTS sender configured, cannot route voice response");
                    active_cmd.status = TaggedCommandStatus::Failed(
                        "No TTS sender configured".to_string()
                    );
                    return Err("No TTS sender configured".to_string());
                }
            }

            // Clean up completed commands
            if response.is_final {
                active_commands.remove(&tag_id);
                debug!("Cleaned up completed tagged command {}", response.tag.short_id());
            }

            Ok(())
        } else {
            warn!("Received response for unknown tag: {}", response.tag.short_id());
            Err(format!("Unknown tag: {}", tag_id))
        }
    }

    /// Check if a tag is currently active
    pub async fn is_tag_active(&self, tag_id: &str) -> bool {
        let active_commands = self.active_commands.read().await;
        active_commands.contains_key(tag_id)
    }

    /// Get the status of a tagged command
    pub async fn get_command_status(&self, tag_id: &str) -> Option<TaggedCommandStatus> {
        let active_commands = self.active_commands.read().await;
        active_commands.get(tag_id).map(|cmd| cmd.status.clone())
    }

    /// Get all active command tags
    pub async fn get_active_tags(&self) -> Vec<VoiceTag> {
        let active_commands = self.active_commands.read().await;
        active_commands.values()
            .map(|cmd| cmd.command.tag.clone())
            .collect()
    }

    /// Clean up expired and timed out commands
    pub async fn cleanup_expired_commands(&self) {
        let mut active_commands = self.active_commands.write().await;
        self.cleanup_expired_commands_internal(&mut active_commands).await;
    }

    /// Internal cleanup implementation
    async fn cleanup_expired_commands_internal(
        &self,
        active_commands: &mut HashMap<String, ActiveVoiceCommand>,
    ) {
        let now = Utc::now();
        let mut expired_tags = Vec::new();

        for (tag_id, active_cmd) in active_commands.iter_mut() {
            let should_remove = if let Some(timeout) = active_cmd.command.timeout {
                // Command has timed out
                if now > timeout {
                    active_cmd.status = TaggedCommandStatus::TimedOut;
                    true
                } else {
                    false
                }
            } else {
                // No explicit timeout, use default aging
                let age = now.signed_duration_since(active_cmd.last_activity);
                age > self.default_timeout
            };

            if should_remove {
                expired_tags.push(tag_id.clone());
            }
        }

        for tag_id in expired_tags {
            if let Some(cmd) = active_commands.remove(&tag_id) {
                match cmd.status {
                    TaggedCommandStatus::TimedOut => {
                        warn!("Cleaned up timed out voice command {}",
                              cmd.command.tag.short_id());
                    }
                    _ => {
                        debug!("Cleaned up expired voice command {}",
                               cmd.command.tag.short_id());
                    }
                }
            }
        }
    }

    /// Get statistics about active commands
    pub async fn get_stats(&self) -> VoiceTagStats {
        let active_commands = self.active_commands.read().await;

        let mut stats = VoiceTagStats {
            total_active: active_commands.len(),
            pending: 0,
            processing: 0,
            completed: 0,
            failed: 0,
            timed_out: 0,
        };

        for cmd in active_commands.values() {
            match cmd.status {
                TaggedCommandStatus::Pending => stats.pending += 1,
                TaggedCommandStatus::Processing => stats.processing += 1,
                TaggedCommandStatus::Completed => stats.completed += 1,
                TaggedCommandStatus::Failed(_) => stats.failed += 1,
                TaggedCommandStatus::TimedOut => stats.timed_out += 1,
            }
        }

        stats
    }

    /// Create a tagged response from agent output
    pub fn create_tagged_response(
        tag: VoiceTag,
        response_text: String,
        is_final: bool,
        use_voice: bool,
    ) -> TaggedVoiceResponse {
        TaggedVoiceResponse {
            response: SwarmVoiceResponse {
                text: response_text,
                use_voice,
                metadata: None,
                follow_up: None,
                voice_tag: Some(tag.tag_id.clone()),
                is_final: Some(false),
            },
            tag,
            is_final,
            responded_at: Utc::now(),
        }
    }
}

/// Statistics about voice tag manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTagStats {
    pub total_active: usize,
    pub pending: usize,
    pub processing: usize,
    pub completed: usize,
    pub failed: usize,
    pub timed_out: usize,
}

impl Default for VoiceTagManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actors::voice_commands::SwarmIntent;

    #[tokio::test]
    async fn test_tag_generation() {
        let manager = VoiceTagManager::new();

        let command = VoiceCommand {
            raw_text: "spawn researcher agent".to_string(),
            parsed_intent: SwarmIntent::SpawnAgent {
                agent_type: "researcher".to_string(),
                capabilities: vec![],
            },
            context: None,
            respond_via_voice: true,
            session_id: "test_session".to_string(),
        };

        let tagged_cmd = manager.create_tagged_command(
            command,
            true,
            SpeechOptions::default(),
            None,
        ).await.unwrap();

        assert!(!tagged_cmd.tag.tag_id.is_empty());
        assert_eq!(tagged_cmd.tag.session_id, "test_session");
        assert!(tagged_cmd.expect_voice_response);
    }

    #[tokio::test]
    async fn test_response_processing() {
        let manager = VoiceTagManager::new();

        // Create a tagged command
        let command = VoiceCommand {
            raw_text: "list agents".to_string(),
            parsed_intent: SwarmIntent::ListAgents,
            context: None,
            respond_via_voice: true,
            session_id: "test_session".to_string(),
        };

        let tagged_cmd = manager.create_tagged_command(
            command,
            true,
            SpeechOptions::default(),
            None,
        ).await.unwrap();

        // Verify tag is active
        assert!(manager.is_tag_active(&tagged_cmd.tag.tag_id).await);

        // Create a response
        let response = VoiceTagManager::create_tagged_response(
            tagged_cmd.tag.clone(),
            "Active agents: researcher, coder".to_string(),
            true,
            true,
        );

        // Process response (without TTS sender for test)
        let result = manager.process_tagged_response(response).await;

        // Should fail because no TTS sender configured
        assert!(result.is_err());

        // Tag should still be tracked but marked as failed
        let status = manager.get_command_status(&tagged_cmd.tag.tag_id).await;
        assert!(matches!(status, Some(TaggedCommandStatus::Failed(_))));
    }

    #[tokio::test]
    async fn test_cleanup() {
        let manager = VoiceTagManager::new();

        let command = VoiceCommand {
            raw_text: "help".to_string(),
            parsed_intent: SwarmIntent::Help,
            context: None,
            respond_via_voice: true,
            session_id: "test_session".to_string(),
        };

        // Create command with very short timeout
        let tagged_cmd = manager.create_tagged_command(
            command,
            true,
            SpeechOptions::default(),
            Some(ChronoDuration::milliseconds(1)),
        ).await.unwrap();

        // Wait for timeout
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Cleanup
        manager.cleanup_expired_commands().await;

        // Should be cleaned up
        assert!(!manager.is_tag_active(&tagged_cmd.tag.tag_id).await);
    }
}
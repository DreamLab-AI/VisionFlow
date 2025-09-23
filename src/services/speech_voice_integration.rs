//! Voice-to-swarm integration for SpeechService
//! 
//! This module connects the SpeechService with the swarm orchestration system,
//! enabling voice commands to control agents through the Queen (Supervisor).

use actix::prelude::*;
use log::{info, debug, error, warn};
use std::sync::Arc;
use tokio::sync::mpsc;
use crate::services::speech_service::SpeechService;
use crate::services::voice_tag_manager::{VoiceTagManager, TaggedVoiceResponse, VoiceTag};
use crate::actors::supervisor::SupervisorActor;
use crate::actors::voice_commands::{VoiceCommand, SwarmVoiceResponse};
use crate::types::speech::SpeechOptions;

/// Extension trait for SpeechService to add voice-to-swarm capabilities with tagging
pub trait VoiceSwarmIntegration {
    /// Process transcribed text as a voice command for the swarm with tag tracking
    async fn process_voice_command_with_tags(
        &self,
        text: String,
        session_id: String,
        tag_manager: Arc<VoiceTagManager>
    ) -> Result<VoiceTag, String>;

    /// Handle tagged swarm response and convert to speech
    async fn handle_tagged_swarm_response(&self, response: TaggedVoiceResponse) -> Result<(), String>;

    /// Legacy method for backwards compatibility
    async fn process_voice_command(&self, text: String, session_id: String) -> Result<(), String>;

    /// Handle swarm response and convert to speech if needed
    async fn handle_swarm_response(&self, response: SwarmVoiceResponse) -> Result<(), String>;
}

impl VoiceSwarmIntegration for SpeechService {
    async fn process_voice_command_with_tags(
        &self,
        text: String,
        session_id: String,
        tag_manager: Arc<VoiceTagManager>
    ) -> Result<VoiceTag, String> {
        info!("Processing tagged voice command: '{}'", text);

        // Parse the text into a voice command
        match VoiceCommand::parse(&text, session_id.clone()) {
            Ok(mut voice_cmd) => {
                debug!("Parsed voice command: {:?}", voice_cmd.parsed_intent);

                // Create tagged command
                let tagged_cmd = tag_manager.create_tagged_command(
                    voice_cmd.clone(),
                    true, // expect voice response
                    SpeechOptions::default(),
                    None, // use default timeout
                ).await?;

                let tag = tagged_cmd.tag.clone();

                // Update voice command with tag
                voice_cmd.voice_tag = Some(tag.tag_id.clone());

                // Send to SupervisorActor (Queen orchestrator)
                let supervisor = SupervisorActor::new("VoiceIntegrationSupervisor".to_string()).start();

                // Send command and await response
                match supervisor.send(voice_cmd).await {
                    Ok(Ok(mut response)) => {
                        // Add tag to response for routing
                        response.voice_tag = Some(tag.tag_id.clone());
                        response.is_final = Some(true);

                        // Create tagged response
                        let tagged_response = TaggedVoiceResponse {
                            response,
                            tag: tag.clone(),
                            is_final: true,
                            responded_at: chrono::Utc::now(),
                        };

                        // Process the tagged response
                        tag_manager.process_tagged_response(tagged_response).await.map_err(|e| e.to_string())?;

                        Ok(tag)
                    }
                    Ok(Err(e)) => {
                        error!("Swarm processing error: {}", e);

                        // Create error response
                        let error_response = TaggedVoiceResponse {
                            response: SwarmVoiceResponse {
                                text: format!("I encountered an error: {}. Please try again.", e),
                                use_voice: true,
                                metadata: None,
                                follow_up: Some("What would you like me to do instead?".to_string()),
                                voice_tag: Some(tag.tag_id.clone()),
                                is_final: Some(true),
                            },
                            tag: tag.clone(),
                            is_final: true,
                            responded_at: chrono::Utc::now(),
                        };

                        tag_manager.process_tagged_response(error_response).await?;
                        Ok(tag)
                    }
                    Err(e) => {
                        error!("Failed to send command to supervisor: {}", e);
                        Err(format!("Communication error: {}", e))
                    }
                }
            }
            Err(e) => {
                warn!("Failed to parse voice command '{}': {}", text, e);
                Err(format!("Failed to parse command: {}", e))
            }
        }
    }

    async fn handle_tagged_swarm_response(&self, response: TaggedVoiceResponse) -> Result<(), String> {
        info!("Handling tagged swarm response: {} (tag: {})",
              response.response.text, response.tag.short_id());

        // Convert tagged response to TTS
        if response.response.use_voice {
            // Build the full response with follow-up if present
            let full_text = if let Some(follow_up) = response.response.follow_up {
                format!("{} {}", response.response.text, follow_up)
            } else {
                response.response.text.clone()
            };

            // Send to TTS with default options
            let options = SpeechOptions::default();
            self.text_to_speech(full_text, options).await.map_err(|e| e.to_string())?;
        }

        // Broadcast the transcription response as well for UI display
        if let Err(e) = self.get_transcription_sender().send(response.response.text) {
            debug!("Failed to broadcast response text: {}", e);
        }

        Ok(())
    }

    async fn process_voice_command(&self, text: String, session_id: String) -> Result<(), String> {
        info!("Processing voice command: '{}'", text);
        
        // Parse the text into a voice command
        match VoiceCommand::parse(&text, session_id.clone()) {
            Ok(voice_cmd) => {
                debug!("Parsed voice command: {:?}", voice_cmd.parsed_intent);

                // Send to SupervisorActor (Queen orchestrator)
                let supervisor = SupervisorActor::new("VoiceProcessingSupervisor".to_string()).start();
                
                // Send command and await response
                match supervisor.send(voice_cmd).await {
                    Ok(Ok(response)) => {
                        // Handle the swarm response
                        self.handle_swarm_response(response).await
                    }
                    Ok(Err(e)) => {
                        error!("Swarm processing error: {}", e);
                        // Send error message via TTS
                        let error_response = SwarmVoiceResponse {
                            text: format!("I encountered an error: {}. Please try again.", e),
                            use_voice: true,
                            metadata: None,
                            follow_up: Some("What would you like me to do instead?".to_string()),
                            voice_tag: None,
                            is_final: Some(true),
                        };
                        self.handle_swarm_response(error_response).await
                    }
                    Err(e) => {
                        error!("Failed to send command to supervisor: {}", e);
                        Err(format!("Communication error: {}", e))
                    }
                }
            }
            Err(e) => {
                warn!("Failed to parse voice command '{}': {}", text, e);
                
                // Send helpful error via TTS
                let help_response = SwarmVoiceResponse {
                    text: "I didn't understand that command. Try saying something like 'spawn a researcher agent' or 'show status'.".to_string(),
                    use_voice: true,
                    metadata: None,
                    follow_up: Some("What would you like me to help with?".to_string()),
                    voice_tag: None,
                    is_final: Some(true),
                };
                self.handle_swarm_response(help_response).await
            }
        }
    }
    
    async fn handle_swarm_response(&self, response: SwarmVoiceResponse) -> Result<(), String> {
        info!("Handling swarm response: {}", response.text);
        
        // If voice response is requested, send to TTS
        if response.use_voice {
            // Build the full response with follow-up if present
            let full_text = if let Some(follow_up) = response.follow_up {
                format!("{} {}", response.text, follow_up)
            } else {
                response.text.clone()
            };
            
            // Send to TTS with default options
            let options = SpeechOptions::default();
            self.text_to_speech(full_text, options).await.map_err(|e| e.to_string())?;
        }

        // Broadcast the transcription response as well for UI display
        if let Err(e) = self.get_transcription_sender().send(response.text) {
            debug!("Failed to broadcast response text: {}", e);
        }
        
        Ok(())
    }
}

/// Modified process_audio_chunk to integrate voice commands
/// This would be added to the existing SpeechService implementation
impl SpeechService {
    pub async fn process_audio_chunk_with_voice_commands(
        &self,
        audio_data: Vec<u8>,
        session_id: String,
        options: crate::types::speech::TranscriptionOptions,
    ) -> Result<String, String> {
        // Process audio chunk (note: this doesn't return transcription directly)
        self.process_audio_chunk(audio_data).await.map_err(|e| e.to_string())?;

        // For now, return a placeholder message until we can implement proper
        // synchronous transcription or async transcription waiting
        // TODO: Implement proper transcription result waiting
        let placeholder_response = "Audio processing initiated. Transcription will be available via subscription.";

        Ok(placeholder_response.to_string())
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_voice_command() {
        assert!(SpeechService::is_voice_command("spawn a researcher agent"));
        assert!(SpeechService::is_voice_command("show me the status"));
        assert!(SpeechService::is_voice_command("list all agents"));
        assert!(!SpeechService::is_voice_command("hello world"));
        assert!(!SpeechService::is_voice_command("the weather is nice"));
    }
}
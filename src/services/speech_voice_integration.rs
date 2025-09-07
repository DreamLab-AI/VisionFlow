//! Voice-to-swarm integration for SpeechService
//! 
//! This module connects the SpeechService with the swarm orchestration system,
//! enabling voice commands to control agents through the Queen (Supervisor).

use actix::prelude::*;
use log::{info, debug, error, warn};
use crate::services::speech_service::SpeechService;
use crate::actors::supervisor::SupervisorActor;
use crate::actors::voice_commands::{VoiceCommand, SwarmVoiceResponse};
use crate::types::speech::SpeechOptions;

/// Extension trait for SpeechService to add voice-to-swarm capabilities
pub trait VoiceSwarmIntegration {
    /// Process transcribed text as a voice command for the swarm
    async fn process_voice_command(&self, text: String, session_id: String) -> Result<(), String>;
    
    /// Handle swarm response and convert to speech if needed
    async fn handle_swarm_response(&self, response: SwarmVoiceResponse) -> Result<(), String>;
}

impl VoiceSwarmIntegration for SpeechService {
    async fn process_voice_command(&self, text: String, session_id: String) -> Result<(), String> {
        info!("Processing voice command: '{}'", text);
        
        // Parse the text into a voice command
        match VoiceCommand::parse(&text, session_id.clone()) {
            Ok(voice_cmd) => {
                debug!("Parsed voice command: {:?}", voice_cmd.parsed_intent);
                
                // Send to SupervisorActor (Queen orchestrator)
                let supervisor = SupervisorActor::from_registry();
                
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
            self.text_to_speech(full_text, options).await?;
        }
        
        // Broadcast the transcription response as well for UI display
        if let Ok(tx) = self.transcription_tx.try_send(response.text) {
            debug!("Broadcast response text to {} receivers", tx);
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
        // First, get the transcription using existing STT
        let transcription = self.process_audio_chunk(audio_data, options).await?;
        
        // Check if this looks like a voice command (simple heuristic)
        if Self::is_voice_command(&transcription) {
            // Process as voice command
            self.process_voice_command(transcription.clone(), session_id).await?;
        }
        
        Ok(transcription)
    }
    
    /// Simple heuristic to detect if text is a voice command
    fn is_voice_command(text: &str) -> bool {
        let command_keywords = [
            "spawn", "agent", "status", "list", "stop", "add", "remove", 
            "help", "show", "create", "delete", "query", "execute", "run"
        ];
        
        let lower = text.to_lowercase();
        command_keywords.iter().any(|keyword| lower.contains(keyword))
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
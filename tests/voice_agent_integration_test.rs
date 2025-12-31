// Test disabled - references deprecated/removed modules (crate::actors::voice_commands, crate::services::speech_service, crate::services::voice_context_manager)
// Voice command actor and speech services may have been restructured per ADR-001
/*
//! Integration test for voice command to agent execution
//!
//! This test verifies that voice commands properly execute on agent swarms
//! through the MCP task orchestration system.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use uuid::Uuid;

    use crate::actors::voice_commands::{SwarmIntent, VoiceCommand};
    use crate::config::AppFullSettings;
    use crate::services::speech_service::SpeechService;
    use crate::services::voice_context_manager::VoiceContextManager;

    #[tokio::test]
    async fn test_voice_command_to_agent_execution() {
        // Initialize test environment
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let speech_service = SpeechService::new(settings);

        // Test voice commands
        let test_commands = vec![
            "spawn a researcher agent",
            "what's the status of all agents",
            "list all agents",
            "execute task: analyze the data with high priority",
        ];

        for command in test_commands {
            println!("Testing voice command: '{}'", command);

            match speech_service
                .process_voice_command(command.to_string())
                .await
            {
                Ok(response) => {
                    println!("Response: {}", response);
                    assert!(!response.is_empty());
                    assert!(!response.contains("mock") || !response.contains("placeholder"));
                }
                Err(e) => {
                    println!("Error processing command '{}': {}", command, e);
                    // Note: Tests might fail if MCP server is not running, which is expected
                }
            }
        }
    }

    // ... rest of tests ...
}
*/

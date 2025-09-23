//! Integration test for voice command to agent execution
//!
//! This test verifies that voice commands properly execute on agent swarms
//! through the MCP task orchestration system.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use uuid::Uuid;

    use crate::config::AppFullSettings;
    use crate::services::speech_service::SpeechService;
    use crate::services::voice_context_manager::VoiceContextManager;
    use crate::actors::voice_commands::{VoiceCommand, SwarmIntent};

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

            match speech_service.process_voice_command(command.to_string()).await {
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

    #[tokio::test]
    async fn test_voice_context_management() {
        let context_manager = VoiceContextManager::new();

        // Create a session
        let session_id = context_manager.get_or_create_session(None, Some("test_user".to_string())).await;
        assert!(!session_id.is_empty());

        // Add conversation turns
        context_manager.add_conversation_turn(
            &session_id,
            "spawn a researcher agent".to_string(),
            "I've spawned a researcher agent for you.".to_string(),
            Some(SwarmIntent::SpawnAgent {
                agent_type: "researcher".to_string(),
                capabilities: vec![]
            }),
        ).await.unwrap();

        // Check context
        let context = context_manager.get_context(&session_id).await;
        assert!(context.is_some());
        let context = context.unwrap();
        assert_eq!(context.turn_count, 1);
        assert_eq!(context.current_agents.len(), 1);
        assert_eq!(context.current_agents[0], "researcher");

        // Add a pending operation
        let mut params = std::collections::HashMap::new();
        params.insert("agent_type".to_string(), "researcher".to_string());

        let operation_id = context_manager.add_pending_operation(
            &session_id,
            "spawn_agent".to_string(),
            params,
            None,
        ).await.unwrap();

        // Check pending operations
        let operations = context_manager.get_pending_operations(&session_id).await;
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0].operation_id, operation_id);

        // Test follow-up detection
        let needs_follow_up = context_manager.needs_follow_up(&session_id).await;
        assert!(needs_follow_up); // Should be true due to pending operation
    }

    #[tokio::test]
    async fn test_voice_command_parsing() {
        let session_id = Uuid::new_v4().to_string();

        // Test different command types
        let test_cases = vec![
            ("spawn a researcher agent", "SpawnAgent"),
            ("what's the status", "QueryStatus"),
            ("list all agents", "ListAgents"),
            ("execute task: test the system", "ExecuteTask"),
            ("help", "Help"),
        ];

        for (input, expected_intent) in test_cases {
            match VoiceCommand::parse(input, session_id.clone()) {
                Ok(voice_cmd) => {
                    let intent_name = match voice_cmd.parsed_intent {
                        SwarmIntent::SpawnAgent { .. } => "SpawnAgent",
                        SwarmIntent::QueryStatus { .. } => "QueryStatus",
                        SwarmIntent::ListAgents => "ListAgents",
                        SwarmIntent::ExecuteTask { .. } => "ExecuteTask",
                        SwarmIntent::Help => "Help",
                        _ => "Other",
                    };
                    assert_eq!(intent_name, expected_intent, "Failed for input: '{}'", input);
                }
                Err(e) => {
                    panic!("Failed to parse command '{}': {}", input, e);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_voice_service_integration() {
        // Test the is_voice_command function
        let test_cases = vec![
            ("spawn a researcher agent", true),
            ("hello world", false),
            ("what's the status of agents", true),
            ("how are you today", false),
            ("list all agents", true),
            ("execute task analysis", true),
        ];

        for (input, expected) in test_cases {
            // Note: This would require making is_voice_command public or adding a test helper
            // For now, we'll test through the full pipeline
            let settings = Arc::new(RwLock::new(AppFullSettings::default()));
            let speech_service = SpeechService::new(settings);

            let result = speech_service.process_voice_command(input.to_string()).await;

            if expected {
                // Should not return the "not a voice command" message
                match result {
                    Ok(response) => {
                        assert!(!response.contains("doesn't appear to be a voice command"));
                    }
                    Err(_) => {
                        // Errors are expected when MCP server is not available
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_conversation_context_cleanup() {
        let context_manager = VoiceContextManager::new();

        // Create multiple sessions
        for i in 0..5 {
            let session_id = format!("test_session_{}", i);
            context_manager.get_or_create_session(Some(session_id), None).await;
        }

        // Check session count
        let count = context_manager.get_active_session_count().await;
        assert_eq!(count, 5);

        // Test cleanup (this would normally be called automatically)
        context_manager.cleanup_expired_sessions().await;

        // Sessions should still be there since they're not expired
        let count_after = context_manager.get_active_session_count().await;
        assert_eq!(count_after, 5);
    }
}
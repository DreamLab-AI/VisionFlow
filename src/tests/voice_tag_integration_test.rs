//! Integration test for voice-to-hive-mind tag system
//!
//! This test demonstrates the complete pipeline:
//! 1. User speaks command → STT → Generate tag
//! 2. Command + tag → Hive mind/agents
//! 3. Agents process and respond with tag
//! 4. Tagged response → TTS → User hears response

use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::services::voice_tag_manager::{VoiceTagManager, TaggedVoiceResponse};
use crate::actors::voice_commands::{VoiceCommand, SwarmVoiceResponse, SwarmIntent};
use crate::types::speech::SpeechOptions;

#[tokio::test]
async fn test_voice_tag_pipeline() {
    // Initialize tag manager
    let mut tag_manager = VoiceTagManager::new();

    // Create TTS response channel to simulate TTS system
    let (tts_tx, mut tts_rx) = mpsc::channel(10);
    tag_manager.set_tts_sender(tts_tx);
    let tag_manager = Arc::new(tag_manager);

    // Simulate user voice command: "spawn researcher agent"
    let session_id = Uuid::new_v4().to_string();
    let voice_command = VoiceCommand {
        raw_text: "spawn researcher agent".to_string(),
        parsed_intent: SwarmIntent::SpawnAgent {
            agent_type: "researcher".to_string(),
            capabilities: vec!["analysis".to_string(), "research".to_string()],
        },
        context: None,
        respond_via_voice: true,
        session_id: session_id.clone(),
        voice_tag: None,
    };

    // Step 1: Create tagged command
    let tagged_cmd = tag_manager.create_tagged_command(
        voice_command,
        true, // expect voice response
        SpeechOptions::default(),
        None, // use default timeout
    ).await.expect("Failed to create tagged command");

    let tag = tagged_cmd.tag.clone();
    println!("Created tagged voice command with tag: {}", tag.short_id());

    // Verify tag is active
    assert!(tag_manager.is_tag_active(&tag.tag_id).await);

    // Step 2: Simulate agent processing and response
    // In real system, this would go through SupervisorActor and MCP agents
    let agent_response = TaggedVoiceResponse {
        response: SwarmVoiceResponse {
            text: "Successfully spawned researcher agent. The agent is ready to analyze data and conduct research.".to_string(),
            use_voice: true,
            metadata: None,
            follow_up: Some("What would you like the researcher to investigate?".to_string()),
            voice_tag: Some(tag.tag_id.clone()),
            is_final: Some(true),
        },
        tag: tag.clone(),
        is_final: true,
        responded_at: chrono::Utc::now(),
    };

    // Step 3: Process the tagged response
    tag_manager.process_tagged_response(agent_response).await
        .expect("Failed to process tagged response");

    // Step 4: Verify response was routed to TTS
    let tts_response = tokio::time::timeout(
        tokio::time::Duration::from_secs(1),
        tts_rx.recv()
    ).await.expect("Timeout waiting for TTS response")
        .expect("No TTS response received");

    // Verify the TTS response contains the correct tag and content
    assert_eq!(tts_response.tag.tag_id, tag.tag_id);
    assert!(tts_response.response.text.contains("Successfully spawned researcher agent"));
    assert!(tts_response.response.use_voice);
    assert!(tts_response.is_final);

    println!("✅ Voice-to-hive-mind tag pipeline test completed successfully!");
    println!("   Tag: {}", tag.short_id());
    println!("   Response: {}", tts_response.response.text);

    // Verify tag is cleaned up after final response
    assert!(!tag_manager.is_tag_active(&tag.tag_id).await);
}

#[tokio::test]
async fn test_tag_timeout_cleanup() {
    let mut tag_manager = VoiceTagManager::new();

    // Create TTS channel
    let (tts_tx, _tts_rx) = mpsc::channel(10);
    tag_manager.set_tts_sender(tts_tx);
    let tag_manager = Arc::new(tag_manager);

    // Create command with very short timeout
    let voice_command = VoiceCommand {
        raw_text: "help".to_string(),
        parsed_intent: SwarmIntent::Help,
        context: None,
        respond_via_voice: true,
        session_id: Uuid::new_v4().to_string(),
        voice_tag: None,
    };

    let tagged_cmd = tag_manager.create_tagged_command(
        voice_command,
        true,
        SpeechOptions::default(),
        Some(chrono::Duration::milliseconds(10)), // Very short timeout
    ).await.expect("Failed to create tagged command");

    let tag = tagged_cmd.tag.clone();

    // Verify tag is active
    assert!(tag_manager.is_tag_active(&tag.tag_id).await);

    // Wait for timeout
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Cleanup expired tags
    tag_manager.cleanup_expired_commands().await;

    // Verify tag was cleaned up
    assert!(!tag_manager.is_tag_active(&tag.tag_id).await);

    println!("✅ Tag timeout cleanup test completed successfully!");
}

#[tokio::test]
async fn test_concurrent_voice_commands() {
    let mut tag_manager = VoiceTagManager::new();

    // Create TTS channel
    let (tts_tx, mut tts_rx) = mpsc::channel(100);
    tag_manager.set_tts_sender(tts_tx);
    let tag_manager = Arc::new(tag_manager);

    // Create multiple concurrent voice commands
    let mut tags = Vec::new();
    for i in 0..5 {
        let voice_command = VoiceCommand {
            raw_text: format!("spawn agent {}", i),
            parsed_intent: SwarmIntent::SpawnAgent {
                agent_type: format!("agent_{}", i),
                capabilities: vec![],
            },
            context: None,
            respond_via_voice: true,
            session_id: Uuid::new_v4().to_string(),
            voice_tag: None,
        };

        let tagged_cmd = tag_manager.create_tagged_command(
            voice_command,
            true,
            SpeechOptions::default(),
            None,
        ).await.expect("Failed to create tagged command");

        tags.push(tagged_cmd.tag.clone());
    }

    // Verify all tags are active
    for tag in &tags {
        assert!(tag_manager.is_tag_active(&tag.tag_id).await);
    }

    // Process responses for all commands
    for (i, tag) in tags.iter().enumerate() {
        let response = TaggedVoiceResponse {
            response: SwarmVoiceResponse {
                text: format!("Agent {} spawned successfully", i),
                use_voice: true,
                metadata: None,
                follow_up: None,
                voice_tag: Some(tag.tag_id.clone()),
                is_final: Some(true),
            },
            tag: tag.clone(),
            is_final: true,
            responded_at: chrono::Utc::now(),
        };

        tag_manager.process_tagged_response(response).await
            .expect("Failed to process tagged response");
    }

    // Verify all responses were routed to TTS
    for i in 0..5 {
        let tts_response = tokio::time::timeout(
            tokio::time::Duration::from_secs(1),
            tts_rx.recv()
        ).await.expect("Timeout waiting for TTS response")
            .expect("No TTS response received");

        assert!(tts_response.response.text.contains("spawned successfully"));
        assert!(tts_response.response.use_voice);
        assert!(tts_response.is_final);
    }

    // Verify all tags are cleaned up
    for tag in &tags {
        assert!(!tag_manager.is_tag_active(&tag.tag_id).await);
    }

    println!("✅ Concurrent voice commands test completed successfully!");
}
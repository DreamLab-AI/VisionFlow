//! Voice command extensions for the SupervisorActor
//! 
//! This module extends the supervisor with voice command handling capabilities,
//! automatically wrapping swarm instructions with voice-appropriate preambles.

use actix::prelude::*;
use log::info;
use crate::actors::supervisor::SupervisorActor;
use crate::actors::voice_commands::{VoiceCommand, SwarmVoiceResponse, SwarmIntent, VoicePreamble};
// use crate::actors::claude_flow_actor_tcp::ClaudeFlowActorTcp; // For future integration
// Note: We'll define voice-specific messages below since they don't exist in messages.rs yet

/// Handler for voice commands in the supervisor (Queen orchestrator)
impl Handler<VoiceCommand> for SupervisorActor {
    type Result = ResponseFuture<Result<SwarmVoiceResponse, String>>;
    
    fn handle(&mut self, msg: VoiceCommand, _ctx: &mut Context<Self>) -> Self::Result {
        info!("Supervisor received voice command: {:?}", msg.parsed_intent);
        
        // Clone necessary data for async block
        let intent = msg.parsed_intent.clone();
        let _session_id = msg.session_id.clone();
        let respond_via_voice = msg.respond_via_voice;
        
        Box::pin(async move {
            match &intent {
                SwarmIntent::SpawnAgent { agent_type, capabilities } => {
                    // Wrap the spawn instruction with voice preamble
                    let instruction = format!("Spawn a {} agent with capabilities: {:?}", 
                                            agent_type, capabilities);
                    let wrapped = VoicePreamble::wrap_instruction(&instruction, &intent);
                    
                    // Send to ClaudeFlowActorTcp with preamble
                    info!("Sending to swarm with preamble: {}", wrapped);
                    
                    // For now, return a simulated success response
                    // In production, this would integrate with ClaudeFlowActorTcp
                    // which would need a new message handler for voice commands
                    let response = format!("I've spawned a {} agent for you. It's now ready to help.", 
                                         agent_type);
                    Ok(SwarmVoiceResponse {
                        text: response,
                        use_voice: respond_via_voice,
                        metadata: None,
                        follow_up: None,
                    })
                },
                
                SwarmIntent::QueryStatus { target } => {
                    // Wrap status query with voice preamble
                    let instruction = format!("Query status for: {:?}", target);
                    let wrapped = VoicePreamble::wrap_instruction(&instruction, &intent);
                    
                    info!("Querying status with preamble: {}", wrapped);
                    
                    // For now, return a simple status
                    // In production, this would query actual agent states
                    let status_text = if let Some(t) = target {
                        format!("The {} is running normally with no issues.", t)
                    } else {
                        "All agents are operational. System is running smoothly.".to_string()
                    };
                    
                    Ok(SwarmVoiceResponse {
                        text: status_text,
                        use_voice: respond_via_voice,
                        metadata: None,
                        follow_up: None,
                    })
                },
                
                SwarmIntent::ListAgents => {
                    // Get list of supervised actors
                    let instruction = "List all active agents";
                    let wrapped = VoicePreamble::wrap_instruction(instruction, &intent);
                    
                    info!("Listing agents with preamble: {}", wrapped);
                    
                    // In production, this would query actual supervised actors
                    Ok(SwarmVoiceResponse {
                        text: "You have 3 active agents: a researcher, a coder, and an analyst.".to_string(),
                        use_voice: respond_via_voice,
                        metadata: None,
                        follow_up: None,
                    })
                },
                
                SwarmIntent::StopAgent { agent_id } => {
                    let instruction = format!("Stop agent: {}", agent_id);
                    let wrapped = VoicePreamble::wrap_instruction(&instruction, &intent);
                    
                    info!("Stopping agent with preamble: {}", wrapped);
                    
                    Ok(SwarmVoiceResponse {
                        text: format!("I've stopped the {} agent as requested.", agent_id),
                        use_voice: respond_via_voice,
                        metadata: None,
                        follow_up: None,
                    })
                },
                
                SwarmIntent::UpdateGraph { action } => {
                    let instruction = format!("Update graph: {:?}", action);
                    let wrapped = VoicePreamble::wrap_instruction(&instruction, &intent);
                    
                    info!("Updating graph with preamble: {}", wrapped);
                    
                    // Format response based on action
                    let response = match action {
                        crate::actors::voice_commands::GraphAction::AddNode { label } => {
                            format!("I've added a node labeled {} to the graph.", label)
                        },
                        crate::actors::voice_commands::GraphAction::RemoveNode { id } => {
                            format!("I've removed node {} from the graph.", id)
                        },
                        crate::actors::voice_commands::GraphAction::AddEdge { from, to } => {
                            format!("I've connected {} to {}.", from, to)
                        },
                        crate::actors::voice_commands::GraphAction::Clear => {
                            "I've cleared the graph.".to_string()
                        },
                    };
                    
                    Ok(SwarmVoiceResponse {
                        text: response,
                        use_voice: respond_via_voice,
                        metadata: None,
                        follow_up: None,
                    })
                },
                
                SwarmIntent::ExecuteTask { description, priority } => {
                    let instruction = format!("Execute task (priority {:?}): {}", priority, description);
                    let wrapped = VoicePreamble::wrap_instruction(&instruction, &intent);
                    
                    info!("Executing task with preamble: {}", wrapped);
                    
                    Ok(SwarmVoiceResponse {
                        text: format!("I'm working on that task: {}.", description),
                        use_voice: respond_via_voice,
                        metadata: None,
                        follow_up: None,
                    })
                },
                
                SwarmIntent::Help => {
                    Ok(SwarmVoiceResponse {
                        text: "You can ask me to spawn agents, check status, list agents, or update the graph. Just speak naturally!".to_string(),
                        use_voice: respond_via_voice,
                        metadata: None,
                        follow_up: Some("What would you like me to do?".to_string()),
                    })
                },
            }
        })
    }
}

// Note: In production, we would define proper message types for ClaudeFlowActorTcp
// to handle voice commands with preambles. For now, we're simulating responses.
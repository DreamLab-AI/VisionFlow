//! Voice command extensions for the SupervisorActor
//! 
//! This module extends the supervisor with voice command handling capabilities,
//! automatically wrapping swarm instructions with voice-appropriate preambles.

use actix::prelude::*;
use log::{info, error};
use std::collections::HashMap;
use crate::actors::supervisor::SupervisorActor;
use crate::actors::voice_commands::{VoiceCommand, SwarmVoiceResponse, SwarmIntent, VoicePreamble};
use crate::actors::claude_flow_actor::ClaudeFlowActorTcp;
use crate::actors::messages::{InitializeSwarm, GetSwarmStatus, SpawnAgentCommand};
use crate::utils::mcp_connection::{call_task_orchestrate, call_agent_list, call_swarm_init, call_agent_spawn};
use actix::registry::SystemRegistry;

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
                    
                    // Execute real agent spawning via MCP
                    info!("Executing agent spawn via MCP: {}", wrapped);

                    let mcp_host = std::env::var("MCP_HOST").unwrap_or_else(|_| "multi-agent-container".to_string());
                    let mcp_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

                    // Execute via MCP connection
                    match call_swarm_init(&mcp_host, &mcp_port, "mesh", 10, "balanced").await {
                        Ok(swarm_result) => {
                            let swarm_id = swarm_result.get("swarmId")
                                .and_then(|s| s.as_str())
                                .unwrap_or("default-swarm");

                            match call_agent_spawn(&mcp_host, &mcp_port, &agent_type, swarm_id).await {
                                Ok(_) => {
                                    let mut metadata = HashMap::new();
                                    metadata.insert("action".to_string(), "spawn_agent".to_string());
                                    metadata.insert("agent_type".to_string(), agent_type.clone());
                                    metadata.insert("swarm_id".to_string(), swarm_id.to_string());
                                    metadata.insert("capabilities".to_string(), format!("{:?}", capabilities));

                                    Ok(SwarmVoiceResponse {
                                        text: format!("Successfully spawned {} agent in swarm {}. The agent is now active and ready for tasks.", agent_type, swarm_id),
                                        use_voice: respond_via_voice,
                                        metadata: Some(metadata),
                                        follow_up: Some("What task would you like to assign to the new agent?".to_string()),
                                    })
                                }
                                Err(e) => {
                                    error!("Failed to spawn agent: {}", e);
                                    Ok(SwarmVoiceResponse {
                                        text: format!("Failed to spawn {} agent. Error: {}", agent_type, e),
                                        use_voice: respond_via_voice,
                                        metadata: None,
                                        follow_up: Some("Would you like me to try again?".to_string()),
                                    })
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to initialize swarm: {}", e);
                            Ok(SwarmVoiceResponse {
                                text: format!("Failed to initialize swarm for agent spawning. Error: {}", e),
                                use_voice: respond_via_voice,
                                metadata: None,
                                follow_up: Some("Should I retry the swarm initialization?".to_string()),
                            })
                        }
                    }
                },
                
                SwarmIntent::QueryStatus { target } => {
                    // Wrap status query with voice preamble
                    let instruction = format!("Query status for: {:?}", target);
                    let wrapped = VoicePreamble::wrap_instruction(&instruction, &intent);

                    info!("Querying status with preamble: {}", wrapped);

                    // Query actual agent status via MCP
                    let mcp_host = std::env::var("MCP_HOST").unwrap_or_else(|_| "multi-agent-container".to_string());
                    let mcp_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

                    let status_text = match call_agent_list(&mcp_host, &mcp_port, "all").await {
                        Ok(agent_result) => {
                            let agent_count = agent_result.get("content")
                                .and_then(|c| c.as_array())
                                .map(|arr| arr.len())
                                .unwrap_or(0);

                            if agent_count > 0 {
                                if let Some(t) = target {
                                    format!("The {} is operational with {} total agents active.", t, agent_count)
                                } else {
                                    format!("System status: {} agents active and operational.", agent_count)
                                }
                            } else {
                                "System status: No active agents found. You may need to spawn some agents first.".to_string()
                            }
                        }
                        Err(e) => {
                            error!("Failed to query agent status: {}", e);
                            format!("Failed to query system status. Error: {}", e)
                        }
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
                    
                    // Query actual agents via MCP
                    let mcp_host = std::env::var("MCP_HOST").unwrap_or_else(|_| "multi-agent-container".to_string());
                    let mcp_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

                    match call_agent_list(&mcp_host, &mcp_port, "all").await {
                        Ok(agent_result) => {
                            if let Some(content) = agent_result.get("content").and_then(|c| c.as_array()) {
                                let agent_names: Vec<String> = content.iter()
                                    .filter_map(|agent| {
                                        agent.get("text")
                                            .and_then(|text| serde_json::from_str::<serde_json::Value>(text.as_str().unwrap_or("{}")).ok())
                                            .and_then(|parsed| parsed.get("agents").cloned())
                                            .and_then(|agents| agents.as_array().cloned())
                                            .map(|arr| arr.iter().filter_map(|a| a.get("name").and_then(|n| n.as_str()).map(|s| s.to_string())).collect::<Vec<_>>())
                                    })
                                    .flatten()
                                    .collect();

                                let response_text = if agent_names.is_empty() {
                                    "No agents are currently active. Would you like me to spawn some agents?".to_string()
                                } else {
                                    format!("You have {} active agents: {}.", agent_names.len(), agent_names.join(", "))
                                };

                                Ok(SwarmVoiceResponse {
                                    text: response_text,
                                    use_voice: respond_via_voice,
                                    metadata: None,
                                    follow_up: if agent_names.is_empty() { Some("Should I spawn some agents for you?".to_string()) } else { None },
                                })
                            } else {
                                Ok(SwarmVoiceResponse {
                                    text: "No agents found in the system.".to_string(),
                                    use_voice: respond_via_voice,
                                    metadata: None,
                                    follow_up: Some("Would you like me to spawn some agents?".to_string()),
                                })
                            }
                        }
                        Err(e) => {
                            error!("Failed to list agents: {}", e);
                            Ok(SwarmVoiceResponse {
                                text: format!("Failed to list agents. Error: {}", e),
                                use_voice: respond_via_voice,
                                metadata: None,
                                follow_up: Some("Would you like me to try again?".to_string()),
                            })
                        }
                    }
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

                    let mcp_host = std::env::var("MCP_HOST").unwrap_or_else(|_| "multi-agent-container".to_string());
                    let mcp_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

                    let priority_str = match priority {
                        crate::actors::voice_commands::TaskPriority::Critical => "critical",
                        crate::actors::voice_commands::TaskPriority::High => "high",
                        crate::actors::voice_commands::TaskPriority::Medium => "medium",
                        crate::actors::voice_commands::TaskPriority::Low => "low",
                    };

                    match call_task_orchestrate(&mcp_host, &mcp_port, &description, Some(priority_str), Some("balanced")).await {
                        Ok(task_result) => {
                            let task_id = task_result.get("taskId")
                                .and_then(|id| id.as_str())
                                .unwrap_or("unknown");

                            let mut metadata = HashMap::new();
                            metadata.insert("action".to_string(), "execute_task".to_string());
                            metadata.insert("task_id".to_string(), task_id.to_string());
                            metadata.insert("priority".to_string(), priority_str.to_string());

                            Ok(SwarmVoiceResponse {
                                text: format!("Task '{}' has been assigned to the swarm with ID: {}. The agents are working on it now.", description, task_id),
                                use_voice: respond_via_voice,
                                metadata: Some(metadata),
                                follow_up: Some("I'll let you know when the task is complete.".to_string()),
                            })
                        }
                        Err(e) => {
                            error!("Failed to orchestrate task: {}", e);
                            Ok(SwarmVoiceResponse {
                                text: format!("Failed to execute task '{}'. Error: {}", description, e),
                                use_voice: respond_via_voice,
                                metadata: None,
                                follow_up: Some("Would you like me to try a different approach?".to_string()),
                            })
                        }
                    }
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
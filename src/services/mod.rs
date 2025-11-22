pub mod agent_visualization_processor;
pub mod agent_visualization_protocol;
pub mod bots_client;
pub mod edge_generation;
pub mod file_service;
pub mod github;
pub mod github_sync_service;
pub mod local_file_sync_service;
pub mod local_markdown_sync;
pub mod management_api_client;
pub mod multi_mcp_agent_discovery;
pub mod natural_language_query_service;
pub mod parsers;
pub mod real_mcp_integration_bridge;
pub mod topology_visualization_engine;
// graph_service module removed - functionality moved to GraphServiceActor
pub mod graph_serialization;
pub mod mcp_relay_manager;
pub mod nostr_service;
pub mod owl_validator;

// horned-functional API stabilized in current implementation
// #[cfg(feature = "ontology")]
// pub mod owl_extractor_service; // Deprecated - functionality merged into owl_validator
pub mod perplexity_service;
pub mod ragflow_service;
pub mod schema_service;
pub mod semantic_analyzer;
pub mod semantic_pathfinding_service;
pub mod settings_watcher;
pub mod speech_service;
pub mod speech_voice_integration;
pub mod voice_context_manager;
pub mod voice_tag_manager;
pub mod streaming_sync_service;
pub mod ontology_converter;
pub mod edge_classifier;
pub mod ontology_reasoner;
pub mod ontology_enrichment_service;
pub mod ontology_reasoning_service;
pub mod ontology_pipeline_service;
pub mod pipeline_events;
pub mod ontology_content_analyzer;
pub mod ontology_file_cache;

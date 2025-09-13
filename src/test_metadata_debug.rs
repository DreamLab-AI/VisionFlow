use webxr::services::file_service::FileService;
use webxr::actors::graph_actor::GraphServiceActor;
use webxr::actors::messages::BuildGraphFromMetadata;
use actix::prelude::*;

#[actix_rt::main]
async fn main() {
    println!("Loading metadata...");
    match FileService::load_or_create_metadata() {
        Ok(metadata) => {
            println!("Loaded {} metadata entries", metadata.len());
            
            // Show first 3 entries
            for (i, (key, value)) in metadata.iter().enumerate() {
                if i >= 3 { break; }
                println!("  {}: {} (size: {})", i+1, key, value.file_size);
            }
            
            // Create a graph actor and build from metadata
            let mut actor = GraphServiceActor::new();
            match actor.build_from_metadata(metadata) {
                Ok(()) => {
                    println!("Graph built successfully!");
                    println!("Graph contains {} nodes", actor.get_graph_data().nodes.len());
                }
                Err(e) => {
                    println!("Failed to build graph: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to load metadata: {}", e);
        }
    }
}
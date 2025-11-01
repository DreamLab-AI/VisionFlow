use crate::models::graph::GraphData;
use std::io::{Error, ErrorKind};
use log::warn;

///
///
pub fn check_empty_graph(graph: &GraphData, min_nodes: usize) -> Result<(), Error> {
    
    if graph.nodes.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData,
            "Graph contains no nodes, cannot perform GPU computation on empty graph"));
    }

    
    if graph.nodes.len() < min_nodes {
        warn!("[Empty Graph Check] Graph contains only {} nodes, which is below the recommended minimum of {}.
              This may cause instability in GPU computation.", graph.nodes.len(), min_nodes);
    }

    Ok(())
}
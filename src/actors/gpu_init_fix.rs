// GPU Initialization Fix Module
// This module ensures GPU is properly initialized for physics simulation

use actix::prelude::*;
use log::{info, warn, error};
use crate::actors::{GPUComputeActor, GraphServiceActor};
use crate::actors::messages::{InitializeGPU, UpdateGPUGraphData};
use crate::models::graph::GraphData;
use std::time::Duration;

/// Retry GPU initialization if it fails
pub async fn ensure_gpu_initialized(
    gpu_addr: &Addr<GPUComputeActor>,
    graph_data: &GraphData,
    max_retries: u32
) -> Result<(), String> {
    for attempt in 1..=max_retries {
        info!("GPU initialization attempt {}/{}", attempt, max_retries);
        
        match gpu_addr.send(InitializeGPU {
            graph: graph_data.clone()
        }).await {
            Ok(Ok(())) => {
                info!("GPU successfully initialized on attempt {}", attempt);
                return Ok(());
            }
            Ok(Err(e)) => {
                warn!("GPU initialization failed on attempt {}: {}", attempt, e);
                if attempt < max_retries {
                    tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
                }
            }
            Err(e) => {
                error!("Failed to send GPU initialization message: {}", e);
                if attempt < max_retries {
                    tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
                }
            }
        }
    }
    
    Err("Failed to initialize GPU after all retries".to_string())
}

/// Verify GPU is ready for physics simulation
pub async fn verify_gpu_ready(gpu_addr: &Addr<GPUComputeActor>) -> bool {
    // Send a test update to see if GPU responds
    let test_graph = GraphData::default();
    
    match gpu_addr.send(UpdateGPUGraphData { graph: test_graph }).await {
        Ok(Ok(())) => {
            info!("GPU verified as ready for physics simulation");
            true
        }
        Ok(Err(e)) => {
            error!("GPU not ready: {}", e);
            false
        }
        Err(e) => {
            error!("Cannot communicate with GPU actor: {}", e);
            false
        }
    }
}
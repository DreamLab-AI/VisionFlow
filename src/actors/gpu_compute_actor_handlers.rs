use actix::prelude::*;
use crate::actors::gpu_compute_actor::{GPUComputeActor, ComputeMode};
use crate::actors::messages::{UpdateVisualAnalyticsParams, SetComputeMode};
use crate::gpu::visual_analytics::VisualAnalyticsParams;
use crate::utils::unified_gpu_compute;
use futures::future::{FutureExt, ready};

// Handler for UpdateVisualAnalyticsParams
impl Handler<UpdateVisualAnalyticsParams> for GPUComputeActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, _msg: UpdateVisualAnalyticsParams, _ctx: &mut Self::Context) -> Self::Result {
        Box::pin(ready(Ok(())).into_actor(self))
    }
}

// Handler for SetComputeMode
impl Handler<SetComputeMode> for GPUComputeActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: SetComputeMode, _ctx: &mut Self::Context) -> Self::Result {
        self.compute_mode = msg.mode;
        
        if let Some(ref mut compute) = self.unified_compute {
            let unified_mode = match msg.mode {
                ComputeMode::Basic => unified_gpu_compute::ComputeMode::Basic,
                ComputeMode::DualGraph => unified_gpu_compute::ComputeMode::Basic,
                ComputeMode::Advanced => unified_gpu_compute::ComputeMode::Constraints,
            };
            compute.set_mode(unified_mode);
        }
        
        Box::pin(ready(Ok(())).into_actor(self))
    }
}
//! Thread-safe wrapper for CudaStream
//!
//! CUDA streams are thread-safe at the CUDA level when properly synchronized.
//! This wrapper provides Rust thread safety guarantees.

use std::sync::Arc;
use cudarc::driver::CudaStream;

/// Thread-safe wrapper for CudaStream
///
/// # Safety
///
/// CUDA streams are thread-safe at the driver level. Multiple threads can submit
/// work to the same stream, and CUDA will serialize the operations properly.
/// The raw pointer is safe to send between threads because:
///
/// 1. CUDA maintains internal synchronization for stream operations
/// 2. We always access through Arc<Mutex> in SharedGPUContext
/// 3. The stream lifetime is managed by Arc reference counting
pub struct SafeCudaStream {
    inner: CudaStream,
}

impl SafeCudaStream {
    pub fn new(stream: CudaStream) -> Self {
        Self { inner: stream }
    }

    pub fn inner(&self) -> &CudaStream {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut CudaStream {
        &mut self.inner
    }

    pub fn into_inner(self) -> CudaStream {
        self.inner
    }
}

// SAFETY: CUDA streams are thread-safe at the driver level.
// The CUDA driver handles synchronization internally.
unsafe impl Send for SafeCudaStream {}
unsafe impl Sync for SafeCudaStream {}

/// Alternative: Wrapper that keeps stream in an Arc
pub struct ArcCudaStream {
    inner: Arc<CudaStream>,
}

impl ArcCudaStream {
    pub fn new(stream: CudaStream) -> Self {
        Self {
            inner: Arc::new(stream),
        }
    }

    pub fn clone_ref(&self) -> Arc<CudaStream> {
        Arc::clone(&self.inner)
    }
}

// SAFETY: CUDA streams are thread-safe at the driver level
unsafe impl Send for ArcCudaStream {}
unsafe impl Sync for ArcCudaStream {}
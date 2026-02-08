//! Thread-safe wrapper for CudaStream
//!
//! CUDA streams are thread-safe at the CUDA level when properly synchronized.
//! This wrapper provides Rust thread safety guarantees.

use cudarc::driver::CudaStream;

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

    pub fn into_inner(self) -> CudaStream {
        self.inner
    }
}

// SAFETY: SafeCudaStream implements Send and Sync manually because CudaStream
// wraps a raw CUDA stream handle (cudaStream_t) which is a pointer type that
// doesn't auto-implement Send/Sync in Rust, but IS thread-safe at the CUDA
// driver level.
//
// Constraints that make this safe:
// 1. CUDA streams are thread-safe: multiple threads can enqueue work on the
//    same stream; the CUDA driver serializes operations within a stream.
// 2. We only expose `&CudaStream` via `inner()`, which prevents aliased
//    mutable access. The previous `inner_mut()` method was removed because
//    it could enable data races: two threads holding &mut CudaStream
//    simultaneously would violate Rust's aliasing rules even though CUDA
//    itself would handle the serialization.
// 3. Ownership transfer is available via `into_inner()` which consumes self,
//    preventing use-after-move.
// 4. The wrapper does not implement Clone, so there is exactly one owner
//    of the underlying stream handle at any time.
unsafe impl Send for SafeCudaStream {}
unsafe impl Sync for SafeCudaStream {}

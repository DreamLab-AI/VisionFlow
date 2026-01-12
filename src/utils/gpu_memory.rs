//! # DEPRECATED: Use `crate::gpu::memory_manager` instead
//!
//! This module is deprecated in favor of the unified `GpuMemoryManager`.
//! The new manager provides:
//! - All functionality from this module (leak detection, tracking)
//! - Dynamic buffer resizing
//! - Async transfers with double buffering
//! - Better performance and safety
//!
//! **Migration Guide**: See `/home/devuser/workspace/project/docs/gpu_memory_consolidation_analysis.md`
//!
//! This module will be removed in a future release.

#![deprecated(
    since = "0.1.0",
    note = "Use crate::gpu::memory_manager::GpuMemoryManager instead"
)]

use cust::memory::DeviceBuffer;
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;

pub struct ManagedDeviceBuffer<T: cust_core::DeviceCopy> {
    buffer: DeviceBuffer<T>,
    name: String,
    size_bytes: usize,
}

impl<T: cust_core::DeviceCopy> ManagedDeviceBuffer<T> {
    pub fn new(buffer: DeviceBuffer<T>, name: String, element_count: usize) -> Self {
        let size_bytes = element_count * std::mem::size_of::<T>();
        debug!("Allocated GPU buffer '{}': {} bytes", name, size_bytes);
        GPU_MEMORY_TRACKER.track_allocation(name.clone(), size_bytes);

        Self {
            buffer,
            name,
            size_bytes,
        }
    }

    pub fn as_device_buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }

    pub fn as_device_buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buffer
    }
}

impl<T: cust_core::DeviceCopy> Drop for ManagedDeviceBuffer<T> {
    fn drop(&mut self) {
        debug!(
            "Freeing GPU buffer '{}': {} bytes",
            self.name, self.size_bytes
        );
        GPU_MEMORY_TRACKER.track_deallocation(self.name.clone(), self.size_bytes);
    }
}

struct GPUMemoryTracker {
    allocations: Arc<std::sync::Mutex<HashMap<String, usize>>>,
    total_allocated: Arc<std::sync::atomic::AtomicUsize>,
}

impl GPUMemoryTracker {
    fn new() -> Self {
        Self {
            allocations: Arc::new(std::sync::Mutex::new(HashMap::new())),
            total_allocated: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    fn track_allocation(&self, name: String, size: usize) {
        
        if let Ok(mut alloc_map) = self.allocations.lock() {
            alloc_map.insert(name.clone(), size);
            let total = self
                .total_allocated
                .fetch_add(size, std::sync::atomic::Ordering::Relaxed);
            debug!(
                "GPU Memory: +{} bytes for '{}', total: {} bytes",
                size,
                name,
                total + size
            );
        }
    }

    fn track_deallocation(&self, name: String, size: usize) {
        
        if let Ok(mut alloc_map) = self.allocations.lock() {
            if alloc_map.remove(&name).is_some() {
                let total = self
                    .total_allocated
                    .fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
                debug!(
                    "GPU Memory: -{} bytes for '{}', total: {} bytes",
                    size,
                    name,
                    total - size
                );
            } else {
                warn!("Attempted to free untracked GPU buffer: {}", name);
            }
        }
    }

    pub fn get_memory_usage(&self) -> (usize, HashMap<String, usize>) {
        let allocations = self.allocations.lock().expect("Mutex poisoned");
        let total = self
            .total_allocated
            .load(std::sync::atomic::Ordering::Relaxed);
        (total, allocations.clone())
    }

    pub fn check_leaks(&self) -> Vec<String> {
        let allocations = self.allocations.lock().expect("Mutex poisoned");
        if !allocations.is_empty() {
            let leaks: Vec<String> = allocations.keys().cloned().collect();
            error!(
                "GPU memory leaks detected: {} buffers still allocated",
                leaks.len()
            );
            for (name, size) in allocations.iter() {
                error!("  Leaked buffer '{}': {} bytes", name, size);
            }
            leaks
        } else {
            debug!("No GPU memory leaks detected");
            Vec::new()
        }
    }
}

static GPU_MEMORY_TRACKER: Lazy<GPUMemoryTracker> = Lazy::new(|| GPUMemoryTracker::new());

pub fn create_managed_buffer<T>(
    capacity: usize,
    name: &str,
) -> Result<ManagedDeviceBuffer<T>, cust::error::CudaError>
where
    T: cust_core::DeviceCopy + Default,
{
    let buffer = DeviceBuffer::from_slice(&vec![T::default(); capacity])?;
    Ok(ManagedDeviceBuffer::new(buffer, name.to_string(), capacity))
}

pub fn create_managed_buffer_from_slice<T>(
    data: &[T],
    name: &str,
) -> Result<ManagedDeviceBuffer<T>, cust::error::CudaError>
where
    T: cust_core::DeviceCopy + Clone,
{
    let buffer = DeviceBuffer::from_slice(data)?;
    Ok(ManagedDeviceBuffer::new(
        buffer,
        name.to_string(),
        data.len(),
    ))
}

pub fn check_gpu_memory_leaks() -> Vec<String> {
    GPU_MEMORY_TRACKER.check_leaks()
}

pub fn get_gpu_memory_usage() -> (usize, HashMap<String, usize>) {
    GPU_MEMORY_TRACKER.get_memory_usage()
}

pub struct MultiStreamManager {
    compute_stream: cust::stream::Stream,
    memory_stream: cust::stream::Stream,
    analysis_stream: cust::stream::Stream,
    current_stream: usize,
}

impl MultiStreamManager {
    pub fn new() -> Result<Self, cust::error::CudaError> {
        Ok(Self {
            compute_stream: cust::stream::Stream::new(
                cust::stream::StreamFlags::NON_BLOCKING,
                None,
            )?,
            memory_stream: cust::stream::Stream::new(
                cust::stream::StreamFlags::NON_BLOCKING,
                None,
            )?,
            analysis_stream: cust::stream::Stream::new(
                cust::stream::StreamFlags::NON_BLOCKING,
                None,
            )?,
            current_stream: 0,
        })
    }

    pub fn get_compute_stream(&self) -> &cust::stream::Stream {
        &self.compute_stream
    }

    pub fn get_memory_stream(&self) -> &cust::stream::Stream {
        &self.memory_stream
    }

    pub fn get_analysis_stream(&self) -> &cust::stream::Stream {
        &self.analysis_stream
    }

    
    pub fn get_next_stream(&mut self) -> &cust::stream::Stream {
        let stream = match self.current_stream % 3 {
            0 => &self.compute_stream,
            1 => &self.memory_stream,
            _ => &self.analysis_stream,
        };
        self.current_stream += 1;
        stream
    }

    
    pub fn synchronize_all(&self) -> Result<(), cust::error::CudaError> {
        self.compute_stream.synchronize()?;
        self.memory_stream.synchronize()?;
        self.analysis_stream.synchronize()?;
        Ok(())
    }

    
    pub async fn synchronize_async(&self) -> Result<(), cust::error::CudaError> {
        
        let compute_event = cust::event::Event::new(cust::event::EventFlags::DEFAULT)?;
        let memory_event = cust::event::Event::new(cust::event::EventFlags::DEFAULT)?;
        let analysis_event = cust::event::Event::new(cust::event::EventFlags::DEFAULT)?;

        
        compute_event.record(&self.compute_stream)?;
        memory_event.record(&self.memory_stream)?;
        analysis_event.record(&self.analysis_stream)?;

        
        loop {
            let compute_done = compute_event
                .query()
                .map(|status| status == cust::event::EventStatus::Ready)
                .unwrap_or(false);
            let memory_done = memory_event
                .query()
                .map(|status| status == cust::event::EventStatus::Ready)
                .unwrap_or(false);
            let analysis_done = analysis_event
                .query()
                .map(|status| status == cust::event::EventStatus::Ready)
                .unwrap_or(false);

            if compute_done && memory_done && analysis_done {
                break;
            }

            
            tokio::task::yield_now().await;
        }

        Ok(())
    }
}

use std::sync::RwLock;

pub struct LabelMappingCache {
    cached_mappings: Arc<RwLock<HashMap<Vec<i32>, Vec<i32>>>>,
    cache_hits: Arc<std::sync::atomic::AtomicU64>,
    cache_misses: Arc<std::sync::atomic::AtomicU64>,
}

impl LabelMappingCache {
    pub fn new() -> Self {
        Self {
            cached_mappings: Arc::new(RwLock::new(HashMap::new())),
            cache_hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            cache_misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn get_or_compute_mapping<F>(&self, labels: &[i32], compute_fn: F) -> Vec<i32>
    where
        F: FnOnce(&[i32]) -> Vec<i32>,
    {
        let key = labels.to_vec();

        
        if let Ok(cache) = self.cached_mappings.read() {
            if let Some(cached_result) = cache.get(&key) {
                self.cache_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return cached_result.clone();
            }
        }

        
        self.cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let result = compute_fn(labels);

        if let Ok(mut cache) = self.cached_mappings.write() {
            
            if cache.len() > 1000 {
                cache.clear();
                debug!("Cleared label mapping cache to prevent memory bloat");
            }
            cache.insert(key, result.clone());
        }

        result
    }

    pub fn get_cache_stats(&self) -> (u64, u64, f64) {
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };
        (hits, misses, hit_rate)
    }
}

//! Comprehensive test suite for unified GPU memory manager
//!
//! Tests cover:
//! - Basic allocation/deallocation
//! - Dynamic resizing with data preservation
//! - Async transfers (double buffering)
//! - Memory leak detection
//! - Concurrent access
//! - Error handling
//! - Performance characteristics
//!
//! NOTE: These tests are disabled because:
//! 1. Tests use `crate::gpu::memory_manager` which doesn't work from external test crate
//! 2. Should use `webxr::gpu::memory_manager` instead
//! 3. The GPU memory manager module may also be private or feature-gated
//!
//! To re-enable:
//! 1. Replace `use crate::gpu::memory_manager` with `use webxr::gpu::memory_manager`
//! 2. Ensure the gpu::memory_manager module is publicly exported
//! 3. Uncomment the code below

/*
#[cfg(all(test, feature = "gpu"))]
mod gpu_memory_tests {
    use crate::gpu::memory_manager::{BufferConfig, GpuMemoryManager};
    use std::sync::{Arc, Mutex};
    use std::thread;

    // ========== Basic Allocation Tests ==========

    #[test]
    fn test_create_manager() {
        let manager = GpuMemoryManager::new();
        assert!(manager.is_ok(), "Manager creation should succeed");

        let stats = manager.unwrap().stats();
        assert_eq!(stats.buffer_count, 0);
        assert_eq!(stats.total_allocated_bytes, 0);
    }

    #[test]
    fn test_create_manager_with_limit() {
        let limit = 1024 * 1024; // 1MB
        let manager = GpuMemoryManager::with_limit(limit);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_allocate_buffer() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        let result = manager.allocate::<f32>("test_buffer", 1000, config);
        assert!(result.is_ok(), "Allocation should succeed");

        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 1);
        assert_eq!(stats.allocation_count, 1);
        assert!(stats.total_allocated_bytes > 0);
    }

    #[test]
    fn test_allocate_multiple_buffers() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("buffer1", 100, config.clone()).unwrap();
        manager.allocate::<f32>("buffer2", 200, config.clone()).unwrap();
        manager.allocate::<f32>("buffer3", 300, config).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 3);
        assert_eq!(stats.allocation_count, 3);
    }

    #[test]
    fn test_allocate_duplicate_name() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("duplicate", 100, config.clone()).unwrap();

        // Second allocation with same name should be skipped
        let result = manager.allocate::<f32>("duplicate", 200, config);
        assert!(result.is_ok()); // Should not error, just skip

        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 1); // Still only 1 buffer
    }

    // ========== Deallocation Tests ==========

    #[test]
    fn test_free_buffer() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("to_free", 100, config).unwrap();

        let result = manager.free("to_free");
        assert!(result.is_ok());

        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 0);
    }

    #[test]
    fn test_free_nonexistent_buffer() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let result = manager.free("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_free_multiple_buffers() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("buf1", 100, config.clone()).unwrap();
        manager.allocate::<f32>("buf2", 100, config.clone()).unwrap();
        manager.allocate::<f32>("buf3", 100, config).unwrap();

        manager.free("buf2").unwrap();
        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 2);

        manager.free("buf1").unwrap();
        manager.free("buf3").unwrap();
        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 0);
    }

    // ========== Dynamic Resizing Tests ==========

    #[test]
    fn test_ensure_capacity_no_resize() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("buffer", 1000, config).unwrap();

        // Request smaller capacity - should not resize
        let result = manager.ensure_capacity::<f32>("buffer", 500);
        assert!(result.is_ok());

        let stats = manager.stats();
        assert_eq!(stats.resize_count, 0);
    }

    #[test]
    fn test_ensure_capacity_with_resize() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::for_positions();
        manager.allocate::<f32>("positions", 100, config).unwrap();

        // Request larger capacity - should resize
        let result = manager.ensure_capacity::<f32>("positions", 500);
        assert!(result.is_ok());

        let stats = manager.stats();
        assert!(stats.resize_count > 0);
    }

    #[test]
    fn test_ensure_capacity_growth_factor() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let mut config = BufferConfig::default();
        config.growth_factor = 2.0; // Double each time

        manager.allocate::<f32>("buffer", 100, config).unwrap();
        manager.ensure_capacity::<f32>("buffer", 150).unwrap();

        // After resize with 2.0 growth factor, should have 200 capacity
        let stats = manager.stats();
        assert!(stats.resize_count > 0);
    }

    #[test]
    fn test_ensure_capacity_exceeds_max() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let mut config = BufferConfig::default();
        config.max_size_bytes = 4096; // Small max

        manager.allocate::<f32>("buffer", 100, config).unwrap();

        // Request capacity that exceeds max
        let result = manager.ensure_capacity::<f32>("buffer", 10_000_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_resize_preserves_data() {
        // This test requires actual GPU operations
        // For now, we verify the resize operation succeeds
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::for_velocities();
        manager.allocate::<f32>("velocities", 100, config).unwrap();

        // Resize larger
        let result = manager.ensure_capacity::<f32>("velocities", 500);
        assert!(result.is_ok());

        // Resize smaller (should not actually shrink)
        let result = manager.ensure_capacity::<f32>("velocities", 200);
        assert!(result.is_ok());
    }

    // ========== Buffer Access Tests ==========

    #[test]
    fn test_get_buffer() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("buffer", 100, config).unwrap();

        let buffer = manager.get_buffer::<f32>("buffer");
        assert!(buffer.is_ok());
    }

    #[test]
    fn test_get_buffer_mut() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("buffer", 100, config).unwrap();

        let buffer_mut = manager.get_buffer_mut::<f32>("buffer");
        assert!(buffer_mut.is_ok());
    }

    #[test]
    fn test_get_nonexistent_buffer() {
        let manager = GpuMemoryManager::new().unwrap();

        let buffer = manager.get_buffer::<f32>("nonexistent");
        assert!(buffer.is_err());
    }

    #[test]
    fn test_get_buffer_wrong_type() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("f32_buffer", 100, config).unwrap();

        // Try to access as i32 - should fail
        let buffer = manager.get_buffer::<i32>("f32_buffer");
        assert!(buffer.is_err());
    }

    // ========== Memory Limit Tests ==========

    #[test]
    fn test_memory_limit_enforcement() {
        let small_limit = 1024; // 1KB
        let mut manager = GpuMemoryManager::with_limit(small_limit).unwrap();

        let config = BufferConfig::default();

        // Allocate within limit
        let result = manager.allocate::<f32>("small", 10, config.clone());
        assert!(result.is_ok());

        // Try to allocate beyond limit
        let result = manager.allocate::<f32>("huge", 1_000_000, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_total_allocated_tracking() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("buf1", 100, config.clone()).unwrap();
        manager.allocate::<f32>("buf2", 200, config).unwrap();

        let stats = manager.stats();
        let expected_bytes = (100 + 200) * std::mem::size_of::<f32>();
        assert!(stats.total_allocated_bytes >= expected_bytes);
    }

    #[test]
    fn test_peak_allocated_tracking() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("buf1", 1000, config.clone()).unwrap();
        manager.allocate::<f32>("buf2", 2000, config.clone()).unwrap();

        let peak_after_alloc = manager.stats().peak_allocated_bytes;

        // Free one buffer
        manager.free("buf2").unwrap();

        let stats = manager.stats();
        // Peak should not decrease
        assert_eq!(stats.peak_allocated_bytes, peak_after_alloc);
        // Current should decrease
        assert!(stats.total_allocated_bytes < peak_after_alloc);
    }

    // ========== Leak Detection Tests ==========

    #[test]
    fn test_no_leaks_when_freed() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("temp", 100, config).unwrap();
        manager.free("temp").unwrap();

        let leaks = manager.check_leaks();
        assert_eq!(leaks.len(), 0);
    }

    #[test]
    fn test_leak_detection() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("leaked", 100, config).unwrap();

        // Don't free it
        let leaks = manager.check_leaks();
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0], "leaked");
    }

    #[test]
    fn test_multiple_leaks() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("leak1", 100, config.clone()).unwrap();
        manager.allocate::<f32>("leak2", 200, config.clone()).unwrap();
        manager.allocate::<f32>("leak3", 300, config).unwrap();

        let leaks = manager.check_leaks();
        assert_eq!(leaks.len(), 3);
    }

    // ========== Async Transfer Tests ==========

    #[test]
    fn test_async_transfer_disabled() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let mut config = BufferConfig::default();
        config.enable_async = false;

        manager.allocate::<f32>("sync_buffer", 100, config).unwrap();

        // Try async download on non-async buffer
        let result = manager.start_async_download::<f32>("sync_buffer");
        assert!(result.is_err());
    }

    #[test]
    fn test_async_transfer_enabled() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let mut config = BufferConfig::default();
        config.enable_async = true;

        manager.allocate::<f32>("async_buffer", 100, config).unwrap();

        // Start async download
        let result = manager.start_async_download::<f32>("async_buffer");
        assert!(result.is_ok());

        // Wait for completion
        let data = manager.wait_for_download::<f32>("async_buffer");
        assert!(data.is_ok());
        assert_eq!(data.unwrap().len(), 100);
    }

    #[test]
    fn test_async_transfer_double_buffering() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::for_positions();
        manager.allocate::<f32>("positions", 100, config).unwrap();

        // First transfer
        manager.start_async_download::<f32>("positions").unwrap();
        let data1 = manager.wait_for_download::<f32>("positions").unwrap();

        // Second transfer (should use alternate buffer)
        manager.start_async_download::<f32>("positions").unwrap();
        let data2 = manager.wait_for_download::<f32>("positions").unwrap();

        assert_eq!(data1.len(), 100);
        assert_eq!(data2.len(), 100);
    }

    // ========== Configuration Tests ==========

    #[test]
    fn test_buffer_config_defaults() {
        let config = BufferConfig::default();
        assert_eq!(config.bytes_per_element, 4);
        assert_eq!(config.growth_factor, 1.5);
        assert_eq!(config.min_size_bytes, 4096);
        assert_eq!(config.max_size_bytes, 1024 * 1024 * 1024);
        assert_eq!(config.enable_async, false);
    }

    #[test]
    fn test_buffer_config_presets() {
        let pos = BufferConfig::for_positions();
        assert_eq!(pos.bytes_per_element, 12);
        assert!(pos.enable_async);

        let vel = BufferConfig::for_velocities();
        assert_eq!(vel.bytes_per_element, 12);
        assert!(vel.enable_async);

        let edges = BufferConfig::for_edges();
        assert_eq!(edges.bytes_per_element, 32);
        assert!(!edges.enable_async);

        let grid = BufferConfig::for_grid_cells();
        assert_eq!(grid.bytes_per_element, 8);
        assert!(!grid.enable_async);
    }

    // ========== Statistics Tests ==========

    #[test]
    fn test_statistics_tracking() {
        let mut manager = GpuMemoryManager::new().unwrap();

        let config = BufferConfig::default();
        manager.allocate::<f32>("buf1", 100, config.clone()).unwrap();
        manager.allocate::<f32>("buf2", 200, config).unwrap();

        manager.ensure_capacity::<f32>("buf1", 500).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 2);
        assert_eq!(stats.allocation_count, 2);
        assert!(stats.resize_count > 0);
        assert!(stats.total_allocated_bytes > 0);
    }

    // ========== Concurrent Access Tests ==========

    #[test]
    fn test_concurrent_allocations() {
        let manager = Arc::new(Mutex::new(GpuMemoryManager::new().unwrap()));
        let mut handles = vec![];

        for i in 0..10 {
            let mgr = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                let mut m = mgr.lock().unwrap();
                let config = BufferConfig::default();
                m.allocate::<f32>(&format!("thread_{}", i), 100, config)
                    .unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let manager = manager.lock().unwrap();
        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 10);
    }

    // ========== Error Handling Tests ==========

    #[test]
    fn test_error_handling_invalid_operations() {
        let mut manager = GpuMemoryManager::new().unwrap();

        // Try to access buffer before allocation
        let result = manager.get_buffer::<f32>("not_allocated");
        assert!(result.is_err());

        // Try to resize buffer before allocation
        let result = manager.ensure_capacity::<f32>("not_allocated", 100);
        assert!(result.is_err());

        // Try to free buffer that doesn't exist
        let result = manager.free("not_allocated");
        assert!(result.is_err());
    }

    #[test]
    fn test_lifecycle_complete() {
        let mut manager = GpuMemoryManager::new().unwrap();

        // Allocate
        let config = BufferConfig::for_positions();
        manager.allocate::<f32>("lifecycle", 100, config).unwrap();

        // Access
        let _buffer = manager.get_buffer::<f32>("lifecycle").unwrap();

        // Resize
        manager.ensure_capacity::<f32>("lifecycle", 500).unwrap();

        // Free
        manager.free("lifecycle").unwrap();

        // Verify cleanup
        let leaks = manager.check_leaks();
        assert_eq!(leaks.len(), 0);
    }
}
*/

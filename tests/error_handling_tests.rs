//! Error Handling Validation Tests
//!
//! Comprehensive testing of the VisionFlow error handling system
//! Tests error propagation, context preservation, and recovery mechanisms

use pretty_assertions::assert_eq;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::test;

use webxr::errors::*;
use webxr::utils::gpu_safety::*;

#[derive(Debug)]
pub struct ErrorHandlingTestSuite {
    test_count: usize,
    passed_tests: usize,
    failed_tests: usize,
}

impl ErrorHandlingTestSuite {
    pub fn new() -> Self {
        Self {
            test_count: 0,
            passed_tests: 0,
            failed_tests: 0,
        }
    }

    pub async fn run_all_tests(&mut self) {
        println!("Running Error Handling Validation Tests...");

        self.test_vision_flow_error_types().await;
        self.test_error_conversion().await;
        self.test_error_context_preservation().await;
        self.test_error_propagation_chains().await;
        self.test_actor_error_scenarios().await;
        self.test_gpu_error_scenarios().await;
        self.test_settings_error_scenarios().await;
        self.test_network_error_scenarios().await;
        self.test_error_recovery_patterns().await;
        self.test_error_logging_integration().await;
        self.test_concurrent_error_handling().await;
        self.test_error_serialization().await;

        self.print_results();
    }

    async fn test_vision_flow_error_types(&mut self) {
        let test_name = "vision_flow_error_types";
        let start = Instant::now();

        // Test all VisionFlowError variants
        let errors = vec![
            VisionFlowError::Actor(ActorError::StartupFailed {
                actor_name: "TestActor".to_string(),
                reason: "Config missing".to_string(),
            }),
            VisionFlowError::GPU(GPUError::DeviceInitializationFailed(
                "CUDA not found".to_string(),
            )),
            VisionFlowError::Settings(SettingsError::FileNotFound("config.yaml".to_string())),
            VisionFlowError::Network(NetworkError::ConnectionFailed {
                host: "localhost".to_string(),
                port: 8080,
                reason: "Refused".to_string(),
            }),
            VisionFlowError::IO(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File missing",
            )),
            VisionFlowError::Serialization("Invalid JSON".to_string()),
            VisionFlowError::Generic {
                message: "Generic error".to_string(),
                source: None,
            },
        ];

        let mut all_passed = true;

        for (i, error) in errors.iter().enumerate() {
            // Test Display implementation
            let display_str = format!("{}", error);
            if display_str.is_empty() {
                eprintln!("Error {} has empty display string", i);
                all_passed = false;
            }

            // Test Debug implementation
            let debug_str = format!("{:?}", error);
            if debug_str.is_empty() {
                eprintln!("Error {} has empty debug string", i);
                all_passed = false;
            }

            // Test error source chain
            let mut current = error as &dyn std::error::Error;
            let mut depth = 0;
            while let Some(source) = current.source() {
                current = source;
                depth += 1;
                if depth > 10 {
                    eprintln!("Error chain too deep for error {}", i);
                    all_passed = false;
                    break;
                }
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_error_conversion(&mut self) {
        let test_name = "error_conversion";
        let start = Instant::now();

        let mut all_passed = true;

        // Test From implementations
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
        let vision_error: VisionFlowError = io_error.into();

        match vision_error {
            VisionFlowError::IO(ref e) => {
                if e.kind() != std::io::ErrorKind::PermissionDenied {
                    eprintln!("IO error kind not preserved during conversion");
                    all_passed = false;
                }
            }
            _ => {
                eprintln!("IO error not converted to VisionFlowError::IO");
                all_passed = false;
            }
        }

        // Test ActorError conversion
        let actor_error = ActorError::RuntimeFailure {
            actor_name: "GraphActor".to_string(),
            reason: "Computation failed".to_string(),
        };
        let vision_error: VisionFlowError = actor_error.into();

        match vision_error {
            VisionFlowError::Actor(ActorError::RuntimeFailure {
                ref actor_name,
                ref reason,
            }) => {
                if actor_name != "GraphActor" || reason != "Computation failed" {
                    eprintln!("Actor error details not preserved during conversion");
                    all_passed = false;
                }
            }
            _ => {
                eprintln!("ActorError not converted properly");
                all_passed = false;
            }
        }

        // Test GPUError conversion
        let gpu_error = GPUError::MemoryAllocationFailed {
            requested_bytes: 1024,
            reason: "Out of VRAM".to_string(),
        };
        let vision_error: VisionFlowError = gpu_error.into();

        match vision_error {
            VisionFlowError::GPU(GPUError::MemoryAllocationFailed {
                requested_bytes,
                ref reason,
            }) => {
                if requested_bytes != 1024 || reason != "Out of VRAM" {
                    eprintln!("GPU error details not preserved during conversion");
                    all_passed = false;
                }
            }
            _ => {
                eprintln!("GPUError not converted properly");
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_error_context_preservation(&mut self) {
        let test_name = "error_context_preservation";
        let start = Instant::now();

        let mut all_passed = true;

        // Test ErrorContext trait
        let io_result: Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Config file missing",
        ));

        let with_context =
            io_result.with_context(|| "Failed to load application configuration".to_string());

        match with_context {
            Err(VisionFlowError::Generic {
                ref message,
                ref source,
            }) => {
                if message != "Failed to load application configuration" {
                    eprintln!("Context message not preserved: {}", message);
                    all_passed = false;
                }

                if source.is_none() {
                    eprintln!("Original error not preserved as source");
                    all_passed = false;
                }
            }
            _ => {
                eprintln!("Error with context not created properly");
                all_passed = false;
            }
        }

        // Test actor context
        let actor_result: Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "Connection lost",
        ));

        let with_actor_context = actor_result.with_actor_context("NetworkActor");

        match with_actor_context {
            Err(VisionFlowError::Actor(ActorError::RuntimeFailure {
                ref actor_name,
                ref reason,
            })) => {
                if actor_name != "NetworkActor" {
                    eprintln!("Actor name not preserved in context: {}", actor_name);
                    all_passed = false;
                }
                if !reason.contains("Connection lost") {
                    eprintln!("Original error reason not preserved: {}", reason);
                    all_passed = false;
                }
            }
            _ => {
                eprintln!("Actor context not applied properly");
                all_passed = false;
            }
        }

        // Test GPU context
        let gpu_result: Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "CUDA driver error",
        ));

        let with_gpu_context = gpu_result.with_gpu_context("force_computation");

        match with_gpu_context {
            Err(VisionFlowError::GPU(GPUError::KernelExecutionFailed {
                ref kernel_name,
                ref reason,
            })) => {
                if kernel_name != "force_computation" {
                    eprintln!("Kernel name not preserved in context: {}", kernel_name);
                    all_passed = false;
                }
                if !reason.contains("CUDA driver error") {
                    eprintln!("Original GPU error reason not preserved: {}", reason);
                    all_passed = false;
                }
            }
            _ => {
                eprintln!("GPU context not applied properly");
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_error_propagation_chains(&mut self) {
        let test_name = "error_propagation_chains";
        let start = Instant::now();

        let mut all_passed = true;

        // Simulate error propagation through multiple layers
        fn layer_3_function() -> Result<String, std::io::Error> {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Database file not found",
            ))
        }

        fn layer_2_function() -> VisionFlowResult<String> {
            layer_3_function().with_context(|| "Failed to read user data".to_string())
        }

        fn layer_1_function() -> VisionFlowResult<String> {
            layer_2_function().with_context(|| "User authentication failed".to_string())
        }

        let result = layer_1_function();

        match result {
            Err(VisionFlowError::Generic {
                ref message,
                ref source,
            }) => {
                if message != "User authentication failed" {
                    eprintln!("Top-level error context not preserved: {}", message);
                    all_passed = false;
                }

                // Check source chain
                if let Some(source) = source {
                    let source_str = format!("{}", source);
                    if !source_str.contains("Failed to read user data") {
                        eprintln!(
                            "Intermediate error context not preserved in source chain: {}",
                            source_str
                        );
                        all_passed = false;
                    }
                } else {
                    eprintln!("Error source chain broken");
                    all_passed = false;
                }
            }
            _ => {
                eprintln!("Error propagation chain not working properly");
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_actor_error_scenarios(&mut self) {
        let test_name = "actor_error_scenarios";
        let start = Instant::now();

        let mut all_passed = true;

        // Test all ActorError variants
        let actor_errors = vec![
            ActorError::StartupFailed {
                actor_name: "GraphActor".to_string(),
                reason: "Configuration validation failed".to_string(),
            },
            ActorError::RuntimeFailure {
                actor_name: "SettingsActor".to_string(),
                reason: "Failed to persist settings".to_string(),
            },
            ActorError::MessageHandlingFailed {
                message_type: "UpdateGraph".to_string(),
                reason: "Invalid graph data format".to_string(),
            },
            ActorError::SupervisionFailed {
                supervisor: "MainSupervisor".to_string(),
                supervised: "GPUActor".to_string(),
                reason: "Child actor crashed repeatedly".to_string(),
            },
            ActorError::MailboxError {
                actor_name: "ClientManagerActor".to_string(),
                reason: "Mailbox overflow - too many pending messages".to_string(),
            },
        ];

        for actor_error in actor_errors {
            // Test error display
            let display_str = format!("{}", actor_error);
            if display_str.is_empty() {
                eprintln!("ActorError has empty display string: {:?}", actor_error);
                all_passed = false;
                continue;
            }

            // Test error contains relevant information
            match &actor_error {
                ActorError::StartupFailed { actor_name, reason } => {
                    if !display_str.contains(actor_name) || !display_str.contains(reason) {
                        eprintln!("StartupFailed error missing actor name or reason");
                        all_passed = false;
                    }
                }
                ActorError::RuntimeFailure { actor_name, reason } => {
                    if !display_str.contains(actor_name) || !display_str.contains(reason) {
                        eprintln!("RuntimeFailure error missing actor name or reason");
                        all_passed = false;
                    }
                }
                ActorError::MessageHandlingFailed {
                    message_type,
                    reason,
                } => {
                    if !display_str.contains(message_type) || !display_str.contains(reason) {
                        eprintln!("MessageHandlingFailed error missing message type or reason");
                        all_passed = false;
                    }
                }
                ActorError::SupervisionFailed {
                    supervisor,
                    supervised,
                    reason,
                } => {
                    if !display_str.contains(supervisor)
                        || !display_str.contains(supervised)
                        || !display_str.contains(reason)
                    {
                        eprintln!(
                            "SupervisionFailed error missing supervisor, supervised, or reason"
                        );
                        all_passed = false;
                    }
                }
                ActorError::MailboxError { actor_name, reason } => {
                    if !display_str.contains(actor_name) || !display_str.contains(reason) {
                        eprintln!("MailboxError error missing actor name or reason");
                        all_passed = false;
                    }
                }
            }

            // Test conversion to VisionFlowError
            let vision_error = VisionFlowError::Actor(actor_error);
            let vision_display = format!("{}", vision_error);
            if !vision_display.contains("Actor Error") {
                eprintln!("VisionFlowError doesn't properly wrap ActorError");
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_gpu_error_scenarios(&mut self) {
        let test_name = "gpu_error_scenarios";
        let start = Instant::now();

        let mut all_passed = true;

        // Test all GPUError variants
        let gpu_errors = vec![
            GPUError::DeviceInitializationFailed("CUDA driver version mismatch".to_string()),
            GPUError::MemoryAllocationFailed {
                requested_bytes: 2147483648, // 2GB
                reason: "Insufficient GPU memory available".to_string(),
            },
            GPUError::KernelExecutionFailed {
                kernel_name: "compute_forces".to_string(),
                reason: "Invalid grid dimensions".to_string(),
            },
            GPUError::DataTransferFailed {
                direction: DataTransferDirection::CPUToGPU,
                reason: "DMA transfer timeout".to_string(),
            },
            GPUError::DataTransferFailed {
                direction: DataTransferDirection::GPUToCPU,
                reason: "Memory alignment error".to_string(),
            },
            GPUError::FallbackToCPU {
                reason: "GPU compute capability insufficient".to_string(),
            },
            GPUError::DriverError("CUDA_ERROR_LAUNCH_FAILED".to_string()),
        ];

        for gpu_error in gpu_errors {
            // Test error display
            let display_str = format!("{}", gpu_error);
            if display_str.is_empty() {
                eprintln!("GPUError has empty display string: {:?}", gpu_error);
                all_passed = false;
                continue;
            }

            // Test error contains relevant information
            match &gpu_error {
                GPUError::DeviceInitializationFailed(reason) => {
                    if !display_str.contains(reason) {
                        eprintln!("DeviceInitializationFailed missing reason: {}", display_str);
                        all_passed = false;
                    }
                }
                GPUError::MemoryAllocationFailed {
                    requested_bytes,
                    reason,
                } => {
                    if !display_str.contains(&requested_bytes.to_string())
                        || !display_str.contains(reason)
                    {
                        eprintln!(
                            "MemoryAllocationFailed missing bytes or reason: {}",
                            display_str
                        );
                        all_passed = false;
                    }
                }
                GPUError::KernelExecutionFailed {
                    kernel_name,
                    reason,
                } => {
                    if !display_str.contains(kernel_name) || !display_str.contains(reason) {
                        eprintln!(
                            "KernelExecutionFailed missing kernel name or reason: {}",
                            display_str
                        );
                        all_passed = false;
                    }
                }
                GPUError::DataTransferFailed { direction, reason } => {
                    let direction_str = format!("{:?}", direction);
                    if !display_str.contains(&direction_str) || !display_str.contains(reason) {
                        eprintln!(
                            "DataTransferFailed missing direction or reason: {}",
                            display_str
                        );
                        all_passed = false;
                    }
                }
                GPUError::FallbackToCPU { reason } => {
                    if !display_str.contains(reason) {
                        eprintln!("FallbackToCPU missing reason: {}", display_str);
                        all_passed = false;
                    }
                }
                GPUError::DriverError(reason) => {
                    if !display_str.contains(reason) {
                        eprintln!("DriverError missing reason: {}", display_str);
                        all_passed = false;
                    }
                }
            }

            // Test conversion to VisionFlowError
            let vision_error = VisionFlowError::GPU(gpu_error);
            let vision_display = format!("{}", vision_error);
            if !vision_display.contains("GPU Error") {
                eprintln!("VisionFlowError doesn't properly wrap GPUError");
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_settings_error_scenarios(&mut self) {
        let test_name = "settings_error_scenarios";
        let start = Instant::now();

        let mut all_passed = true;

        let settings_errors = vec![
            SettingsError::FileNotFound("/config/missing_config.yaml".to_string()),
            SettingsError::ParseError {
                file_path: "/config/invalid.yaml".to_string(),
                reason: "Invalid YAML syntax at line 15".to_string(),
            },
            SettingsError::ValidationFailed {
                setting_path: "physics.spring_constant".to_string(),
                reason: "Value must be between 0.0 and 10.0".to_string(),
            },
            SettingsError::SaveFailed {
                file_path: "/config/settings.yaml".to_string(),
                reason: "Permission denied".to_string(),
            },
            SettingsError::CacheError(
                "Cache corrupted, unable to read cached settings".to_string(),
            ),
        ];

        for settings_error in settings_errors {
            // Test error display
            let display_str = format!("{}", settings_error);
            if display_str.is_empty() {
                eprintln!(
                    "SettingsError has empty display string: {:?}",
                    settings_error
                );
                all_passed = false;
                continue;
            }

            // Test error contains relevant information
            match &settings_error {
                SettingsError::FileNotFound(path) => {
                    if !display_str.contains(path) {
                        eprintln!("FileNotFound missing path: {}", display_str);
                        all_passed = false;
                    }
                }
                SettingsError::ParseError { file_path, reason } => {
                    if !display_str.contains(file_path) || !display_str.contains(reason) {
                        eprintln!("ParseError missing file path or reason: {}", display_str);
                        all_passed = false;
                    }
                }
                SettingsError::ValidationFailed {
                    setting_path,
                    reason,
                } => {
                    if !display_str.contains(setting_path) || !display_str.contains(reason) {
                        eprintln!(
                            "ValidationFailed missing setting path or reason: {}",
                            display_str
                        );
                        all_passed = false;
                    }
                }
                SettingsError::SaveFailed { file_path, reason } => {
                    if !display_str.contains(file_path) || !display_str.contains(reason) {
                        eprintln!("SaveFailed missing file path or reason: {}", display_str);
                        all_passed = false;
                    }
                }
                SettingsError::CacheError(reason) => {
                    if !display_str.contains(reason) {
                        eprintln!("CacheError missing reason: {}", display_str);
                        all_passed = false;
                    }
                }
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_network_error_scenarios(&mut self) {
        let test_name = "network_error_scenarios";
        let start = Instant::now();

        let mut all_passed = true;

        let network_errors = vec![
            NetworkError::ConnectionFailed {
                host: "api.example.com".to_string(),
                port: 443,
                reason: "SSL certificate verification failed".to_string(),
            },
            NetworkError::WebSocketError("Connection closed unexpectedly".to_string()),
            NetworkError::MCPError {
                method: "initialize".to_string(),
                reason: "Invalid client capabilities".to_string(),
            },
            NetworkError::HTTPError {
                url: "https://api.service.com/data".to_string(),
                status: Some(429),
                reason: "Rate limit exceeded".to_string(),
            },
            NetworkError::HTTPError {
                url: "https://unreachable.service.com".to_string(),
                status: None,
                reason: "Network timeout".to_string(),
            },
            NetworkError::Timeout {
                operation: "WebSocket handshake".to_string(),
                timeout_ms: 30000,
            },
        ];

        for network_error in network_errors {
            // Test error display
            let display_str = format!("{}", network_error);
            if display_str.is_empty() {
                eprintln!("NetworkError has empty display string: {:?}", network_error);
                all_passed = false;
                continue;
            }

            // Test error contains relevant information
            match &network_error {
                NetworkError::ConnectionFailed { host, port, reason } => {
                    if !display_str.contains(host)
                        || !display_str.contains(&port.to_string())
                        || !display_str.contains(reason)
                    {
                        eprintln!(
                            "ConnectionFailed missing host, port, or reason: {}",
                            display_str
                        );
                        all_passed = false;
                    }
                }
                NetworkError::WebSocketError(reason) => {
                    if !display_str.contains(reason) {
                        eprintln!("WebSocketError missing reason: {}", display_str);
                        all_passed = false;
                    }
                }
                NetworkError::MCPError { method, reason } => {
                    if !display_str.contains(method) || !display_str.contains(reason) {
                        eprintln!("MCPError missing method or reason: {}", display_str);
                        all_passed = false;
                    }
                }
                NetworkError::HTTPError {
                    url,
                    status,
                    reason,
                } => {
                    if !display_str.contains(url) || !display_str.contains(reason) {
                        eprintln!("HTTPError missing URL or reason: {}", display_str);
                        all_passed = false;
                    }
                    if let Some(status_code) = status {
                        if !display_str.contains(&status_code.to_string()) {
                            eprintln!("HTTPError missing status code: {}", display_str);
                            all_passed = false;
                        }
                    }
                }
                NetworkError::Timeout {
                    operation,
                    timeout_ms,
                } => {
                    if !display_str.contains(operation)
                        || !display_str.contains(&timeout_ms.to_string())
                    {
                        eprintln!("Timeout missing operation or timeout: {}", display_str);
                        all_passed = false;
                    }
                }
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_error_recovery_patterns(&mut self) {
        let test_name = "error_recovery_patterns";
        let start = Instant::now();

        let mut all_passed = true;

        // Test GPU fallback pattern
        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config.clone());

        // Initially should not use CPU fallback
        if validator.should_use_cpu_fallback() {
            eprintln!("Should not start in CPU fallback mode");
            all_passed = false;
        }

        // Record failures to trigger fallback
        for i in 0..safety_config.cpu_fallback_threshold {
            validator.record_failure();

            // Should not trigger fallback until threshold reached
            let should_fallback = validator.should_use_cpu_fallback();
            let expected_fallback = i >= safety_config.cpu_fallback_threshold - 1;

            if should_fallback != expected_fallback {
                eprintln!(
                    "Fallback threshold logic incorrect at failure {}: expected {}, got {}",
                    i, expected_fallback, should_fallback
                );
                all_passed = false;
            }
        }

        // Test recovery
        validator.reset_failures();
        if validator.should_use_cpu_fallback() {
            eprintln!("Should recover from fallback mode after reset");
            all_passed = false;
        }

        // Test partial recovery (adding successes between failures)
        for _ in 0..safety_config.cpu_fallback_threshold - 1 {
            validator.record_failure();
        }

        if validator.should_use_cpu_fallback() {
            eprintln!("Should not trigger fallback just before threshold");
            all_passed = false;
        }

        validator.reset_failures();
        if validator.should_use_cpu_fallback() {
            eprintln!("Should reset failures even before threshold");
            all_passed = false;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_error_logging_integration(&mut self) {
        let test_name = "error_logging_integration";
        let start = Instant::now();

        let mut all_passed = true;

        // Test that errors can be logged without panicking
        let errors = vec![
            VisionFlowError::Actor(ActorError::StartupFailed {
                actor_name: "TestActor".to_string(),
                reason: "Config invalid".to_string(),
            }),
            VisionFlowError::GPU(GPUError::DeviceInitializationFailed(
                "No CUDA devices".to_string(),
            )),
            VisionFlowError::Network(NetworkError::ConnectionFailed {
                host: "localhost".to_string(),
                port: 8080,
                reason: "Connection refused".to_string(),
            }),
        ];

        for (i, error) in errors.iter().enumerate() {
            // Test that error can be formatted for logging
            let log_message = format!("Error {}: {}", i, error);
            if log_message.is_empty() {
                eprintln!("Error {} produces empty log message", i);
                all_passed = false;
            }

            // Test debug formatting
            let debug_message = format!("Error {} debug: {:?}", i, error);
            if debug_message.is_empty() {
                eprintln!("Error {} produces empty debug message", i);
                all_passed = false;
            }

            // Test that we can extract error details for structured logging
            match error {
                VisionFlowError::Actor(actor_err) => match actor_err {
                    ActorError::StartupFailed { actor_name, reason } => {
                        if actor_name.is_empty() || reason.is_empty() {
                            eprintln!("ActorError missing structured data for logging");
                            all_passed = false;
                        }
                    }
                    _ => {}
                },
                VisionFlowError::GPU(gpu_err) => match gpu_err {
                    GPUError::DeviceInitializationFailed(reason) => {
                        if reason.is_empty() {
                            eprintln!("GPUError missing structured data for logging");
                            all_passed = false;
                        }
                    }
                    _ => {}
                },
                VisionFlowError::Network(net_err) => match net_err {
                    NetworkError::ConnectionFailed { host, port, reason } => {
                        if host.is_empty() || *port == 0 || reason.is_empty() {
                            eprintln!("NetworkError missing structured data for logging");
                            all_passed = false;
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_concurrent_error_handling(&mut self) {
        let test_name = "concurrent_error_handling";
        let start = Instant::now();

        let mut all_passed = true;

        // Test thread-safe error handling
        use std::sync::Arc;
        use std::thread;

        let safety_config = GPUSafetyConfig::default();
        let validator = Arc::new(GPUSafetyValidator::new(safety_config));

        let mut handles = vec![];

        // Spawn threads that concurrently record failures and check fallback status
        for thread_id in 0..5 {
            let validator_clone = Arc::clone(&validator);
            let handle = thread::spawn(move || {
                let mut thread_success = true;

                // Each thread records some failures
                for i in 0..3 {
                    validator_clone.record_failure();

                    // Check that fallback status can be read safely
                    let fallback_status = validator_clone.should_use_cpu_fallback();

                    // This is just to verify the call doesn't panic
                    // The actual value depends on timing with other threads
                    let _ = fallback_status;
                }

                // Reset some failures
                if thread_id % 2 == 0 {
                    validator_clone.reset_failures();
                }

                thread_success
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(success) => {
                    if !success {
                        eprintln!("Thread {} reported failure", i);
                        all_passed = false;
                    }
                }
                Err(_) => {
                    eprintln!("Thread {} panicked", i);
                    all_passed = false;
                }
            }
        }

        // Test that the validator is still functional after concurrent access
        let final_status = validator.should_use_cpu_fallback();
        let _ = final_status; // Just ensure it doesn't panic

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_error_serialization(&mut self) {
        let test_name = "error_serialization";
        let start = Instant::now();

        let mut all_passed = true;

        // Test that all error types can be serialized for transmission or storage
        let errors = vec![
            VisionFlowError::Actor(ActorError::RuntimeFailure {
                actor_name: "TestActor".to_string(),
                reason: "Runtime error".to_string(),
            }),
            VisionFlowError::GPU(GPUError::KernelExecutionFailed {
                kernel_name: "test_kernel".to_string(),
                reason: "Execution failed".to_string(),
            }),
            VisionFlowError::Settings(SettingsError::ValidationFailed {
                setting_path: "test.setting".to_string(),
                reason: "Invalid value".to_string(),
            }),
            VisionFlowError::Network(NetworkError::Timeout {
                operation: "test_operation".to_string(),
                timeout_ms: 1000,
            }),
        ];

        for (i, error) in errors.iter().enumerate() {
            // Test JSON-like serialization (using Display)
            let serialized = format!("{}", error);
            if serialized.is_empty() {
                eprintln!("Error {} serializes to empty string", i);
                all_passed = false;
            }

            // Test that serialized form contains key information
            match error {
                VisionFlowError::Actor(ActorError::RuntimeFailure { actor_name, reason }) => {
                    if !serialized.contains(actor_name) || !serialized.contains(reason) {
                        eprintln!("Actor error serialization missing key info: {}", serialized);
                        all_passed = false;
                    }
                }
                VisionFlowError::GPU(GPUError::KernelExecutionFailed {
                    kernel_name,
                    reason,
                }) => {
                    if !serialized.contains(kernel_name) || !serialized.contains(reason) {
                        eprintln!("GPU error serialization missing key info: {}", serialized);
                        all_passed = false;
                    }
                }
                VisionFlowError::Settings(SettingsError::ValidationFailed {
                    setting_path,
                    reason,
                }) => {
                    if !serialized.contains(setting_path) || !serialized.contains(reason) {
                        eprintln!(
                            "Settings error serialization missing key info: {}",
                            serialized
                        );
                        all_passed = false;
                    }
                }
                VisionFlowError::Network(NetworkError::Timeout {
                    operation,
                    timeout_ms,
                }) => {
                    if !serialized.contains(operation)
                        || !serialized.contains(&timeout_ms.to_string())
                    {
                        eprintln!(
                            "Network error serialization missing key info: {}",
                            serialized
                        );
                        all_passed = false;
                    }
                }
                _ => {}
            }

            // Test debug serialization
            let debug_serialized = format!("{:?}", error);
            if debug_serialized.is_empty() {
                eprintln!("Error {} debug serializes to empty string", i);
                all_passed = false;
            }

            // Debug format should contain more detail than display format
            if debug_serialized.len() <= serialized.len() {
                eprintln!(
                    "Error {} debug format should be more detailed than display format",
                    i
                );
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    fn record_test_result(&mut self, test_name: &str, duration: Duration, passed: bool) {
        self.test_count += 1;

        if passed {
            self.passed_tests += 1;
            println!("✓ {} completed in {:.2}ms", test_name, duration.as_millis());
        } else {
            self.failed_tests += 1;
            println!("✗ {} failed after {:.2}ms", test_name, duration.as_millis());
        }
    }

    fn print_results(&self) {
        println!("\n=== Error Handling Test Results ===");
        println!("Total Tests: {}", self.test_count);
        println!("Passed: {}", self.passed_tests);
        println!("Failed: {}", self.failed_tests);
        println!(
            "Success Rate: {:.1}%",
            (self.passed_tests as f64 / self.test_count as f64) * 100.0
        );
    }
}

#[tokio::test]
async fn run_error_handling_validation() {
    let mut test_suite = ErrorHandlingTestSuite::new();
    test_suite.run_all_tests().await;

    // Ensure all tests passed
    assert!(
        test_suite.failed_tests == 0,
        "All error handling tests should pass"
    );
    assert!(
        test_suite.passed_tests > 10,
        "Should have meaningful test coverage"
    );
}

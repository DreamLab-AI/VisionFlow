# Build Configuration Instructions - Cargo.toml

## **Task 1.1: Update Rust Dependencies and Configuration**
*   **Goal:** Upgrade Rust dependencies to latest stable versions, add type generation capabilities, and improve build configuration
*   **Actions:**
    1. In `Cargo.toml`: Update web framework dependencies:
       - Upgrade `actix-web` from "4.9" to "4.11.0"
       - Upgrade `actix-cors` from "0.7" to "0.7.1"
       - Downgrade `tungstenite` from "0.24" to "0.21.0" (compatibility fix)
       - Downgrade `tokio-tungstenite` from "0.24" to "0.21.0" (compatibility fix)
    
    2. Update async runtime dependencies:
       - Upgrade `tokio` from "1.43" to "1.47.1"
       - Upgrade `bytes` from "1.8" to "1.10.1"
    
    3. Update serialization dependencies:
       - Upgrade `serde` from "1.0" to "1.0.219"
       - Add `validator = { version = "0.18", features = ["derive"] }`
    
    4. Add TypeScript type generation support:
       - Add `specta = { version = "2.0.0-rc.22", features = ["derive"] }`
       - Add `specta-typescript = "0.0.9"`
    
    5. Update configuration dependencies:
       - Upgrade `config` from "0.14" to "0.15.15" 
       - Upgrade `dotenvy` from "0.15" to "0.15.7"
       - Upgrade `toml` from "0.8" to "0.9.5"
    
    6. Update logging dependencies:
       - Upgrade `env_logger` from "0.11" to "0.11.8"
    
    7. Update error handling dependencies:
       - Upgrade `thiserror` from "2.0" to "2.0.16"
       - Upgrade `anyhow` from "1.0" to "1.0.99"
    
    8. Update HTTP and Nostr dependencies:
       - Upgrade `reqwest` from "0.12" to "0.12.23"
       - Upgrade `nostr-sdk` from "0.36" to "0.43.0"
    
    9. Update utility dependencies:
       - Add `heck = "0.5"`
       - Upgrade `uuid` from "1.12" to "1.18.0"
       - Upgrade `chrono` from "0.4" to "0.4.41"
       - Upgrade `base64` from "0.22" to "0.22.1"
       - Upgrade `regex` from "1.11" to "1.11.2"
    
    10. Add path validation:
        - Add `path_abs = "0.5"`
    
    11. Update math dependencies:
        - Upgrade `nalgebra` from "0.33" to "0.34.0"
        - Upgrade `glam` from "0.29" to "0.30.5"
        - Upgrade `sysinfo` from "0.32" to "0.37.0"
    
    12. Update parallel processing:
        - Upgrade `rayon` from "1.10" to "1.11.0"
    
    13. Update binary configuration:
        - Change binary name from `test-tcp` to `generate-types`
        - Change binary path from `src/bin/test_tcp_connection.rs` to `src/bin/generate-types.rs`

## **Task 1.2: CUDA Dependencies Configuration**
*   **Goal:** Ensure CUDA dependencies are properly configured for GPU compute features
*   **Actions:**
    1. Verify GPU compute dependencies remain stable:
       - `bytemuck = { version = "1.21", features = ["derive"] }`
       - `cudarc = { version = "0.12.1", features = ["driver"] }` (downgraded for compatibility)
       - `cust = "0.3.2"`
       - `cust_core = "0.1.1"`

## **Implementation Notes:**
- All version updates are for stability and security improvements
- Type generation support enables TypeScript integration
- CUDA version downgrade resolves compilation issues
- Binary rename reflects new TypeScript type generation purpose
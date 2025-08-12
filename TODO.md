Of course. Here is a TODO list that analyzes the current state of your debugging system and provides a clear action plan to centralize control through a `.env` file.

### Current State Analysis

Your project's debugging is controlled by multiple, sometimes conflicting, sources across the frontend, backend, and Docker environments. This makes it difficult to reliably enable or disable debugging features.

*   **Backend (Rust):**
    *   **`RUST_LOG` Environment Variable:** This is set in `docker-compose.dev.yml` (`warn,webxr::...=trace`), `docker-compose.production.yml` (`info`), `docker-compose.yml` (`info`), and `.env_template` (`warn`). This is the standard way to control Rust logging, but it's inconsistent across files.
    *   **`data/settings.yaml`:** This file has a large `system.debug` section with flags like `enabled: true`, `enable_data_debug: true`, and `log_level: debug`. The Rust application reads this file at startup.
    *   **Conflict:** The application initializes logging based on `settings.yaml` (`src/main.rs`), but the `RUST_LOG` environment variable is also set, which can override it. This creates ambiguity about which setting takes precedence.

*   **Frontend (Vite/React):**
    *   **`.env` Variables:** Debugging is well-managed through Vite's environment variables (`VITE_DEBUG`, `VITE_DEBUG_PRESET`, etc.), defined in `client/.env.example`.
    *   **Implementation:** The logic in `client/src/utils/debugConfig.ts` and `console.ts` correctly reads these variables to control frontend logging. This part of the system is working well.

*   **Docker Environment:**
    *   **Inconsistency:** The `RUST_LOG` level varies between `docker-compose.dev.yml`, `docker-compose.production.yml`, and the base `docker-compose.yml`.
    *   **Legacy Variable:** The `.env_template` defines a `DEBUG_MODE=true` variable that does not appear to be used anywhere in the provided Rust code, making it obsolete.

The core problem is the lack of a single source of truth, especially for the backend, leading to confusion and unpredictable behavior.

### Proposed Unified Debugging System

The goal is to make the project's root `.env` file the **single source of truth** for all debugging settings.

1.  **Backend Debug Control:**
    *   Use a single `DEBUG_ENABLED=true|false` variable in `.env` to globally toggle all debug functionalities (e.g., performance overlays, extra API data).
    *   Use the standard `RUST_LOG` variable in `.env` to control the verbosity of backend logs (e.g., `RUST_LOG=info`, `RUST_LOG=debug,webxr=trace`).
    *   Remove all debug-related configuration from `data/settings.yaml`.

2.  **Frontend Debug Control:**
    *   Continue using the `VITE_...` variables.
    *   In the Docker environment, the value of `VITE_DEBUG` will be set based on the backend's `DEBUG_ENABLED` variable for consistency between services.

3.  **Docker Environment:**
    *   All `docker-compose.*.yml` files will inherit `DEBUG_ENABLED` and `RUST_LOG` from the `.env` file, providing a single place to configure behavior for different environments.

---

### TODO List: Aligning the Debugging System

Here is your action plan to refactor the debugging system for better control.

#### ✅ **Task 1: Centralize Debug Configuration in `.env`**

-   [ ] **Modify `.env_template`:** This file will become the blueprint for all debug settings.
    -   Remove the unused `DEBUG_MODE` variable.
    -   Add a new `DEBUG_ENABLED` variable to act as the master switch.
    -   Update `RUST_LOG` with a comprehensive default for development.

    ```diff
    # .env_template

    # Server Configuration
    - RUST_LOG=warn                        # Log level (debug, info, warn, error)
    + # RUST_LOG levels: trace, debug, info, warn, error
    + # Example for development: RUST_LOG=debug,webxr=trace,actix_web=info
    + RUST_LOG=info,webxr=debug
    BIND_ADDRESS=0.0.0.0                 # Server bind address
    - DEBUG_MODE=true                     # When true, only processes Debug Test Page.md
    + DEBUG_ENABLED=true                  # Master switch for all debug features (backend & frontend)
    ```

-   [ ] **Remove Debug Section from `data/settings.yaml`:**
    -   Delete the entire `debug:` block under the `system:` section in `data/settings.yaml`. The application should no longer read debug states from this file.

#### ✅ **Task 2: Refactor Backend to Use Environment Variables**

-   [ ] **Update Logging Initialization:** Modify the Rust code to respect the `RUST_LOG` environment variable. This is the most critical step.
    *   **File:** `src/utils/logging.rs`
    *   **Action:** Replace the `simplelog` setup with `env_logger` or configure `simplelog` to use `RUST_LOG`. `env_logger` is a standard choice.
    *   **Example (`env_logger`):**
        ```rust
        // In src/main.rs or a logging module
        pub fn init_logging() {
            // This will automatically read the RUST_LOG env var
            env_logger::init();
            log::info!("Logging initialized via env_logger");
        }
        ```
    *   **Note:** You will need to add `env_logger` to your `Cargo.toml`.

-   [ ] **Read `DEBUG_ENABLED` in Rust:** Use the new master switch to control debug-only logic.
    *   **File:** `src/main.rs` or `src/config/mod.rs`
    *   **Action:** Read the `DEBUG_ENABLED` environment variable at startup and store it in your `AppState`. This can then be used to conditionally enable features like performance overlays or extra API response data.
    *   **Example:**
        ```rust
        // In your settings/config loading logic
        let debug_enabled = std::env::var("DEBUG_ENABLED")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        // Add it to AppState to make it available throughout the application
        ```

#### ✅ **Task 3: Align Docker Environment Files**

-   [ ] **Update `docker-compose.dev.yml`:**
    -   Remove the hardcoded `RUST_LOG` value and ensure it's inherited from the `.env` file.
    -   Pass `DEBUG_ENABLED` to the Vite frontend as `VITE_DEBUG`.

    ```diff
    # docker-compose.dev.yml
    services:
      webxr:
        # ...
        environment:
          - NVIDIA_VISIBLE_DEVICES=0
    -     - RUST_LOG=warn,webxr::services::claude_flow=trace,webxr=warn,actix_web=warn
    +     # RUST_LOG is now inherited from the .env file via env_file directive
    +     - VITE_DEBUG=${DEBUG_ENABLED:-true} # Pass master debug switch to frontend
          - NODE_ENV=development
          # ...
    ```

-   [ ] **Update `docker-compose.production.yml` and `docker-compose.yml`:**
    -   Ensure these files correctly inherit `RUST_LOG` and `DEBUG_ENABLED` from the `.env` file. The `env_file: .env` directive should handle this, but remove any hardcoded `RUST_LOG` overrides to be sure.

-   [ ] **Update `Dockerfile.dev` and `Dockerfile.production`:**
    -   Change the hardcoded `ENV RUST_LOG=warn` to a default that can be overridden by Docker Compose.
    -   **Example:** `ENV RUST_LOG=${RUST_LOG:-info}`

#### ✅ **Task 4: Final Cleanup and Verification**

-   [ ] **Review Code:** Search the codebase for any remaining debug flags that are not controlled by the new `.env` system and refactor them.
-   [ ] **Update Documentation:** Update `README.md` or other developer guides to explain the new, simplified method for controlling debug settings.
-   [ ] **Test:** Run the application in both development and production modes (`docker-compose -f ... up`) and verify that changing `DEBUG_ENABLED` and `RUST_LOG` in your `.env` file correctly controls the behavior of both the frontend and backend.
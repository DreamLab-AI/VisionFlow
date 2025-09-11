# How to Apply the Physics Settings Fixes

## Understanding the Setup
Your development environment uses Docker with **volume mounts**, which means:
- Source code changes in `/src/` are immediately visible inside the container
- BUT the Rust backend needs to be recompiled for changes to take effect

## ❌ What WON'T Work
```bash
./scripts/launch.sh build
```
This only rebuilds the Docker **image**, not the Rust binary inside the running container.

## ✅ Correct Methods to Apply the Fixes

### Method 1: Rebuild Inside the Container (Recommended for Dev)
```bash
# 1. Enter the running container
docker exec -it visionflow_container /bin/bash

# 2. Inside the container, rebuild the Rust backend
cd /app
cargo build --release --features gpu

# 3. Restart the backend service
supervisorctl restart rust-backend

# 4. Exit the container
exit
```

### Method 2: Full Restart with Rebuild
```bash
# 1. Stop the current environment
./scripts/launch.sh down

# 2. Start with forced rebuild
./scripts/launch.sh -f up
# The -f flag forces a rebuild of the Rust binary during startup
```

### Method 3: Quick Development Cycle
```bash
# If you're actively developing, use this one-liner:
docker exec visionflow_container bash -c "cd /app && cargo build --release --features gpu && supervisorctl restart rust-backend"
```

## Verification After Rebuild

### 1. Check the Logs for Propagation Messages
After making a physics change in the UI, you should see:
```
[INFO] Physics setting changed, propagating to GPU actors
[INFO] Sending UpdateSimulationParams to GraphServiceActor
```

### 2. Run the Test Script
```bash
# From your host machine
docker exec visionflow_container /app/scripts/test-physics-update.sh
```

### 3. Manual Test
1. Open the UI at http://localhost:4000
2. Go to Control Center → Physics Engine tab
3. Change a physics parameter (e.g., repelK)
4. The graph should immediately respond without needing a server restart

## What the Fixes Do

### 1. `settings_handler.rs` Changes
- Added `propagate_physics_to_gpu` calls when physics settings are updated
- This ensures the GraphServiceActor is notified of changes

### 2. `graph_actor.rs` Changes
- Fixed UpdateSimulationParams to update both `simulation_params` AND `target_params`
- Prevents the smooth transition function from reverting values

### 3. Client Changes
- Fixed settings paths to use `visualisation.graphs.[graph].physics.*`
- Added physics settings to ESSENTIAL_PATHS for proper loading

## Current File Status
✅ All code changes have been applied to:
- `/src/handlers/settings_handler.rs`
- `/src/actors/graph_actor.rs`
- `/client/src/store/settingsStore.ts`
- `/client/src/features/control-center/components/tabs/PhysicsEngineTab.tsx`

## Summary
The volume-mounted development setup means your code changes are already in the container, but you need to **recompile the Rust backend** for them to take effect. Use Method 1 or 3 for quick iteration during development.
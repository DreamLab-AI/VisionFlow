# Build Configuration Patches Converted to Instructions

This directory contains all build and configuration file patches from the WebXR project converted into actionable instruction format.

## **Files Processed:**

### **1. Cargo Configuration**
- **File:** `cargo-toml-instructions.md`
- **Source Patches:** `Cargo.toml` dependency updates
- **Key Changes:** 
  - Rust dependency upgrades for stability
  - TypeScript type generation support (specta)
  - CUDA compatibility fixes (cudarc downgrade)
  - Binary rename for type generation

### **2. Docker Build Script**
- **File:** `docker-build-script-instructions.md` 
- **Source Patches:** `docker-build.sh` (new file)
- **Key Changes:**
  - Automated Docker build script creation
  - CUDA architecture configuration
  - Development and production build targets

### **3. Dockerfile Updates**
- **File:** `dockerfile-instructions.md`
- **Source Patches:** `Dockerfile.dev` multi-stage build refactor
- **Key Changes:**
  - Multi-stage build implementation
  - Build optimization and security improvements
  - Removal of complex build.rs script
  - CUDA version compatibility updates

### **4. Package Dependencies**
- **File:** `package-json-instructions.md`
- **Source Patches:** Frontend dependency reorganization
- **Key Changes:**
  - Move dev-only dependencies to devDependencies
  - Add TypeScript type generation scripts
  - Build process integration

### **5. Settings Configuration**
- **File:** `settings-yaml-instructions.md`
- **Source Patches:** `data/settings.yaml` visualization updates
- **Key Changes:**
  - Enable glow effects in WebXR
  - Reorganize controller button mappings
  - Add new view control options

## **Implementation Order:**

1. **Cargo.toml updates** - Foundation dependency changes
2. **Dockerfile.dev refactor** - Build system optimization  
3. **docker-build.sh creation** - Automated build tooling
4. **package.json reorganization** - Frontend dependency cleanup
5. **settings.yaml updates** - Runtime configuration changes

## **Key Benefits:**

- **Performance:** Multi-stage builds reduce image size
- **Stability:** Updated dependencies with security patches
- **Integration:** TypeScript type generation from Rust
- **Compatibility:** CUDA/cudarc version alignment
- **Maintainability:** Simplified build process without complex scripts

## **Dependencies:**

All instructions are designed to work together and should be implemented in the suggested order to avoid conflicts.
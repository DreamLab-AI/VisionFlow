# Unified Build System Implementation Plan

## Executive Summary

This plan details the step-by-step migration from separate build configurations (Vite for client, TypeScript for SDK) to a unified build system that supports both packages efficiently. The migration follows a safe, phased approach to minimize breakage and ensure rollback capability.

## Current State Analysis

### Client (`/client`)
- **Build Tool**: Vite 6.2.6
- **Framework**: React 18.2.0
- **TypeScript**: 5.8.3
- **Key Features**: HMR, Docker support, proxy configuration
- **Output**: `/client/dist`

### SDK (`/sdk/vircadia-world-sdk-ts`)
- **Build Tool**: TypeScript Compiler (tsc)
- **Runtime**: Bun
- **Target**: Browser/Vue exports
- **Output**: `/browser/dist`

### Integration Points
- Shared TypeScript types
- Common dependencies (lodash, three.js ecosystem)
- Potential code sharing between client and SDK

---

## Implementation Phases

---

## PHASE 1: CREATE UNIFIED CONFIGURATION (NON-BREAKING)

**Goal**: Create unified build system alongside existing configuration without breaking current builds.

**Duration**: 2-3 hours
**Risk Level**: LOW (no existing files modified)
**Rollback**: Delete new files

---

### Step 1.1: Create Build Tools Directory Structure

**Commands**:
```bash
# Create unified build configuration directory
mkdir -p /home/devuser/workspace/project/build-config/{scripts,templates,validators}

# Create shared configuration directory
mkdir -p /home/devuser/workspace/project/build-config/shared

# Create validation output directory
mkdir -p /home/devuser/workspace/project/build-config/validation-results
```

**Success Criteria**:
- Directories created successfully
- No impact on existing builds
- `npm run build` still works in both `/client` and `/sdk`

**Validation**:
```bash
# Verify directory structure
ls -la /home/devuser/workspace/project/build-config/
tree /home/devuser/workspace/project/build-config/ -L 2

# Test existing builds still work
cd /home/devuser/workspace/project/client && npm run build
cd /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts && bun run build
```

---

### Step 1.2: Create Shared TypeScript Configuration Base

**File**: `/home/devuser/workspace/project/build-config/shared/tsconfig.base.json`

**Content**:
```json
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "isolatedModules": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "composite": false,
    "lib": ["ES2022"],
    "types": ["node"]
  },
  "exclude": ["node_modules", "dist", "build", "coverage", "**/*.test.ts", "**/*.spec.ts"]
}
```

**Success Criteria**:
- File created with valid JSON
- No syntax errors
- Schema validation passes

**Validation**:
```bash
# Validate JSON syntax
cat /home/devuser/workspace/project/build-config/shared/tsconfig.base.json | jq .

# Check file exists and is readable
test -r /home/devuser/workspace/project/build-config/shared/tsconfig.base.json && echo "✓ Base config readable"
```

---

### Step 1.3: Create Client-Specific TypeScript Configuration

**File**: `/home/devuser/workspace/project/build-config/shared/tsconfig.client.json`

**Content**:
```json
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "baseUrl": ".",
    "paths": {
      "@/*": ["../../client/src/*"],
      "@/components/*": ["../../client/src/components/*"],
      "@/features/*": ["../../client/src/features/*"],
      "@/ui/*": ["../../client/src/ui/*"],
      "@/services/*": ["../../client/src/services/*"],
      "@/utils/*": ["../../client/src/utils/*"],
      "@/types/*": ["../../client/src/types/*"],
      "@/contexts/*": ["../../client/src/contexts/*"],
      "@/store/*": ["../../client/src/store/*"]
    },
    "typeRoots": ["../../client/node_modules/@types"],
    "outDir": "../../client/dist",
    "rootDir": "../../client"
  },
  "include": ["../../client/src/**/*.ts", "../../client/src/**/*.tsx"],
  "exclude": ["../../client/node_modules", "../../client/dist"]
}
```

**Success Criteria**:
- Extends base configuration correctly
- Client-specific paths configured
- React JSX support enabled

**Validation**:
```bash
# Validate JSON and check extends
cat /home/devuser/workspace/project/build-config/shared/tsconfig.client.json | jq '.extends'

# Verify it doesn't break current build
cd /home/devuser/workspace/project/client && npm run build
```

---

### Step 1.4: Create SDK-Specific TypeScript Configuration

**File**: `/home/devuser/workspace/project/build-config/shared/tsconfig.sdk.json`

**Content**:
```json
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "lib": ["ES2022", "DOM"],
    "emitDeclarationOnly": false,
    "outDir": "../../sdk/vircadia-world-sdk-ts/dist",
    "rootDir": "../../sdk/vircadia-world-sdk-ts",
    "tsBuildInfoFile": "../../sdk/vircadia-world-sdk-ts/dist/tsconfig.tsbuildinfo",
    "paths": {
      "@/*": ["../../sdk/vircadia-world-sdk-ts/*"],
      "@vircadia/world-sdk/browser/vue": ["../../sdk/vircadia-world-sdk-ts/browser/src"]
    }
  },
  "include": ["../../sdk/vircadia-world-sdk-ts/**/*.ts"],
  "exclude": ["../../sdk/vircadia-world-sdk-ts/node_modules", "../../sdk/vircadia-world-sdk-ts/dist"]
}
```

**Success Criteria**:
- SDK paths configured correctly
- Browser/Vue exports supported
- Compatible with Bun runtime

**Validation**:
```bash
# Validate configuration
cat /home/devuser/workspace/project/build-config/shared/tsconfig.sdk.json | jq '.compilerOptions.paths'

# Test SDK build still works
cd /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts && bun run build
```

---

### Step 1.5: Create Unified Vite Configuration Factory

**File**: `/home/devuser/workspace/project/build-config/shared/vite.config.factory.ts`

**Content**:
```typescript
import { defineConfig, UserConfig } from 'vite';
import path from 'path';

export interface BuildTarget {
  name: string;
  rootDir: string;
  outDir: string;
  plugins?: any[];
  alias?: Record<string, string>;
  server?: {
    port?: number;
    proxy?: Record<string, any>;
    hmr?: any;
  };
}

export function createViteConfig(target: BuildTarget): UserConfig {
  return defineConfig({
    plugins: target.plugins || [],

    build: {
      outDir: target.outDir,
      emptyOutDir: true,
      sourcemap: true,
      minify: 'esbuild',
      target: 'es2020',
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom'],
          },
        },
      },
    },

    resolve: {
      alias: {
        '@': path.resolve(target.rootDir, './src'),
        ...target.alias,
      },
    },

    server: target.server || {},

    optimizeDeps: {
      include: ['react', 'react-dom'],
    },
  });
}

export function createClientConfig(customizations?: Partial<BuildTarget>): UserConfig {
  return createViteConfig({
    name: 'client',
    rootDir: path.resolve(__dirname, '../../client'),
    outDir: path.resolve(__dirname, '../../client/dist'),
    ...customizations,
  });
}

export function createSdkConfig(customizations?: Partial<BuildTarget>): UserConfig {
  return createViteConfig({
    name: 'sdk',
    rootDir: path.resolve(__dirname, '../../sdk/vircadia-world-sdk-ts'),
    outDir: path.resolve(__dirname, '../../sdk/vircadia-world-sdk-ts/dist'),
    ...customizations,
  });
}
```

**Success Criteria**:
- TypeScript compiles without errors
- Factory functions are properly typed
- Extensible for both client and SDK

**Validation**:
```bash
# Install dependencies for config (if needed)
cd /home/devuser/workspace/project/build-config && npm init -y
cd /home/devuser/workspace/project/build-config && npm install --save-dev vite typescript @types/node

# Validate TypeScript syntax
cd /home/devuser/workspace/project/build-config && npx tsc --noEmit shared/vite.config.factory.ts
```

---

### Step 1.6: Create Build Orchestration Script

**File**: `/home/devuser/workspace/project/build-config/scripts/build-all.ts`

**Content**:
```typescript
#!/usr/bin/env node
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../../');

interface BuildTarget {
  name: string;
  path: string;
  command: string;
  args: string[];
}

const targets: BuildTarget[] = [
  {
    name: 'client',
    path: path.join(projectRoot, 'client'),
    command: 'npm',
    args: ['run', 'build'],
  },
  {
    name: 'sdk',
    path: path.join(projectRoot, 'sdk/vircadia-world-sdk-ts'),
    command: 'bun',
    args: ['run', 'build'],
  },
];

async function buildTarget(target: BuildTarget): Promise<void> {
  console.log(`\n🔨 Building ${target.name}...`);

  return new Promise((resolve, reject) => {
    const proc = spawn(target.command, target.args, {
      cwd: target.path,
      stdio: 'inherit',
      shell: true,
    });

    proc.on('close', (code) => {
      if (code === 0) {
        console.log(`✓ ${target.name} built successfully`);
        resolve();
      } else {
        console.error(`✗ ${target.name} build failed with code ${code}`);
        reject(new Error(`Build failed: ${target.name}`));
      }
    });
  });
}

async function buildAll(parallel: boolean = false): Promise<void> {
  console.log('🚀 Starting unified build process...');
  console.log(`Mode: ${parallel ? 'parallel' : 'sequential'}\n`);

  const startTime = Date.now();

  try {
    if (parallel) {
      await Promise.all(targets.map(buildTarget));
    } else {
      for (const target of targets) {
        await buildTarget(target);
      }
    }

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`\n✅ All builds completed successfully in ${duration}s`);
  } catch (error) {
    console.error('\n❌ Build failed:', error);
    process.exit(1);
  }
}

// Parse CLI arguments
const args = process.argv.slice(2);
const parallel = args.includes('--parallel') || args.includes('-p');
const sequential = args.includes('--sequential') || args.includes('-s');

if (args.includes('--help') || args.includes('-h')) {
  console.log(`
Usage: build-all.ts [options]

Options:
  -p, --parallel      Build all targets in parallel (faster)
  -s, --sequential    Build targets sequentially (default)
  -h, --help         Show this help message
  `);
  process.exit(0);
}

buildAll(parallel && !sequential);
```

**Success Criteria**:
- Script can execute both builds
- Supports parallel and sequential modes
- Proper error handling

**Validation**:
```bash
# Make script executable
chmod +x /home/devuser/workspace/project/build-config/scripts/build-all.ts

# Test help
cd /home/devuser/workspace/project && node build-config/scripts/build-all.ts --help

# Test sequential build (safer for validation)
cd /home/devuser/workspace/project && node build-config/scripts/build-all.ts --sequential
```

---

### Step 1.7: Create Validation Script

**File**: `/home/devuser/workspace/project/build-config/scripts/validate-build.ts`

**Content**:
```typescript
#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../../');

interface ValidationResult {
  target: string;
  passed: boolean;
  checks: {
    name: string;
    passed: boolean;
    message: string;
  }[];
}

interface ValidationTarget {
  name: string;
  distPath: string;
  requiredFiles: string[];
  requiredDirs: string[];
}

const targets: ValidationTarget[] = [
  {
    name: 'client',
    distPath: path.join(projectRoot, 'client/dist'),
    requiredFiles: ['index.html'],
    requiredDirs: ['assets'],
  },
  {
    name: 'sdk',
    distPath: path.join(projectRoot, 'sdk/vircadia-world-sdk-ts/browser/dist'),
    requiredFiles: [],
    requiredDirs: ['browser', 'types'],
  },
];

function checkExists(itemPath: string, type: 'file' | 'dir'): boolean {
  try {
    const stats = fs.statSync(itemPath);
    return type === 'file' ? stats.isFile() : stats.isDirectory();
  } catch {
    return false;
  }
}

function validateTarget(target: ValidationTarget): ValidationResult {
  const result: ValidationResult = {
    target: target.name,
    passed: true,
    checks: [],
  };

  // Check dist directory exists
  const distExists = checkExists(target.distPath, 'dir');
  result.checks.push({
    name: 'Dist directory exists',
    passed: distExists,
    message: distExists ? `✓ ${target.distPath}` : `✗ Missing: ${target.distPath}`,
  });

  if (!distExists) {
    result.passed = false;
    return result;
  }

  // Check required files
  for (const file of target.requiredFiles) {
    const filePath = path.join(target.distPath, file);
    const exists = checkExists(filePath, 'file');
    result.checks.push({
      name: `Required file: ${file}`,
      passed: exists,
      message: exists ? `✓ ${file}` : `✗ Missing: ${file}`,
    });
    if (!exists) result.passed = false;
  }

  // Check required directories
  for (const dir of target.requiredDirs) {
    const dirPath = path.join(target.distPath, dir);
    const exists = checkExists(dirPath, 'dir');
    result.checks.push({
      name: `Required directory: ${dir}`,
      passed: exists,
      message: exists ? `✓ ${dir}/` : `✗ Missing: ${dir}/`,
    });
    if (!exists) result.passed = false;
  }

  return result;
}

function printResults(results: ValidationResult[]): void {
  console.log('\n📋 Build Validation Results\n');

  for (const result of results) {
    const icon = result.passed ? '✅' : '❌';
    console.log(`${icon} ${result.target.toUpperCase()}`);

    for (const check of result.checks) {
      console.log(`  ${check.message}`);
    }
    console.log('');
  }

  const allPassed = results.every(r => r.passed);

  if (allPassed) {
    console.log('✅ All validations passed!\n');
  } else {
    console.log('❌ Some validations failed. Check output above.\n');
    process.exit(1);
  }
}

// Run validation
const results = targets.map(validateTarget);
printResults(results);

// Save results to file
const outputPath = path.join(projectRoot, 'build-config/validation-results/latest.json');
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
console.log(`📄 Results saved to: ${outputPath}\n`);
```

**Success Criteria**:
- Script validates build outputs
- Checks required files and directories
- Generates JSON report

**Validation**:
```bash
# Make executable
chmod +x /home/devuser/workspace/project/build-config/scripts/validate-build.ts

# Run validation after builds complete
cd /home/devuser/workspace/project && node build-config/scripts/validate-build.ts

# Check validation report
cat /home/devuser/workspace/project/build-config/validation-results/latest.json | jq .
```

---

### Step 1.8: Create Root Package.json with Unified Scripts

**File**: `/home/devuser/workspace/project/build-config/package.json`

**Content**:
```json
{
  "name": "visionflow-build-config",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "description": "Unified build configuration for VisionFlow project",
  "scripts": {
    "build": "node scripts/build-all.ts --sequential",
    "build:parallel": "node scripts/build-all.ts --parallel",
    "build:client": "cd ../client && npm run build",
    "build:sdk": "cd ../sdk/vircadia-world-sdk-ts && bun run build",
    "validate": "node scripts/validate-build.ts",
    "test:config": "npm run build && npm run validate",
    "clean": "npm run clean:client && npm run clean:sdk",
    "clean:client": "rm -rf ../client/dist",
    "clean:sdk": "rm -rf ../sdk/vircadia-world-sdk-ts/browser/dist",
    "prebuild": "npm run clean"
  },
  "devDependencies": {
    "@types/node": "^22.14.1",
    "typescript": "^5.8.3",
    "vite": "^6.2.6"
  }
}
```

**Success Criteria**:
- All scripts defined correctly
- Can be executed from build-config directory
- Provides unified interface

**Validation**:
```bash
# Install dependencies
cd /home/devuser/workspace/project/build-config && npm install

# Test scripts
cd /home/devuser/workspace/project/build-config && npm run build:client
cd /home/devuser/workspace/project/build-config && npm run build:sdk
cd /home/devuser/workspace/project/build-config && npm run validate
```

---

### Step 1.9: Phase 1 Validation Checkpoint

**Commands**:
```bash
# Full Phase 1 validation
cd /home/devuser/workspace/project/build-config

# 1. Test unified build (sequential)
npm run build

# 2. Validate outputs
npm run validate

# 3. Verify original builds still work
cd /home/devuser/workspace/project/client && npm run build
cd /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts && bun run build

# 4. Check no files modified in original locations
git status /home/devuser/workspace/project/client/
git status /home/devuser/workspace/project/sdk/
```

**Success Criteria**:
- ✅ Unified build completes successfully
- ✅ Validation passes for all targets
- ✅ Original builds unaffected
- ✅ No modifications to existing config files
- ✅ All new files in `/build-config` directory

**Rollback (if needed)**:
```bash
# Remove entire build-config directory
rm -rf /home/devuser/workspace/project/build-config
```

---

## PHASE 2: TEST UNIFIED SYSTEM (PARALLEL TESTING)

**Goal**: Thoroughly test unified configuration with existing projects to ensure feature parity.

**Duration**: 1-2 hours
**Risk Level**: LOW (still non-breaking)
**Rollback**: Continue using original configs

---

### Step 2.1: Create Test Client Using Unified Config

**File**: `/home/devuser/workspace/project/build-config/test/vite.config.client-test.ts`

**Content**:
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { createClientConfig } from '../shared/vite.config.factory';

export default defineConfig({
  ...createClientConfig({
    plugins: [react()],
    server: {
      port: 5173,
      hmr: {
        clientPort: 3001,
        path: '/vite-hmr',
      },
      proxy: {
        '/api': {
          target: 'http://visionflow_container:4000',
          changeOrigin: true,
          secure: false,
        },
      },
    },
    alias: {
      '@': '/home/devuser/workspace/project/client/src',
    },
  }),
});
```

**Success Criteria**:
- Config extends factory correctly
- Client-specific customizations applied
- Compatible with existing client setup

**Validation**:
```bash
# Test build with unified config
cd /home/devuser/workspace/project/client && npx vite build --config ../build-config/test/vite.config.client-test.ts

# Compare outputs
diff -r /home/devuser/workspace/project/client/dist /home/devuser/workspace/project/client/dist-test 2>/dev/null || echo "Differences found - review required"
```

---

### Step 2.2: Create Test SDK Using Unified Config

**File**: `/home/devuser/workspace/project/build-config/test/tsconfig.sdk-test.json`

**Content**:
```json
{
  "extends": "../shared/tsconfig.sdk.json",
  "compilerOptions": {
    "outDir": "../../sdk/vircadia-world-sdk-ts/dist-test"
  }
}
```

**Validation**:
```bash
# Test SDK build with unified config
cd /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts && bun run tsc --build ../../build-config/test/tsconfig.sdk-test.json

# Compare outputs
diff -r /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts/browser/dist /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts/dist-test 2>/dev/null || echo "Review differences"
```

---

### Step 2.3: Performance Benchmarking

**File**: `/home/devuser/workspace/project/build-config/scripts/benchmark.ts`

**Content**:
```typescript
#!/usr/bin/env node
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../../');

interface BenchmarkResult {
  name: string;
  duration: number;
  success: boolean;
}

async function runBuild(command: string, args: string[], cwd: string): Promise<number> {
  const startTime = Date.now();

  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      cwd,
      stdio: 'ignore',
      shell: true,
    });

    proc.on('close', (code) => {
      const duration = Date.now() - startTime;
      if (code === 0) {
        resolve(duration);
      } else {
        reject(new Error(`Build failed with code ${code}`));
      }
    });
  });
}

async function benchmark(): Promise<void> {
  console.log('🏁 Starting build benchmarks...\n');

  const results: BenchmarkResult[] = [];

  // Benchmark original client build
  console.log('Testing original client build...');
  try {
    const duration = await runBuild('npm', ['run', 'build'], path.join(projectRoot, 'client'));
    results.push({ name: 'Original Client', duration, success: true });
    console.log(`✓ Completed in ${(duration / 1000).toFixed(2)}s\n`);
  } catch (error) {
    results.push({ name: 'Original Client', duration: 0, success: false });
    console.log('✗ Failed\n');
  }

  // Benchmark unified client build
  console.log('Testing unified client build...');
  try {
    const duration = await runBuild(
      'npx',
      ['vite', 'build', '--config', '../build-config/test/vite.config.client-test.ts'],
      path.join(projectRoot, 'client')
    );
    results.push({ name: 'Unified Client', duration, success: true });
    console.log(`✓ Completed in ${(duration / 1000).toFixed(2)}s\n`);
  } catch (error) {
    results.push({ name: 'Unified Client', duration: 0, success: false });
    console.log('✗ Failed\n');
  }

  // Print summary
  console.log('📊 Benchmark Results:\n');
  for (const result of results) {
    if (result.success) {
      console.log(`${result.name}: ${(result.duration / 1000).toFixed(2)}s`);
    } else {
      console.log(`${result.name}: FAILED`);
    }
  }

  // Calculate performance difference
  const originalClient = results.find(r => r.name === 'Original Client');
  const unifiedClient = results.find(r => r.name === 'Unified Client');

  if (originalClient?.success && unifiedClient?.success) {
    const diff = ((unifiedClient.duration - originalClient.duration) / originalClient.duration) * 100;
    console.log(`\n📈 Performance: ${diff > 0 ? '+' : ''}${diff.toFixed(1)}%`);
  }
}

benchmark();
```

**Validation**:
```bash
chmod +x /home/devuser/workspace/project/build-config/scripts/benchmark.ts
cd /home/devuser/workspace/project && node build-config/scripts/benchmark.ts
```

---

### Step 2.4: Feature Parity Testing

Create comprehensive test checklist:

**File**: `/home/devuser/workspace/project/build-config/test/feature-parity-checklist.md`

**Content**:
```markdown
# Feature Parity Checklist

## Client Build Features

- [ ] TypeScript compilation works
- [ ] React JSX transformation works
- [ ] Path aliases (@/* imports) resolve correctly
- [ ] Source maps generated
- [ ] Assets bundled correctly
- [ ] HMR functions in dev mode
- [ ] Production build is minified
- [ ] Vendor chunks created
- [ ] Environment variables processed
- [ ] Proxy configuration works

## SDK Build Features

- [ ] TypeScript declaration files generated
- [ ] Browser target exports correctly
- [ ] Vue integration works
- [ ] Source maps generated
- [ ] Paths resolve correctly (@vircadia/*)
- [ ] ES modules output correctly
- [ ] Compatible with Bun runtime

## Build Performance

- [ ] Build time within 10% of original
- [ ] Output size similar to original
- [ ] Memory usage acceptable

## Developer Experience

- [ ] Clear error messages
- [ ] Fast incremental builds
- [ ] Good watch mode performance
```

**Validation**:
```bash
# Manual testing checklist - go through each item
cat /home/devuser/workspace/project/build-config/test/feature-parity-checklist.md
```

---

### Step 2.5: Phase 2 Validation Checkpoint

**Commands**:
```bash
# Run all Phase 2 tests
cd /home/devuser/workspace/project/build-config

# 1. Build with test configs
npm run test:config

# 2. Run benchmarks
node scripts/benchmark.ts

# 3. Validate feature parity manually
cat test/feature-parity-checklist.md

# 4. Compare build outputs
diff -r ../client/dist ../client/dist-test
```

**Success Criteria**:
- ✅ Test builds complete successfully
- ✅ Performance within acceptable range (±10%)
- ✅ All features from checklist working
- ✅ Build outputs functionally equivalent

**Decision Point**: Proceed to Phase 3 only if ALL criteria met.

---

## PHASE 3: MIGRATION TO UNIFIED SYSTEM

**Goal**: Migrate projects to use unified configuration and remove old config files.

**Duration**: 1 hour
**Risk Level**: MEDIUM (modifies existing files)
**Rollback**: Git reset or restore from backup

---

### Step 3.1: Create Backup Before Migration

**Commands**:
```bash
# Create backup directory
mkdir -p /home/devuser/workspace/project/.backup/pre-unified-migration

# Backup client configuration
cp /home/devuser/workspace/project/client/vite.config.ts /home/devuser/workspace/project/.backup/pre-unified-migration/
cp /home/devuser/workspace/project/client/tsconfig.json /home/devuser/workspace/project/.backup/pre-unified-migration/client-tsconfig.json
cp /home/devuser/workspace/project/client/package.json /home/devuser/workspace/project/.backup/pre-unified-migration/client-package.json

# Backup SDK configuration
cp /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts/tsconfig.json /home/devuser/workspace/project/.backup/pre-unified-migration/sdk-tsconfig.json
cp /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts/package.json /home/devuser/workspace/project/.backup/pre-unified-migration/sdk-package.json

# Create restore script
cat > /home/devuser/workspace/project/.backup/pre-unified-migration/RESTORE.sh << 'EOF'
#!/bin/bash
echo "Restoring pre-migration configuration..."
cp vite.config.ts ../../client/
cp client-tsconfig.json ../../client/tsconfig.json
cp client-package.json ../../client/package.json
cp sdk-tsconfig.json ../../sdk/vircadia-world-sdk-ts/tsconfig.json
cp sdk-package.json ../../sdk/vircadia-world-sdk-ts/package.json
echo "✓ Configuration restored"
EOF

chmod +x /home/devuser/workspace/project/.backup/pre-unified-migration/RESTORE.sh
```

**Success Criteria**:
- All configuration files backed up
- Restore script created and tested

**Validation**:
```bash
# Verify backup
ls -la /home/devuser/workspace/project/.backup/pre-unified-migration/

# Test restore script (dry run)
cd /home/devuser/workspace/project/.backup/pre-unified-migration && bash -n RESTORE.sh
```

---

### Step 3.2: Update Client Vite Configuration

**Operation**: Modify `/home/devuser/workspace/project/client/vite.config.ts`

**Changes**:
```typescript
// NEW CONTENT
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { createClientConfig } from '../build-config/shared/vite.config.factory';

export default defineConfig({
  ...createClientConfig({
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: parseInt(process.env.VITE_DEV_SERVER_PORT || '5173'),
      strictPort: true,
      allowedHosts: [
        'www.visionflow.info',
        'visionflow.info',
        'localhost',
        '192.168.0.51'
      ],
      hmr: {
        clientPort: 3001,
        path: '/vite-hmr',
      },
      watch: {
        usePolling: true,
        interval: 1000,
      },
      cors: true,
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
      proxy: {
        '/api': {
          target: process.env.VITE_API_URL || 'http://visionflow_container:4000',
          changeOrigin: true,
          secure: false,
        },
        '/ws': {
          target: process.env.VITE_WS_URL || 'ws://visionflow_container:4000',
          ws: true,
          changeOrigin: true
        },
        '/wss': {
          target: process.env.VITE_WS_URL || 'ws://visionflow_container:4000',
          ws: true,
          changeOrigin: true
        }
      }
    },
  }),
  optimizeDeps: {
    include: ['@getalby/sdk']
  }
});
```

**Success Criteria**:
- Configuration imports factory
- All Docker/HMR settings preserved
- Proxy configuration intact

**Validation**:
```bash
# Test client build
cd /home/devuser/workspace/project/client && npm run build

# Test client dev server (background)
cd /home/devuser/workspace/project/client && timeout 10s npm run dev || echo "Dev server started successfully"
```

---

### Step 3.3: Update Client TypeScript Configuration

**Operation**: Modify `/home/devuser/workspace/project/client/tsconfig.json`

**Changes**:
```json
{
  "extends": "../build-config/shared/tsconfig.client.json"
}
```

**Success Criteria**:
- Extends unified config
- All path aliases work
- TypeScript compilation succeeds

**Validation**:
```bash
# Test TypeScript compilation
cd /home/devuser/workspace/project/client && npx tsc --noEmit

# Test build
cd /home/devuser/workspace/project/client && npm run build
```

---

### Step 3.4: Update SDK TypeScript Configuration

**Operation**: Modify `/home/devuser/workspace/project/sdk/vircadia-world-sdk-ts/tsconfig.json`

**Changes**:
```json
{
  "extends": "../../build-config/shared/tsconfig.sdk.json"
}
```

**Success Criteria**:
- Extends unified SDK config
- Browser/Vue exports work
- Bun build succeeds

**Validation**:
```bash
# Test SDK build
cd /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts && bun run build

# Verify outputs
ls -la /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts/browser/dist/
```

---

### Step 3.5: Update Root Package.json

**Operation**: Add unified build scripts to `/home/devuser/workspace/project/package.json`

**Changes** (add to scripts section):
```json
{
  "scripts": {
    "build": "cd build-config && npm run build",
    "build:parallel": "cd build-config && npm run build:parallel",
    "build:client": "cd client && npm run build",
    "build:sdk": "cd sdk/vircadia-world-sdk-ts && bun run build",
    "build:validate": "cd build-config && npm run validate",
    "build:clean": "cd build-config && npm run clean"
  }
}
```

**Success Criteria**:
- Scripts callable from project root
- Build commands work from any location

**Validation**:
```bash
# Test from project root
cd /home/devuser/workspace/project && npm run build:client
cd /home/devuser/workspace/project && npm run build:sdk
cd /home/devuser/workspace/project && npm run build
```

---

### Step 3.6: Phase 3 Validation Checkpoint

**Commands**:
```bash
# Full system test after migration
cd /home/devuser/workspace/project

# 1. Clean build from scratch
npm run build:clean
npm run build

# 2. Validate outputs
npm run build:validate

# 3. Test individual builds
npm run build:client
npm run build:sdk

# 4. Test dev mode
cd client && timeout 10s npm run dev

# 5. Verify no regressions
git diff client/package.json
git diff sdk/vircadia-world-sdk-ts/package.json
```

**Success Criteria**:
- ✅ All builds complete successfully
- ✅ Validation passes
- ✅ Dev mode works
- ✅ Build outputs identical to Phase 2 tests

**Rollback (if needed)**:
```bash
cd /home/devuser/workspace/project/.backup/pre-unified-migration
./RESTORE.sh
```

---

## PHASE 4: CLEANUP AND DOCUMENTATION

**Goal**: Remove legacy configuration files and update documentation.

**Duration**: 30 minutes
**Risk Level**: LOW
**Rollback**: Git reset individual files

---

### Step 4.1: Remove Obsolete Configuration Files

**Commands**:
```bash
# Remove test configurations
rm -rf /home/devuser/workspace/project/build-config/test/

# Clean test outputs
rm -rf /home/devuser/workspace/project/client/dist-test
rm -rf /home/devuser/workspace/project/sdk/vircadia-world-sdk-ts/dist-test

# Remove validation results (optional - keep for records)
# rm -rf /home/devuser/workspace/project/build-config/validation-results/
```

**Success Criteria**:
- Test files removed
- Production configs remain
- Git status clean

**Validation**:
```bash
git status /home/devuser/workspace/project/build-config/
ls -la /home/devuser/workspace/project/build-config/
```

---

### Step 4.2: Update Build Documentation

**File**: `/home/devuser/workspace/project/docs/BUILD_SYSTEM.md`

**Content**:
```markdown
# VisionFlow Unified Build System

## Overview

The VisionFlow project uses a unified build configuration system that manages both the client (React/Vite) and SDK (TypeScript) builds from a central location.

## Structure

```
/build-config
├── shared/                    # Shared configurations
│   ├── tsconfig.base.json    # Base TypeScript config
│   ├── tsconfig.client.json  # Client-specific TS config
│   ├── tsconfig.sdk.json     # SDK-specific TS config
│   └── vite.config.factory.ts # Vite configuration factory
├── scripts/                   # Build automation scripts
│   ├── build-all.ts          # Unified build orchestration
│   ├── validate-build.ts     # Build output validation
│   └── benchmark.ts          # Performance benchmarking
└── package.json              # Build system dependencies
```

## Quick Start

### Build Everything
```bash
npm run build                 # Sequential build (safer)
npm run build:parallel        # Parallel build (faster)
```

### Build Individual Targets
```bash
npm run build:client          # Build client only
npm run build:sdk             # Build SDK only
```

### Validation
```bash
npm run build:validate        # Validate build outputs
```

### Clean
```bash
npm run build:clean           # Remove all build artifacts
```

## Configuration

### Client Configuration

The client uses Vite with React. Configuration is in `/client/vite.config.ts`, which extends the unified factory.

**Key Features**:
- Hot Module Replacement (HMR)
- Docker environment support
- API proxy configuration
- Path aliases (@/*)

### SDK Configuration

The SDK uses TypeScript compiler targeting ESM output. Configuration is in `/sdk/vircadia-world-sdk-ts/tsconfig.json`.

**Key Features**:
- Type declaration generation
- Browser/Vue exports
- Bun runtime compatibility

## Development

### Adding New Build Targets

1. Create configuration in `/build-config/shared/`
2. Add build script in `/build-config/scripts/build-all.ts`
3. Update validation in `/build-config/scripts/validate-build.ts`

### Modifying Shared Configuration

Edit files in `/build-config/shared/`. Changes propagate to all targets that extend the config.

## Troubleshooting

### Build Fails

1. Clean build artifacts: `npm run build:clean`
2. Check validation: `npm run build:validate`
3. Review logs in console output

### Performance Issues

Run benchmarks to identify bottlenecks:
```bash
cd build-config && node scripts/benchmark.ts
```

## Migration History

- **Phase 1**: Created unified configuration alongside existing configs
- **Phase 2**: Tested unified system for feature parity
- **Phase 3**: Migrated projects to unified configuration
- **Phase 4**: Removed legacy configurations and updated docs

## Rollback

If issues occur, restore previous configuration:
```bash
cd .backup/pre-unified-migration
./RESTORE.sh
```
```

**Success Criteria**:
- Documentation is comprehensive
- All commands documented
- Troubleshooting guide included

---

### Step 4.3: Update CLAUDE.md with Build Instructions

**Operation**: Add build system section to `/home/devuser/workspace/project/CLAUDE.md`

**Addition** (add after "Build Commands" section):
```markdown
## Unified Build System

The project uses a centralized build configuration in `/build-config/`:

### Commands
- `npm run build` - Build all targets sequentially
- `npm run build:parallel` - Build all targets in parallel (faster)
- `npm run build:client` - Build client only
- `npm run build:sdk` - Build SDK only
- `npm run build:validate` - Validate build outputs
- `npm run build:clean` - Clean all build artifacts

### Configuration
- **Client**: Vite + React, config extends `/build-config/shared/vite.config.factory.ts`
- **SDK**: TypeScript compiler, config extends `/build-config/shared/tsconfig.sdk.json`

See `/docs/BUILD_SYSTEM.md` for detailed documentation.
```

---

### Step 4.4: Create Migration Summary Report

**File**: `/home/devuser/workspace/project/docs/MIGRATION_REPORT.md`

**Content**:
```markdown
# Unified Build System Migration Report

**Date**: [CURRENT_DATE]
**Status**: COMPLETED

## Summary

Successfully migrated VisionFlow project from separate build configurations to a unified build system.

## Changes Made

### New Files Created
- `/build-config/` - Unified build configuration directory
- `/build-config/shared/tsconfig.base.json` - Base TypeScript config
- `/build-config/shared/tsconfig.client.json` - Client TypeScript config
- `/build-config/shared/tsconfig.sdk.json` - SDK TypeScript config
- `/build-config/shared/vite.config.factory.ts` - Vite config factory
- `/build-config/scripts/build-all.ts` - Build orchestration
- `/build-config/scripts/validate-build.ts` - Build validation
- `/build-config/scripts/benchmark.ts` - Performance benchmarking
- `/build-config/package.json` - Build system package file

### Modified Files
- `/client/vite.config.ts` - Now extends factory
- `/client/tsconfig.json` - Now extends shared config
- `/sdk/vircadia-world-sdk-ts/tsconfig.json` - Now extends shared config
- `/package.json` - Added unified build scripts

### Removed Files
- None (backward compatibility maintained)

## Benefits

1. **Centralized Configuration**: Single source of truth for build settings
2. **Reduced Duplication**: Shared TypeScript settings across projects
3. **Better Maintainability**: Changes propagate automatically
4. **Improved DX**: Unified commands from project root
5. **Validation**: Automated build output validation

## Performance

| Build Type | Before | After | Change |
|------------|--------|-------|--------|
| Client     | [TIME]s | [TIME]s | [DELTA]% |
| SDK        | [TIME]s | [TIME]s | [DELTA]% |
| Parallel   | N/A    | [TIME]s | N/A |

## Testing

All Phase 2 tests passed:
- ✅ Build output validation
- ✅ Feature parity confirmed
- ✅ Performance within acceptable range
- ✅ Developer experience improved

## Rollback Plan

Backup location: `.backup/pre-unified-migration/`
Restore command: `cd .backup/pre-unified-migration && ./RESTORE.sh`

## Next Steps

1. Monitor build performance in production
2. Gather team feedback on new system
3. Consider extending to other parts of the project
4. Document any issues in GitHub Issues

## Contributors

- Migration executed by: [AI Agent]
- Approved by: [TBD]
- Date: [CURRENT_DATE]
```

---

### Step 4.5: Phase 4 Validation Checkpoint

**Commands**:
```bash
# Final system validation
cd /home/devuser/workspace/project

# 1. Build from clean state
npm run build:clean
npm run build

# 2. Validate
npm run build:validate

# 3. Check documentation
cat docs/BUILD_SYSTEM.md
cat docs/MIGRATION_REPORT.md

# 4. Verify git status
git status

# 5. Test all commands
npm run build:client
npm run build:sdk
npm run build:parallel
```

**Success Criteria**:
- ✅ All documentation updated
- ✅ All commands work correctly
- ✅ Migration report complete
- ✅ System ready for production use

---

## Post-Migration Actions

### Immediate (Day 1)
```bash
# Commit changes
git add build-config/
git add client/vite.config.ts client/tsconfig.json
git add sdk/vircadia-world-sdk-ts/tsconfig.json
git add package.json docs/
git commit -m "feat: implement unified build system

- Created centralized build configuration in /build-config
- Migrated client and SDK to use shared configs
- Added build orchestration and validation scripts
- Updated documentation with new build system

BREAKING CHANGE: Build commands now run from project root
See docs/BUILD_SYSTEM.md for migration details"
```

### Short-term (Week 1)
- Monitor CI/CD pipelines for issues
- Gather team feedback
- Address any edge cases
- Update IDE configurations if needed

### Long-term (Month 1)
- Evaluate extending to additional packages
- Consider additional optimizations
- Review and update benchmarks
- Document lessons learned

---

## Troubleshooting Guide

### Issue: Build fails with "Cannot find module"

**Solution**:
```bash
# Reinstall dependencies
cd /home/devuser/workspace/project/build-config && npm install
cd /home/devuser/workspace/project/client && npm install
```

### Issue: TypeScript errors after migration

**Solution**:
```bash
# Clear TypeScript cache
rm -rf client/.tsbuildinfo
rm -rf sdk/vircadia-world-sdk-ts/.tsbuildinfo
npx tsc --build --clean
```

### Issue: Vite dev server not working

**Solution**:
```bash
# Check config inheritance
cd client && npx vite --debug
```

### Issue: Need to rollback

**Solution**:
```bash
cd .backup/pre-unified-migration && ./RESTORE.sh
npm run build  # Test with old configuration
```

---

## Success Metrics

### Build Performance
- Sequential build time: ≤ original + 10%
- Parallel build time: < sequential time
- Validation overhead: < 5 seconds

### Code Quality
- No TypeScript errors
- All tests passing
- No console warnings

### Developer Experience
- Single command for all builds
- Clear error messages
- Fast incremental builds

---

## Appendix A: File Mapping

### Before → After

| Original File | New File | Change |
|--------------|----------|---------|
| `/client/vite.config.ts` | `/client/vite.config.ts` | Modified to extend factory |
| `/client/tsconfig.json` | `/client/tsconfig.json` | Modified to extend shared |
| `/sdk/.../tsconfig.json` | `/sdk/.../tsconfig.json` | Modified to extend shared |
| N/A | `/build-config/**/*` | New unified configs |

---

## Appendix B: Command Reference

### Build Commands
```bash
npm run build                 # Build all (sequential)
npm run build:parallel        # Build all (parallel)
npm run build:client          # Client only
npm run build:sdk             # SDK only
```

### Validation Commands
```bash
npm run build:validate        # Validate outputs
npm run build:clean           # Clean artifacts
```

### Development Commands
```bash
cd client && npm run dev      # Client dev server
cd sdk/... && bun run build   # SDK build
```

### Utility Commands
```bash
cd build-config && node scripts/benchmark.ts    # Benchmark
cd build-config && node scripts/validate-build.ts  # Validate
```

---

## Appendix C: Configuration Schema

### Base TypeScript Config
```json
{
  "target": "ES2022",
  "module": "ESNext",
  "strict": true,
  "skipLibCheck": true,
  "declaration": true,
  "sourceMap": true
}
```

### Client Extensions
```json
{
  "jsx": "react-jsx",
  "lib": ["ES2020", "DOM"],
  "paths": { "@/*": ["client/src/*"] }
}
```

### SDK Extensions
```json
{
  "lib": ["ES2022", "DOM"],
  "outDir": "sdk/.../dist",
  "paths": { "@vircadia/*": ["sdk/.../*"] }
}
```

---

## Implementation Timeline

| Phase | Duration | Tasks | Status |
|-------|----------|-------|--------|
| Phase 1 | 2-3 hours | Create unified configs | ⏳ Pending |
| Phase 2 | 1-2 hours | Test unified system | ⏳ Pending |
| Phase 3 | 1 hour | Migrate projects | ⏳ Pending |
| Phase 4 | 30 mins | Cleanup & docs | ⏳ Pending |
| **Total** | **4-6 hours** | | |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Build breaks | Low | High | Full backup, phased approach |
| Performance regression | Low | Medium | Benchmarking in Phase 2 |
| Feature loss | Very Low | High | Feature parity testing |
| Team confusion | Medium | Low | Comprehensive docs |

---

## Approval Checklist

- [ ] All phases validated successfully
- [ ] Documentation complete
- [ ] Team reviewed migration plan
- [ ] Backup created
- [ ] Rollback procedure tested
- [ ] CI/CD pipeline updated
- [ ] Performance acceptable

---

**END OF IMPLEMENTATION PLAN**

# Dependency Management

## Philosophy

The CachyOS Docker environment uses a **hermetic local dependencies pattern** for all auxiliary tools. This ensures reproducible builds, eliminates version conflicts, and maintains a self-contained project structure.

## Dependency Categories

### Global Dependencies
Only one package is installed globally:

- **agentic-flow**: Main application CLI

```dockerfile
RUN cd /tmp/agentic-flow && npm install -g .
```

This provides the `agentic-flow` command system-wide.

### Local Dependencies (devDependencies)
All auxiliary tools are installed locally as devDependencies:

- **pm2**: Process manager for production deployments
- **@clduab11/gemini-flow**: AI swarm orchestration framework
- **@anthropic-ai/claude-code**: Claude CLI for interactive development

```json
{
  "devDependencies": {
    "pm2": "^5.4.3",
    "@clduab11/gemini-flow": "^1.0.0",
    "@anthropic-ai/claude-code": "^0.1.0"
  }
}
```

### Runtime Dependencies
Core application dependencies:

```json
{
  "dependencies": {
    "@anthropic-ai/claude-agent-sdk": "^0.1.5",
    "@anthropic-ai/sdk": "^0.65.0",
    "@google/genai": "^1.22.0",
    "claude-flow": "^2.0.0",
    "express": "^5.1.0",
    "fastmcp": "^3.19.0"
  }
}
```

## Benefits of Local Pattern

### 1. Version Locking
All tools pinned to specific versions in `package-lock.json`:
- Eliminates "works on my machine" issues
- Reproducible across all environments
- Easy to audit what versions are running

### 2. No Conflicts
Different projects can use different versions:
```bash
# Project A uses pm2@5.3.0
cd /project-a && npm install

# Project B uses pm2@5.4.3
cd /project-b && npm install

# No conflict - each has its own version
```

### 3. Hermetic Builds
All dependencies self-contained:
- No reliance on global npm state
- Fresh installs guaranteed to work
- Docker layer caching works optimally

### 4. Easier Updates
Update all tools in one command:
```bash
cd /tmp/agentic-flow
npm update
```

### 5. Better CI/CD
Consistent across development, staging, production:
- Same versions everywhere
- No surprises in production
- Faster debugging

## How It Works

### Installation Flow

1. **Copy package files** (Docker layer caching)
```dockerfile
COPY --chown=devuser:devuser agentic-flow/package*.json /tmp/agentic-flow/
```

2. **Install all dependencies** (including devDependencies)
```dockerfile
RUN npm install && npm cache clean --force
```

3. **Copy source and build**
```dockerfile
COPY --chown=devuser:devuser agentic-flow/ /tmp/agentic-flow/
RUN npm run build
```

4. **Install main package globally**
```dockerfile
RUN npm install -g .
```

5. **Create convenience symlinks**
```dockerfile
RUN ln -sf /tmp/agentic-flow/node_modules/.bin/pm2 /usr/local/bin/pm2 && \
    ln -sf /tmp/agentic-flow/node_modules/.bin/gemini-flow /usr/local/bin/gemini-flow && \
    ln -sf /tmp/agentic-flow/node_modules/.bin/claude-code /usr/local/bin/claude-code
```

### PATH Configuration

The shell configuration prioritizes local binaries:

```bash
# .zshrc
export PATH="/tmp/agentic-flow/node_modules/.bin:$PATH"
export PATH="$HOME/.npm-global/bin:$PATH"
```

This ensures:
1. Project-local tools are found first
2. Global tools available as fallback
3. System binaries available last

### Execution Methods

Tools can be executed three ways:

#### 1. Direct Execution (via PATH)
```bash
pm2 start app.js
gemini-flow init
claude-code
```

#### 2. npm Scripts
```bash
npm run pm2 -- start app.js
npm run gemini-flow -- init
npm run claude-code
```

#### 3. Direct Path
```bash
/tmp/agentic-flow/node_modules/.bin/pm2 start app.js
```

## Package.json Configuration

### npm Scripts
Convenience scripts for common operations:

```json
{
  "scripts": {
    "pm2": "pm2",
    "gemini-flow": "gemini-flow",
    "claude-code": "claude-code"
  }
}
```

Usage:
```bash
npm run pm2 -- start ecosystem.config.js
npm run gemini-flow -- agents spawn --count 10
npm run claude-code -- --help
```

## Dockerfile Best Practices

### Layer Caching Optimization

**Bad** (breaks caching on any source change):
```dockerfile
COPY agentic-flow/ /tmp/agentic-flow/
RUN npm install && npm run build
```

**Good** (only rebuilds if dependencies change):
```dockerfile
# 1. Copy package files only
COPY agentic-flow/package*.json /tmp/agentic-flow/
WORKDIR /tmp/agentic-flow

# 2. Install dependencies (cached unless package files change)
RUN npm install && npm cache clean --force

# 3. Copy source (frequently changes)
COPY agentic-flow/ /tmp/agentic-flow/

# 4. Build (fast if dependencies cached)
RUN npm run build
```

### No Global Installations

**Bad** (hard to version, conflicts possible):
```dockerfile
RUN npm install -g pm2 gemini-flow claude-code
```

**Good** (version-locked, no conflicts):
```dockerfile
# Dependencies in package.json
COPY package*.json /tmp/agentic-flow/
RUN npm install
```

## Updating Dependencies

### Update All Dependencies
```bash
docker exec agentic-flow-cachyos bash -c "
  cd /tmp/agentic-flow
  npm update
  npm audit fix
"
```

### Update Specific Package
```bash
docker exec agentic-flow-cachyos bash -c "
  cd /tmp/agentic-flow
  npm update pm2
"
```

### Check Outdated Packages
```bash
docker exec agentic-flow-cachyos bash -c "
  cd /tmp/agentic-flow
  npm outdated
"
```

### Rebuild After Updates
```bash
docker-compose build --no-cache
docker-compose up -d
```

## Troubleshooting

### Tool Not Found

**Symptom**:
```bash
$ pm2 --version
bash: pm2: command not found
```

**Solution**:
```bash
# 1. Check PATH
echo $PATH | grep node_modules

# 2. Check binary exists
ls -la /tmp/agentic-flow/node_modules/.bin/pm2

# 3. Check symlink
ls -la /usr/local/bin/pm2

# 4. Reinstall dependencies
cd /tmp/agentic-flow && npm install

# 5. Recreate symlinks
ln -sf /tmp/agentic-flow/node_modules/.bin/pm2 /usr/local/bin/pm2
```

### Version Mismatch

**Symptom**:
```bash
$ pm2 --version
5.3.0  # Expected 5.4.3
```

**Solution**:
```bash
# 1. Check package.json version
cat /tmp/agentic-flow/package.json | jq '.devDependencies.pm2'

# 2. Check installed version
npm list pm2

# 3. Reinstall to match package.json
cd /tmp/agentic-flow
rm -rf node_modules package-lock.json
npm install
```

### Dependency Installation Fails

**Symptom**:
```bash
npm ERR! code ENOTFOUND
npm ERR! network request to https://registry.npmjs.org/pm2 failed
```

**Solution**:
```bash
# 1. Check network connectivity
ping registry.npmjs.org

# 2. Try with different registry
npm config set registry https://registry.npm.taobao.org
npm install

# 3. Clear npm cache
npm cache clean --force
npm install

# 4. Use offline install if available
npm ci --offline
```

## Migration from Global Pattern

If migrating from global npm installations:

### Before
```dockerfile
RUN npm install -g pm2 @clduab11/gemini-flow @anthropic-ai/claude-code
```

### After
1. **Add to package.json**:
```json
{
  "devDependencies": {
    "pm2": "^5.4.3",
    "@clduab11/gemini-flow": "^1.0.0",
    "@anthropic-ai/claude-code": "^0.1.0"
  }
}
```

2. **Update Dockerfile**:
```dockerfile
COPY package*.json /tmp/agentic-flow/
RUN npm install
COPY . /tmp/agentic-flow/

# Create symlinks for convenience
RUN ln -sf /tmp/agentic-flow/node_modules/.bin/pm2 /usr/local/bin/pm2
```

3. **Update PATH**:
```bash
export PATH="/tmp/agentic-flow/node_modules/.bin:$PATH"
```

4. **Rebuild**:
```bash
docker-compose build --no-cache
```

## Best Practices

### DO
- ✅ Add auxiliary tools to `devDependencies`
- ✅ Use `package-lock.json` for version locking
- ✅ Create symlinks for convenience
- ✅ Add local `node_modules/.bin` to PATH
- ✅ Test with `npm ci` for reproducibility

### DON'T
- ❌ Install tools globally unless necessary
- ❌ Commit `node_modules` to git
- ❌ Use `npm install -g` for auxiliary tools
- ❌ Skip package-lock.json
- ❌ Manually manage versions

## Performance Considerations

### Docker Layer Caching
Properly ordering Dockerfile commands ensures fast rebuilds:

```dockerfile
# 1. Package files (changes rarely)
COPY package*.json /app/
RUN npm install  # ← Cached unless package files change

# 2. Source code (changes frequently)
COPY . /app/
RUN npm run build  # ← Fast if dependencies cached
```

### npm ci vs npm install
Use `npm ci` for faster, reproducible installs:

```dockerfile
# Development
RUN npm install

# Production/CI
RUN npm ci --only=production
```

`npm ci`:
- Faster than `npm install`
- Requires package-lock.json
- Removes node_modules before installing
- Never modifies package-lock.json
- More reproducible

## Security Considerations

### Audit Dependencies
Regular security audits:

```bash
# Check for vulnerabilities
npm audit

# Auto-fix vulnerabilities
npm audit fix

# Force fix (may break compatibility)
npm audit fix --force
```

### Lock File Integrity
Verify package-lock.json integrity:

```bash
# Generate integrity hashes
npm install --package-lock-only

# Verify integrity
npm ci
```

### Minimal Installs
Production images should exclude devDependencies:

```dockerfile
# Production
RUN npm ci --only=production

# Development
RUN npm ci
```

## Conclusion

The local dependencies pattern provides:
- **Reproducibility**: Same versions everywhere
- **Isolation**: No global conflicts
- **Security**: Auditable dependency tree
- **Performance**: Optimal Docker caching
- **Maintainability**: Easy updates and debugging

This approach scales from development workstations to production deployments while maintaining consistency and reliability.

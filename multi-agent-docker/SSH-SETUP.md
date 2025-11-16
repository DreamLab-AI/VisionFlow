# SSH Credentials Setup

This guide explains how SSH credentials are configured in the Agentic Workstation container.

## Quick Start

### Automatic Setup (Recommended)

SSH credentials are automatically mounted from your host system at runtime - **no rebuild required**.

```bash
# Your host's ~/.ssh directory is mounted to the container
# Just restart the container to pick up new/updated keys
docker-compose -f docker-compose.unified.yml restart
```

### Verify SSH Setup

```bash
# Check SSH configuration status
./unified-config/scripts/ssh-setup.sh status

# Verify SSH keys are accessible
./unified-config/scripts/ssh-setup.sh verify

# Check SSH agent status
./unified-config/scripts/ssh-setup.sh agent

# Test SSH connection (e.g., to GitHub)
./unified-config/scripts/ssh-setup.sh test git@github.com
```

## How It Works

### 1. Volume Mount (docker-compose.unified.yml)

Your host SSH directory is mounted as a read-only volume:

```yaml
volumes:
  # SSH credentials (runtime mount - no rebuild needed)
  - ${HOME}/.ssh:/home/devuser/.ssh:ro
```

**Benefits:**
- ✅ No rebuild required when keys change
- ✅ Secure read-only mount
- ✅ Single source of truth (your host ~/.ssh)
- ✅ All keys automatically available
- ✅ Changes reflected immediately on container restart

### 2. Runtime Configuration (entrypoint-unified.sh)

On container startup (Phase 7.3), the entrypoint script:

1. Detects SSH credentials mount
2. Verifies ownership and permissions
3. Counts available keys
4. Configures SSH agent auto-start in .zshrc

**Auto-start SSH Agent:**

When devuser logs in, the SSH agent automatically starts and loads keys:

```bash
# Automatically added to ~/.zshrc
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval "$(ssh-agent -s)" > /dev/null 2>&1
    find ~/.ssh -type f -name "id_*" ! -name "*.pub" -exec ssh-add {} \; 2>/dev/null
fi
```

### 3. Directory Structure (Dockerfile.unified)

SSH directory is pre-created during build for consistency:

```dockerfile
# Create SSH directory structure (will be mounted at runtime)
RUN mkdir -p /home/devuser/.ssh && \
    chown devuser:devuser /home/devuser/.ssh && \
    chmod 700 /home/devuser/.ssh
```

## Usage Inside Container

### SSH to External Hosts

```bash
# SSH agent is already running with keys loaded
ssh user@example.com

# Git operations work automatically
git clone git@github.com:user/repo.git
```

### Check Available Keys

```bash
# List loaded keys
ssh-add -l

# List all SSH keys in directory
ls -la ~/.ssh/id_*

# Test GitHub connection
ssh -T git@github.com
```

### Add Additional Keys

```bash
# Inside container
ssh-add ~/.ssh/id_rsa_custom
```

## Troubleshooting

### Keys Not Found

**Problem:** SSH keys not showing in container

**Solution:**
```bash
# On host: verify keys exist
ls -la ~/.ssh/

# Check docker-compose.unified.yml has the mount
grep -A1 "SSH credentials" docker-compose.unified.yml

# Restart container
docker-compose -f docker-compose.unified.yml restart

# Verify from outside container
./unified-config/scripts/ssh-setup.sh status
```

### Permission Denied

**Problem:** SSH complains about permissions

**Solution:**
```bash
# SSH requires strict permissions on host
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_*
chmod 644 ~/.ssh/*.pub

# Restart container to pick up changes
docker-compose -f docker-compose.unified.yml restart
```

### SSH Agent Not Running

**Problem:** `SSH_AUTH_SOCK` not set

**Solution:**
```bash
# Inside container: manually start agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# Or reload shell to trigger auto-start
exec zsh
```

### Read-Only Mount Issues

The mount is intentionally read-only for security. If you need to modify keys:

**Option 1 (Recommended):** Modify on host, restart container
```bash
# On host
ssh-keygen -t ed25519 -C "your_email@example.com"

# Restart container
docker-compose -f docker-compose.unified.yml restart
```

**Option 2:** Manual copy (not recommended - lost on restart)
```bash
./unified-config/scripts/ssh-setup.sh copy ~/.ssh/id_new_key
```

## Security Considerations

### Read-Only Mount
- SSH directory is mounted read-only (`:ro`)
- Keys cannot be modified inside container
- Prevents accidental key corruption
- Malicious code in container cannot steal/modify keys

### Key Security Best Practices

1. **Use SSH Agent:** Keys stay in memory, not exposed to processes
2. **Passphrase Protection:** Always use passphrases on private keys
3. **Key Rotation:** Regularly rotate SSH keys
4. **Separate Keys:** Use different keys for different services
5. **Audit Access:** Regularly check `ssh-add -l` for loaded keys

## Advanced Configuration

### Custom SSH Config

Your host's `~/.ssh/config` is also mounted:

```bash
# Inside container
cat ~/.ssh/config

# Example config (on host ~/.ssh/config)
Host github
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github

Host gitlab
    HostName gitlab.com
    User git
    IdentityFile ~/.ssh/id_rsa_gitlab
```

### Multiple Key Types

All key types are supported:
- RSA: `id_rsa`, `id_rsa.pub`
- Ed25519: `id_ed25519`, `id_ed25519.pub`
- ECDSA: `id_ecdsa`, `id_ecdsa.pub`
- DSA: `id_dsa`, `id_dsa.pub` (deprecated)

### Known Hosts

The `known_hosts` file is also mounted:

```bash
# Add new host (persists to host ~/.ssh/known_hosts)
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Verify host key
ssh-keygen -F github.com
```

## Integration with Other Services

### Git Credentials

SSH keys work seamlessly with Git:

```bash
# Clone via SSH
git clone git@github.com:user/repo.git

# Change remote to SSH
git remote set-url origin git@github.com:user/repo.git

# Git operations use SSH keys automatically
git push origin main
```

### Docker Socket Access

Container has access to host Docker socket for Docker-in-Docker:

```bash
# Inside container: control host Docker
docker ps

# Clone and build projects that require Docker
git clone git@github.com:user/docker-project.git
cd docker-project
docker build -t myapp .
```

### VS Code Remote SSH

When using VS Code Remote-SSH to connect to container:

```bash
# SSH from VS Code uses container's SSH keys
# Can clone repos, push changes, etc. from VS Code terminal
```

## Helper Script Reference

### ssh-setup.sh Commands

```bash
# Show all available commands
./unified-config/scripts/ssh-setup.sh help

# Check configuration status
./unified-config/scripts/ssh-setup.sh status

# Verify keys are properly configured
./unified-config/scripts/ssh-setup.sh verify

# Check SSH agent status and loaded keys
./unified-config/scripts/ssh-setup.sh agent

# Test SSH connection to a host
./unified-config/scripts/ssh-setup.sh test git@github.com
./unified-config/scripts/ssh-setup.sh test user@example.com

# Copy key manually (not recommended)
./unified-config/scripts/ssh-setup.sh copy ~/.ssh/id_custom_key
```

## FAQ

**Q: Do I need to rebuild after changing SSH keys?**
A: No! Just restart the container: `docker-compose -f docker-compose.unified.yml restart`

**Q: Can I use multiple SSH keys?**
A: Yes, all keys in `~/.ssh/` are mounted and auto-loaded by SSH agent.

**Q: Are my keys secure?**
A: Yes, the mount is read-only and keys cannot be modified inside the container.

**Q: Can I use SSH config?**
A: Yes, `~/.ssh/config` is mounted and respected by SSH client.

**Q: What if I don't have SSH keys?**
A: Generate them on your host: `ssh-keygen -t ed25519 -C "your_email@example.com"`

**Q: Can I use different keys for different containers?**
A: Yes, modify `docker-compose.unified.yml` to mount a different directory:
```yaml
- /path/to/other/ssh:/home/devuser/.ssh:ro
```

**Q: Does this work with SSH agent forwarding?**
A: Yes, when connecting to the container via SSH, use `ssh -A` to forward your agent.

## Related Documentation

- [Docker Compose Configuration](docker-compose.unified.yml)
- [Entrypoint Script](unified-config/entrypoint-unified.sh)
- [Build Script](build-unified.sh)
- [Main CLAUDE.md](CLAUDE.md)

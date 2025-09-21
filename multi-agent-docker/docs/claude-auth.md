# Claude Code Authentication in Docker

This guide explains how to set up automatic Claude Code authentication for the Multi-Agent Docker environment.

## Overview

The Multi-Agent Docker container can automatically authenticate with Claude Code using your existing credentials from the host system. This eliminates the need to manually authenticate inside each container.

## Setup

### 1. Get Your Claude Credentials

On your host system where Claude Code is already authenticated, find your credentials:

```bash
cat ~/.claude/.credentials.json
```

You'll see output like:
```json
{
  "claudeAiOauth": {
    "accessToken": "sk-ant-oat01-...",
    "refreshToken": "sk-ant-ort01-...",
    "expiresAt": 1758512244525,
    "scopes": ["user:inference", "user:profile"],
    "subscriptionType": "max"
  }
}
```

### 2. Configure Environment Variables

Add your tokens to the `.env` file in the `multi-agent-docker` directory:

```bash
# Claude Code Authentication
CLAUDE_CODE_ACCESS=sk-ant-oat01-your-actual-access-token
CLAUDE_CODE_REFRESH=sk-ant-ort01-your-actual-refresh-token
```

**Important:** 
- Keep your `.env` file secure and never commit it to version control
- Add `.env` to your `.gitignore` file

### 3. Build and Run

When you start the container, it will automatically:
1. Create the necessary `.claude` directories
2. Generate a properly formatted credentials file
3. Set correct permissions
4. Configure authentication for both `/home/dev` and `/home/ubuntu` users

```bash
docker compose up -d multi-agent
```

### 4. Verify Authentication

Inside the container, you can verify Claude is authenticated:

```bash
docker exec -it multi-agent-container claude --version
```

## Updating Credentials

If your credentials expire or change, you have several options:

### Option 1: Update .env and Rebuild
1. Update the tokens in your `.env` file
2. Restart the container: `docker compose restart multi-agent`

### Option 2: Use the Update Script
Inside the container, run:
```bash
update-claude-auth
```

Or provide tokens directly:
```bash
update-claude-auth "sk-ant-oat01-new-token" "sk-ant-ort01-new-refresh"
```

### Option 3: Environment Variables
```bash
export CLAUDE_CODE_ACCESS="sk-ant-oat01-new-token"
export CLAUDE_CODE_REFRESH="sk-ant-ort01-new-refresh"
update-claude-auth
```

## Security Considerations

1. **Never commit credentials**: Always use `.env` files that are gitignored
2. **Rotate tokens regularly**: Claude tokens expire after 30 days
3. **Limit access**: Ensure your Docker host and containers have appropriate access controls
4. **Use secrets management**: For production, consider using Docker secrets or a secrets management system

## Troubleshooting

### Authentication Not Working

1. Check if credentials are set:
   ```bash
   docker exec -it multi-agent-container cat /home/dev/.claude/.credentials.json
   ```

2. Verify environment variables were passed:
   ```bash
   docker exec -it multi-agent-container env | grep CLAUDE_CODE
   ```

3. Check permissions:
   ```bash
   docker exec -it multi-agent-container ls -la /home/dev/.claude/
   ```

### Token Expired

If you see authentication errors, your token may have expired. Update it using any of the methods described above.

### Permission Issues

The container automatically sets correct permissions, but if you have issues:
```bash
docker exec -it multi-agent-container bash -c "chmod 600 /home/dev/.claude/.credentials.json && chown dev:dev /home/dev/.claude -R"
```

## Integration with CI/CD

For automated deployments:

1. Store tokens as secrets in your CI/CD platform
2. Pass them as build arguments or environment variables
3. Never log or expose tokens in build output

Example GitHub Actions:
```yaml
- name: Build Docker image
  env:
    CLAUDE_CODE_ACCESS: ${{ secrets.CLAUDE_CODE_ACCESS }}
    CLAUDE_CODE_REFRESH: ${{ secrets.CLAUDE_CODE_REFRESH }}
  run: |
    docker compose build multi-agent
```
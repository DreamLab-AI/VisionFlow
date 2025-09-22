# Claude Code Authentication in Docker

This guide explains the simplified and robust methods for authenticating Claude Code within the Multi-Agent Docker environment. The system is designed to "just work" for local development while providing secure options for CI/CD and other non-interactive environments.

## Authentication Methods

There are three primary ways to authenticate, in order of recommendation:

1. **Host Directory Mount** (Recommended for Local Development): The container directly and securely mounts your host machine's Claude configuration.
2. **OAuth Token** (Recommended for CI/CD): Use a pre-generated, long-lived OAuth token passed as an environment variable.
3. **API Key** (Alternative): Use a standard Anthropic API key, also passed as an environment variable.

## Method 1: Host Directory Mount (Default)

This is the easiest and most secure method for local development.

**How it Works:**
The `docker-compose.yml` file is configured to mount your host's `~/.claude` directory and `~/.claude.json` file directly into the container. This means the container inherits your exact authentication state.

**Setup Steps:**

1. **Log in on Your Host**: If you haven't already, run `claude login` in your local terminal (not in Docker) and complete the browser authentication.

2. **Start the Container**:
   ```bash
   ./multi-agent.sh start
   ```

That's it. The container will start with you already authenticated as your Claude user.

**Benefits:**
- **Seamless**: No configuration needed.
- **Persistent**: Your login state survives container restarts.
- **Secure**: Your credentials are never copied into the container; they are securely mounted from the host.

## Method 2: OAuth Token (For CI/CD & Non-Interactive Use)

This method is ideal for automated environments where you cannot perform a browser login.

**Setup Steps:**

1. **Generate a Token**: On any machine where you are logged into Claude, run:
   ```bash
   claude setup-token
   ```
   This will generate a long-lived OAuth token that starts with `sk-ant-oat01-...`.

2. **Set the Environment Variable**: Copy the generated token and add it to your `.env` file:
   ```env
   # .env file
   CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...your-long-token...
   ```

3. **Start the Container**:
   ```bash
   ./multi-agent.sh start
   ```

The container will detect the environment variable and use it to authenticate non-interactively.

## Method 3: API Key (Alternative)

You can also use a standard Anthropic API key.

1. **Get Your API Key**: Obtain an API key from your [Anthropic account dashboard](https://console.anthropic.com/).

2. **Set the Environment Variable**: Add the key to your `.env` file:
   ```env
   # .env file
   ANTHROPIC_API_KEY=sk-ant-api03-...your-api-key...
   ```

3. **Start the Container**:
   ```bash
   ./multi-agent.sh start
   ```

## Verification

To verify that you are authenticated inside the container, run:

```bash
# From your host
./multi-agent.sh shell

# Inside the container
claude --version
```

If authentication is successful, this command will return the version number without prompting you to log in. You can also use the alias `dsp --version` (dsp stands for `--dangerously-skip-permissions`).

## Troubleshooting

- **Permission Denied Errors**: Ensure the user running Docker on your host has the correct permissions to read `~/.claude` and `~/.claude.json`. The Dockerfile is already configured to match the host's User ID and Group ID to prevent this.

- **"Interactive login required" Error**: This can happen if both the mount and environment variables are missing. Ensure one of the setup methods above is correctly configured.

- **Interactive Login Prompt in Container**: Recent versions of Claude may still show an interactive login prompt when run in a TTY session, even with valid mounted credentials. This is a known behavior. You can either:
  - Complete the login flow once inside the container
  - Use Claude non-interactively (e.g., in scripts or with piped input)
  - Set `CLAUDE_CODE_OAUTH_TOKEN` in your `.env` file using Method 2

- **Check the .env file**: Make sure you've copied `.env.example` to `.env` and that the variables are set correctly.

## Security Notes

- The host directory mount method is read-only for `~/.claude.json` to prevent accidental modifications.
- OAuth tokens and API keys should be treated as secrets. Never commit them to version control.
- Use environment variables only in secure CI/CD environments where secrets are properly managed.
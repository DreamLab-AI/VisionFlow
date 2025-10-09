FROM node:20-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for Claude Code
RUN useradd -m -s /bin/bash claude && \
    mkdir -p /home/claude/.claude && \
    chown -R claude:claude /home/claude

# Install Claude Code globally (official package)
RUN npm install -g @anthropic-ai/claude-code

# Create app directory
WORKDIR /app

# Create wrapper service directory
RUN mkdir -p /app/wrapper

# Copy wrapper files
COPY claude-zai/wrapper/ /app/wrapper/

# Install wrapper dependencies
WORKDIR /app/wrapper
RUN npm install

# Configure Claude Code for Z.AI
COPY claude-zai/claude-config.json /home/claude/.claude/.claude.json
RUN chown -R claude:claude /home/claude/.claude

# Change ownership of app directory
RUN chown -R claude:claude /app

# Set Claude config directory before switching user
ENV CLAUDE_CONFIG_DIR=/home/claude/.claude
ENV HOME=/home/claude

# Switch to non-root user
USER claude

# Expose HTTP API port
EXPOSE 9600

# Start the wrapper service
CMD ["node", "server.js"]

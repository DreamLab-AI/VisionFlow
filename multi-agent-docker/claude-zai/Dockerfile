FROM node:20-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Claude Code globally
RUN npm install -g claude-code@latest

# Create wrapper service directory
RUN mkdir -p /app/wrapper

# Copy wrapper files (will be created next)
COPY claude-zai/wrapper/ /app/wrapper/

# Install wrapper dependencies
WORKDIR /app/wrapper
RUN npm install

# Configure Claude Code for Z.AI
RUN mkdir -p /root/.claude
COPY claude-zai/claude-config.json /root/.claude/.claude.json

# Expose HTTP API port
EXPOSE 9600

# Start the wrapper service
CMD ["node", "server.js"]

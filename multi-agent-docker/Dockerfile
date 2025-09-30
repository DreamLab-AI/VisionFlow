# Stage 0 – CUDA + cuDNN (official NVIDIA image)
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS base

################################################################################
# Stage 1 – OS deps, Python 3.12 & 3.13 venvs, Rust, Node, ML stack, WasmEdge, Blender
################################################################################
ARG DEBIAN_FRONTEND=noninteractive

# 1. Set Environment Variables
# Set the application workspace directory
ENV APP_HOME="/app"
WORKDIR $APP_HOME

# 2. Install Dependencies
# Add any dependencies needed to run Blender and its Python environment.
# wget and other utilities are for downloading and extracting Blender.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    ca-certificates curl gnupg software-properties-common
# Add Docker's official GPG key
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg
# Set up the repository
RUN echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null
# Add Deadsnakes PPA for newer Python versions
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:kicad/kicad-9.0-releases
# Add NodeSource repository for up-to-date NodeJS (v22+)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
# Add Google Cloud SDK repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
# Install all packages including network utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
      wget libxi6 libxxf86vm1 libxfixes3 libxrender1 \
      build-essential clang git pkg-config libssl-dev \
      lsb-release shellcheck hyperfine openssh-client tmux sudo \
      docker-ce docker-ce-cli containerd.io unzip 7zip texlive-full latexmk chktex \
      iputils-ping netcat-openbsd net-tools dnsutils traceroute tcpdump nmap \
      iproute2 iptables curl wget telnet mtr-tiny \
      sqlite3 libsqlite3-dev \
      libgl1 libglu1-mesa libglib2.0-0 libsm6 libxext6 \
      libfontconfig1 libxkbcommon0 libxkbcommon-x11-0 libdbus-1-3 \
      supervisor \
      python3.12 python3.12-venv python3.12-dev \
      nodejs \
      jq \
      google-cloud-sdk \
      libvulkan1 vulkan-tools ocl-icd-libopencl1 && \
    rm -rf /var/lib/apt/lists/* && \
    wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 && \
    chmod +x /usr/local/bin/hadolint

# 3. Create dev user with single consolidated setup
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -g ${HOST_GID} dev && \
    useradd -m -u ${HOST_UID} -g dev -s /bin/bash dev && \
    usermod -aG sudo dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev && \
    chmod 0440 /etc/sudoers.d/dev && \
    mkdir -p /app/mcp-logs /workspace && \
    chown -R dev:dev /app/mcp-logs /workspace

# 4. Install Python and additional tool libraries (consolidated)
RUN python3.12 -m venv /opt/venv312 && \
    apt-get update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    imagemagick inkscape ffmpeg \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
    colmap libpng-dev libjpeg-dev libtiff-dev libopenexr-dev \
    kicad ngspice libngspice0 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv312/bin:$PATH"

# 5. Set up Deno and copy application files
RUN curl -fsSL https://deno.land/x/install/install.sh | sh && \
    mkdir -p /app/core-assets/scripts

ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

COPY entrypoint.sh /entrypoint.sh
COPY setup-workspace.sh /app/setup-workspace.sh
COPY mcp-helper.sh /app/mcp-helper.sh
COPY core-assets/scripts/ /app/core-assets/scripts/
RUN chmod +x /entrypoint.sh /app/setup-workspace.sh /app/mcp-helper.sh /app/core-assets/scripts/*.js /app/core-assets/scripts/*.sh

# 6. Install Node.js packages and tools (consolidated)
COPY package.json package-lock.json* ./
RUN npm cache clean --force && \
    npm install --production && \
    npm install -g . @clduab11/gemini-flow claude-flow@latest playwright@latest @executeautomation/playwright-mcp-server goalie && \
    npm cache clean --force

# Apply build-time claude-flow patches
RUN if [ -f /app/core-assets/patches/claude-flow/mcp-server.patch ]; then \
        CLAUDE_FLOW_PATH=$(npm root -g)/claude-flow && \
        if [ -d "$CLAUDE_FLOW_PATH" ]; then \
            echo "Applying claude-flow patches at build time..." && \
            cd "$CLAUDE_FLOW_PATH" && \
            patch -p0 < /app/core-assets/patches/claude-flow/mcp-server.patch && \
            echo "Build-time patching complete"; \
        fi; \
    fi

# Install Playwright browsers
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN npx playwright@latest install --with-deps chromium && \
    chmod -R 755 /opt/playwright-browsers && \
    chown -R dev:dev "$(npm config get prefix)/lib/node_modules" "$(npm config get prefix)/bin"

# 7. Install Python packages, Rust, and WasmEdge (consolidated as dev user)
COPY requirements.txt .
RUN /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 -r requirements.txt && \
    /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 --pre modular && \
    /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

USER dev
WORKDIR /home/dev
ENV HOME=/home/dev
RUN curl -fsSL https://claude.ai/install.sh | bash || true && \
    for i in 1 2 3; do curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path && break || sleep 5; done
ENV PATH="/home/dev/.cargo/bin:/home/dev/.local/bin:${PATH}"

USER root
WORKDIR $APP_HOME
RUN if [ -f /home/dev/.local/bin/claude ]; then ln -sf /home/dev/.local/bin/claude /usr/local/bin/claude; fi && \
    curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | \
    bash -s -- -p /usr/local --plugins wasi_nn-openvino && \
    ldconfig

# 8. Copy core assets and set up workspace
COPY --chown=dev:dev core-assets/ /app/core-assets/
COPY core-assets/patches /app/core-assets/patches
COPY README.md AGENT-BRIEFING.md /app/
RUN cd /app/core-assets/scripts && npm install && \
    mkdir -p /var/run/mcp /workspace && \
    chown -R dev:dev /var/run/mcp /workspace /app/core-assets && \
    git config --global user.email "agent@multi-agent-docker.com" && \
    git config --global user.name "Development Agent" && \
    chmod 2775 /workspace /app

# 9. Shell configuration and final setup
RUN touch /home/dev/.bashrc && chown dev:dev /home/dev/.bashrc && \
    echo '\n\
if [ -f "/app/core-assets/scripts/welcome-message.sh" ]; then\n\
    source "/app/core-assets/scripts/welcome-message.sh"\n\
fi\n\
export PATH="/opt/venv312/bin:/home/dev/.cargo/bin:/root/.deno/bin:/home/dev/.local/bin:/app/core-assets/scripts:/app/core-assets/mcp-tools:$PATH"\n\
alias dsp="claude --dangerously-skip-permissions"\n\
source /opt/venv312/bin/activate\n' >> /home/dev/.bashrc

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

USER dev
WORKDIR /workspace
RUN echo "/exit" | timeout 10s claude --dangerously-skip-permissions || true

USER root
RUN cd /workspace && /app/setup-workspace.sh --quiet && \
    chown -R dev:dev /workspace /app/mcp-logs

USER dev
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
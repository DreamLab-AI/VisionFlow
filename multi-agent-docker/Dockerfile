# NVIDIA CUDA base with GUI support
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

ENV APP_HOME="/app"
WORKDIR $APP_HOME

# Install base packages including VNC and GUI support
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    ca-certificates curl gnupg software-properties-common \
    xfce4 xfce4-goodies xvfb x11vnc dbus-x11 terminator xinit net-tools
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
RUN mkdir -p /usr/share/keyrings && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Install all packages including network utilities, GUI tools, browser dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
      wget libxi6 libxxf86vm1 libxfixes3 libxrender1 xz-utils \
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
      libvulkan1 vulkan-tools ocl-icd-libopencl1 \
      qgis qgis-plugin-grass \
      libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libatspi2.0-0 \
      libx11-6 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxcb1 \
      libpango-1.0-0 libcairo2 libasound2t64 fonts-liberation fonts-noto-color-emoji && \
    rm -rf /var/lib/apt/lists/* && \
    wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 && \
    chmod +x /usr/local/bin/hadolint

# 3. Create dev user with idempotent setup (handles existing UID/GID)
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN \
    # Idempotent group creation
    if getent group ${HOST_GID} >/dev/null; then \
        if [ "$(getent group ${HOST_GID} | cut -d: -f1)" != "dev" ]; then \
            groupmod -n dev "$(getent group ${HOST_GID} | cut -d: -f1)"; \
        fi; \
    else \
        groupadd -g ${HOST_GID} dev; \
    fi && \
    \
    # Idempotent user creation
    if getent passwd ${HOST_UID} >/dev/null; then \
        EXISTING_USER=$(getent passwd ${HOST_UID} | cut -d: -f1) && \
        if [ "$EXISTING_USER" != "dev" ]; then \
            usermod -l dev -d /home/dev -m "$EXISTING_USER" && \
            usermod -g dev dev; \
        fi; \
    else \
        useradd --uid ${HOST_UID} --gid ${HOST_GID} -m -s /bin/bash dev; \
    fi && \
    \
    usermod -aG sudo dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev && \
    chmod 0440 /etc/sudoers.d/dev && \
    mkdir -p /app/mcp-logs /workspace && \
    chown -R dev:dev /app/mcp-logs /workspace

# 4. Install Python, additional tool libraries, and Chrome
RUN python3.12 -m venv /opt/venv312 && \
    apt-get update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    imagemagick inkscape ffmpeg \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
    colmap libpng-dev libjpeg-dev libtiff-dev libopenexr-dev \
    kicad ngspice libngspice0 && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb && \
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
    npm install -g . @clduab11/gemini-flow claude-flow@latest ruv-swarm@latest flow-nexus@latest \
    playwright@latest @executeautomation/playwright-mcp-server chrome-devtools-mcp goalie && \
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

# Install Playwright browsers and copy GUI tool assets
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN npx playwright@latest install chromium firefox webkit && \
    chmod -R 755 /opt/playwright-browsers && \
    chown -R dev:dev "$(npm config get prefix)/lib/node_modules" "$(npm config get prefix)/bin"

# Install Blender 4.5.1
COPY gui-based-tools-docker.old/blender-4.5.1-linux-x64.tar.xz /tmp/blender-4.5.1-linux-x64.tar.xz
RUN cd /opt && \
    tar -xf /tmp/blender-4.5.1-linux-x64.tar.xz && \
    rm /tmp/blender-4.5.1-linux-x64.tar.xz && \
    ln -s /opt/blender-4.5.1-linux-x64 /opt/blender-4.5

# Install PBR Generator
COPY gui-based-tools-docker.old/tessellating-pbr-generator /opt/tessellating-pbr-generator
RUN chown -R dev:dev /opt/tessellating-pbr-generator

# 7. Install Python packages, Rust, and WasmEdge (consolidated as dev user)
COPY requirements.txt .
RUN /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 -r requirements.txt && \
    /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 --pre modular && \
    /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

USER dev
WORKDIR /home/dev
ENV HOME=/home/dev

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Switch back to root to install PBR Generator dependencies into system venv
USER root
RUN /home/dev/.local/bin/uv pip install --python /opt/venv312/bin/python -r /opt/tessellating-pbr-generator/requirements.txt

USER dev

# Install Claude CLI and Rust
RUN curl -fsSL https://claude.ai/install.sh | bash || true && \
    for i in 1 2 3; do curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path && break || sleep 5; done

# Set up Blender addon directories and QGIS plugin directory
RUN mkdir -p /home/dev/.config/blender && \
    mkdir -p /home/dev/.local/share/QGIS/QGIS3/profiles/default/python/plugins

ENV PATH="/opt/blender-4.5/blender:/home/dev/.cargo/bin:/home/dev/.local/bin:${PATH}"

USER root
WORKDIR $APP_HOME
RUN if [ -f /home/dev/.local/bin/claude ]; then ln -sf /home/dev/.local/bin/claude /usr/local/bin/claude; fi && \
    curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | \
    bash -s -- -p /usr/local --plugins wasi_nn-openvino && \
    ldconfig

# 8. Copy GUI tool scripts and core assets
COPY gui-based-tools-docker.old/addon.py /home/dev/addon.py
COPY gui-based-tools-docker.old/autostart.py /home/dev/autostart.py
COPY gui-based-tools-docker.old/playwright-mcp-server.js /opt/playwright-mcp/server.js
COPY gui-based-tools-docker.old/qgis-mcp-server.js /opt/qgis-mcp-server.js
COPY gui-based-tools-docker.old/pbr-mcp-simple.py /opt/pbr-mcp-simple.py

# Install QGIS MCP plugin
RUN git clone https://github.com/syauqi-uqi/qgis_mcp_modify1.git /tmp/qgis_mcp_modify1 && \
    cp -r /tmp/qgis_mcp_modify1/qgis_mcp_plugin /home/dev/.local/share/QGIS/QGIS3/profiles/default/python/plugins/ && \
    rm -rf /tmp/qgis_mcp_modify1

COPY --chown=dev:dev core-assets/ /app/core-assets/
COPY core-assets/patches /app/core-assets/patches
RUN cd /app/core-assets/scripts && npm install && \
    mkdir -p /var/run/mcp /workspace /opt/playwright-mcp && \
    chmod +x /opt/playwright-mcp/server.js /opt/qgis-mcp-server.js /opt/pbr-mcp-simple.py && \
    chown -R dev:dev /var/run/mcp /workspace /app/core-assets /home/dev /opt/playwright-mcp /opt/qgis-mcp-server.js /opt/pbr-mcp-simple.py && \
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
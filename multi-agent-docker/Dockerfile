# CachyOS-optimized Arch Linux with NVIDIA GPU support
FROM cachyos/cachyos:latest AS base

ENV APP_HOME="/app"
WORKDIR $APP_HOME

# Initialize pacman keyring and update system
RUN pacman-key --init && \
    pacman-key --populate archlinux && \
    pacman -Syu --noconfirm

# Install core build tools and NVIDIA GPU drivers
RUN pacman -S --noconfirm \
    base-devel git wget curl gnupg sudo cmake \
    nvidia nvidia-utils opencl-nvidia \
    xorg-server xorg-xinit xorg-xauth \
    tigervnc \
    python python-pip python-virtualenv \
    nodejs npm \
    docker docker-compose \
    sqlite jq unzip p7zip \
    shellcheck openssh tmux \
    net-tools iputils traceroute nmap tcpdump bind \
    supervisor \
    imagemagick inkscape ffmpeg \
    vulkan-tools ocl-icd \
    nss pango cairo gtk3 \
    noto-fonts ttf-liberation \
    && pacman -Scc --noconfirm && \
    systemd-machine-id-setup 2>/dev/null || dbus-uuidgen > /etc/machine-id || echo "00000000000000000000000000000000" > /etc/machine-id

# Install VirtualGL, Chrome/Chromium, and desktop environment
RUN pacman -S --noconfirm virtualgl lib32-virtualgl chromium xfce4 xfce4-goodies

# Install texlive
RUN pacman -S --noconfirm texlive-basic texlive-latex texlive-latexrecommended

# Install Google Cloud SDK
RUN cd /tmp && \
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz && \
    tar -xf google-cloud-cli-linux-x86_64.tar.gz -C /opt && \
    /opt/google-cloud-sdk/install.sh --quiet && \
    rm google-cloud-cli-linux-x86_64.tar.gz

ENV PATH="/opt/google-cloud-sdk/bin:$PATH"

# Install KiCAD and ngspice
RUN pacman -S --noconfirm kicad kicad-library kicad-library-3d ngspice

# Install QGIS
RUN pacman -S --noconfirm qgis

# Install hadolint
RUN wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 && \
    chmod +x /usr/local/bin/hadolint

# Create dev user with idempotent setup
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN \
    if getent group ${HOST_GID} >/dev/null; then \
        if [ "$(getent group ${HOST_GID} | cut -d: -f1)" != "dev" ]; then \
            groupmod -n dev "$(getent group ${HOST_GID} | cut -d: -f1)"; \
        fi; \
    else \
        groupadd -g ${HOST_GID} dev; \
    fi && \
    \
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
    usermod -aG wheel,docker dev && \
    echo "dev:password" | chpasswd && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev && \
    chmod 0440 /etc/sudoers.d/dev && \
    mkdir -p /app/mcp-logs /workspace && \
    chown -R dev:dev /app/mcp-logs /workspace

# Create Python virtual environment
RUN python -m venv /opt/venv312

ENV PATH="/opt/venv312/bin:$PATH"

# Set up Deno
RUN curl -fsSL https://deno.land/x/install/install.sh | sh

ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

# Copy application files
COPY entrypoint.sh /entrypoint.sh
COPY setup-workspace.sh /app/setup-workspace.sh
COPY mcp-helper.sh /app/mcp-helper.sh
COPY core-assets/scripts/ /app/core-assets/scripts/
COPY scripts/ /app/scripts/
RUN chmod +x /entrypoint.sh /app/setup-workspace.sh /app/mcp-helper.sh /app/core-assets/scripts/*.js /app/core-assets/scripts/*.sh /app/scripts/*.sh 2>/dev/null || true

# Install Node.js packages
COPY package.json package-lock.json* ./
RUN /opt/venv312/bin/pip install setuptools && \
    npm cache clean --force && \
    npm install --production && \
    npm install -g . @clduab11/gemini-flow claude-flow@latest ruv-swarm@latest flow-nexus@latest \
    playwright@latest @executeautomation/playwright-mcp-server chrome-devtools-mcp goalie @anthropic-ai/claude-code && \
    npm cache clean --force

# Apply claude-flow patches
RUN if [ -f /app/core-assets/patches/claude-flow/mcp-server.patch ]; then \
        CLAUDE_FLOW_PATH=$(npm root -g)/claude-flow && \
        if [ -d "$CLAUDE_FLOW_PATH" ]; then \
            echo "Applying claude-flow patches..." && \
            cd "$CLAUDE_FLOW_PATH" && \
            patch -p0 < /app/core-assets/patches/claude-flow/mcp-server.patch; \
        fi; \
    fi

# Install Playwright browsers
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN npx playwright@latest install chromium firefox webkit && \
    chmod -R 755 /opt/playwright-browsers && \
    chown -R dev:dev "$(npm config get prefix)/lib/node_modules" "$(npm config get prefix)/bin"

# Install Blender
COPY gui-tools-assets/blender-4.5.3-linux-x64.tar.xz /tmp/blender-4.5.3-linux-x64.tar.xz
RUN cd /opt && \
    tar -xf /tmp/blender-4.5.3-linux-x64.tar.xz && \
    rm /tmp/blender-4.5.3-linux-x64.tar.xz && \
    ln -s /opt/blender-4.5.3-linux-x64 /opt/blender-4.5

# Install PBR Generator
COPY gui-tools-assets/tessellating-pbr-generator /opt/tessellating-pbr-generator
RUN chown -R dev:dev /opt/tessellating-pbr-generator

# Install Python packages
COPY requirements.txt .
RUN /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 -r requirements.txt && \
    /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 --pre modular && \
    /opt/venv312/bin/pip install --no-cache-dir --retries 10 --timeout 60 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy GUI tool scripts (before USER dev so they exist for setup)
COPY gui-tools-assets/addon.py /home/dev/addon.py
COPY gui-tools-assets/autostart.py /home/dev/autostart.py
COPY gui-tools-assets/playwright-mcp-server.js /opt/playwright-mcp/server.js
COPY gui-tools-assets/qgis-mcp-server.js /opt/qgis-mcp-server.js
COPY gui-tools-assets/pbr-mcp-simple.py /opt/pbr-mcp-simple.py
COPY gui-tools-assets/web-summary-mcp-server.py /opt/web-summary-mcp-server.py
RUN chown dev:dev /home/dev/addon.py /home/dev/autostart.py

USER dev
WORKDIR /home/dev
ENV HOME=/home/dev

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install PBR Generator dependencies
USER root
RUN /home/dev/.local/bin/uv pip install --python /opt/venv312/bin/python -r /opt/tessellating-pbr-generator/requirements.txt

USER dev

# Install Rust
RUN for i in 1 2 3; do curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path && break || sleep 5; done

# Set up directories and Blender MCP addon
RUN mkdir -p /home/dev/.config/blender/4.5/scripts/addons && \
    mkdir -p /home/dev/.config/blender/4.5/scripts/startup && \
    mkdir -p /home/dev/.local/share/QGIS/QGIS3/profiles/default/python/plugins && \
    cp /home/dev/addon.py /home/dev/.config/blender/4.5/scripts/addons/blender_mcp_addon.py && \
    cp /home/dev/autostart.py /home/dev/.config/blender/4.5/scripts/startup/autostart.py && \
    echo 'import bpy\nif "blender_mcp_addon" not in bpy.context.preferences.addons:\n    bpy.ops.preferences.addon_enable(module="blender_mcp_addon")' > /home/dev/.config/blender/4.5/scripts/startup/enable_mcp.py && \
    chown -R dev:dev /home/dev/.config/blender

ENV PATH="/opt/blender-4.5:/home/dev/.cargo/bin:/home/dev/.local/bin:${PATH}"

USER root
WORKDIR $APP_HOME
RUN curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | \
    bash -s -- -p /usr/local --plugins wasi_nn-openvino && \
    ldconfig

# Install QGIS MCP plugin
RUN git clone https://github.com/syauqi-uqi/qgis_mcp_modify1.git /tmp/qgis_mcp_modify1 && \
    cp -r /tmp/qgis_mcp_modify1/qgis_mcp_plugin /home/dev/.local/share/QGIS/QGIS3/profiles/default/python/plugins/ && \
    rm -rf /tmp/qgis_mcp_modify1

COPY --chown=dev:dev core-assets/ /app/core-assets/
COPY core-assets/patches /app/core-assets/patches
RUN cd /app/core-assets/scripts && npm install && \
    mkdir -p /var/run/mcp /workspace /opt/playwright-mcp && \
    chmod +x /opt/playwright-mcp/server.js /opt/qgis-mcp-server.js /opt/pbr-mcp-simple.py /opt/web-summary-mcp-server.py && \
    chown -R dev:dev /var/run/mcp /workspace /app/core-assets /home/dev /opt/playwright-mcp /opt/qgis-mcp-server.js /opt/pbr-mcp-simple.py /opt/web-summary-mcp-server.py && \
    git config --global user.email "agent@multi-agent-docker.com" && \
    git config --global user.name "Development Agent" && \
    chmod 2775 /workspace /app

# Configure VirtualGL
RUN vglserver_config -config +s +f -t || true

# Shell configuration
RUN touch /home/dev/.bashrc && chown dev:dev /home/dev/.bashrc && \
    cat >> /home/dev/.bashrc <<'BASHRC'

if [ -f "/app/core-assets/scripts/welcome-message.sh" ]; then
    source "/app/core-assets/scripts/welcome-message.sh"
fi
export PATH="/opt/blender-4.5:/opt/venv312/bin:/home/dev/.cargo/bin:/root/.deno/bin:/home/dev/.local/bin:/app/core-assets/scripts:/app/core-assets/mcp-tools:$PATH"
alias dsp="claude --dangerously-skip-permissions"
source /opt/venv312/bin/activate
BASHRC

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install websockify for noVNC
RUN /opt/venv312/bin/pip install --no-cache-dir websockify

# Install copyparty file server
RUN /opt/venv312/bin/pip install --no-cache-dir copyparty

# Configure VNC for dev user
USER dev
RUN mkdir -p ~/.vnc && \
    echo "password" | vncpasswd -f > ~/.vnc/passwd && \
    chmod 600 ~/.vnc/passwd && \
    echo "exec startxfce4" > ~/.vnc/xstartup && \
    chmod +x ~/.vnc/xstartup

USER root
WORKDIR /workspace
RUN echo "/exit" | timeout 10s claude --dangerously-skip-permissions || true

USER root
RUN cd /workspace && /app/setup-workspace.sh --quiet && \
    chown -R dev:dev /workspace /app/mcp-logs

USER dev
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

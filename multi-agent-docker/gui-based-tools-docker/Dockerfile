# Use an NVIDIA CUDA image as the base for GPU access
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set non-interactive frontend for package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages for a lightweight desktop environment and VNC
RUN apt-get update && apt-get install -y \
    xfce4 \
    xfce4-goodies \
    xvfb \
    x11vnc \
    dbus-x11 \
    terminator \
    wget \
    python3-pip \
    xinit \
    net-tools \
    curl \
    gnupg \
    software-properties-common \
    git \
    xz-utils \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js 18+ (required for Playwright - needs modern JS syntax support)
# Using NodeSource repository for consistent version across Ubuntu releases
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    node --version && npm --version && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install browser dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libatspi2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxcb1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    fonts-liberation \
    fonts-noto-color-emoji \
    ffmpeg \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add QGIS repository and install QGIS
RUN mkdir -p /etc/apt/keyrings/
RUN wget -O /etc/apt/keyrings/qgis-archive-keyring.gpg https://download.qgis.org/downloads/qgis-archive-keyring.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/qgis.list
RUN apt-get update && apt-get install -y qgis qgis-plugin-grass

# Install Blender 4.5.1 from tar archive
COPY blender-4.5.1-linux-x64.tar.xz /tmp/blender-4.5.1-linux-x64.tar.xz
RUN cd /opt && \
    tar -xf /tmp/blender-4.5.1-linux-x64.tar.xz && \
    rm /tmp/blender-4.5.1-linux-x64.tar.xz && \
    ln -s /opt/blender-4.5.1-linux-x64 /opt/blender-4.5

# Create a non-root user for security
RUN useradd -ms /bin/bash blender

# Add sudo access for the blender user
RUN apt-get update && apt-get install -y sudo && \
    adduser blender sudo && \
    echo "blender ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create directories for Blender as root (use version-agnostic approach)
RUN mkdir -p /home/blender/.config/blender

# Copy Blender addon and startup scripts as root
# We'll create the version-specific directory and install the addon at runtime
COPY addon.py /home/blender/addon.py
COPY autostart.py /home/blender/autostart.py
COPY startup.sh /home/blender/startup.sh
COPY xstartup /home/blender/.vnc/xstartup

# Install QGIS MCP plugin
RUN mkdir -p /home/blender/.local/share/QGIS/QGIS3/profiles/default/python/plugins

RUN set -x && \
    git clone https://github.com/syauqi-uqi/qgis_mcp_modify1.git /tmp/qgis_mcp_modify1 && \
    ls -la /tmp/qgis_mcp_modify1 && \
    cp -r /tmp/qgis_mcp_modify1/qgis_mcp_plugin /home/blender/.local/share/QGIS/QGIS3/profiles/default/python/plugins/ && \
    rm -rf /tmp/qgis_mcp_modify1

# Install PBR Generator
COPY tessellating-pbr-generator /opt/tessellating-pbr-generator

# Set permissions for the blender user's home directory as root
RUN chmod +x /home/blender/startup.sh /home/blender/.vnc/xstartup
RUN chown -R blender:blender /home/blender
RUN chown -R blender:blender /opt/tessellating-pbr-generator

# Switch to the non-root user
USER blender
WORKDIR /home/blender

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install PBR Generator Python dependencies (switch to root for system-wide installation)
USER root
RUN /home/blender/.local/bin/uv pip install --system -r /opt/tessellating-pbr-generator/requirements.txt

# Install Playwright and MCP server globally
RUN npm install -g playwright@latest @executeautomation/playwright-mcp-server@latest

# Install Chromium and Chrome DevTools MCP
RUN apt-get update && apt-get install -y chromium-browser && \
    npm install -g chrome-devtools-mcp && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Playwright browser path
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers

# Install Playwright browsers with dependencies
# Note: We install system dependencies first, then browsers
RUN npx playwright@latest install-deps chromium firefox webkit && \
    npx playwright@latest install chromium firefox webkit && \
    chmod -R 755 /opt/playwright-browsers

# Create MCP scripts
RUN mkdir -p /opt/playwright-mcp /opt/qgis-mcp
COPY playwright-mcp-server.js /opt/playwright-mcp/server.js
COPY qgis-mcp-server.js /opt/qgis-mcp-server.js
COPY pbr-mcp-simple.py /opt/pbr-mcp-simple.py
RUN chmod +x /opt/playwright-mcp/server.js /opt/qgis-mcp-server.js /opt/pbr-mcp-simple.py
RUN chown -R blender:blender /opt/playwright-mcp /opt/qgis-mcp-server.js /opt/pbr-mcp-simple.py

USER blender

# Set environment variables for X server, uv, and Blender
ENV DISPLAY=:1
ENV PATH="/opt/blender-4.5/blender:/home/blender/.cargo/bin:${PATH}"

# Expose the VNC port
EXPOSE 5901

# Expose the Chrome DevTools MCP port
EXPOSE 9222

# Set the entrypoint to the startup script
ENTRYPOINT ["/home/blender/startup.sh"]
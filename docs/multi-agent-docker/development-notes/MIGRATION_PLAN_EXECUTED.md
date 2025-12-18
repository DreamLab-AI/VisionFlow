# Turbo Flow Unified CachyOS Workstation Upgrade Plan

This document describes how to:
- Sync our CachyOS based unified container with the latest turbo-flow-claude upstream features.
- Migrate from XFCE + Xvnc to a Hyprland + wayvnc Wayland desktop at 3840x2160.
- Bring up tmux, user shells, Chromium + Chrome DevTools MCP, Blender, QGIS, and all skills in a coherent startup flow.

The plan is written so it can be executed iteratively and validated via docker-compose builds and docker exec checks.

## 1. Baseline and Upstream Analysis

1. Clone the upstream project to a throwaway location (for example `/tmp/turbo-flow-claude`) and inspect the current Docker and runtime stack:
   - Main Dockerfiles and compose files.
   - Entrypoint scripts, supervisord configs, and tmux layouts.
   - MCP skills, MCP registry, and any Hyprland or new desktop related changes.

2. Compare upstream with our unified CachyOS image:
   - Our main build and runtime files:
     - 
     - 
     - 
     - 
     - 
   - Management API and Z.AI wrapper:
     - 
     - 

3. Produce a short diff oriented summary:
   - List upstream features we do not have yet (desktop, MCP, healthchecks, skills, tmux layout, multi vendor, multi skills, and whetever else it's got).
   - List our CachyOS specific additions (CUDA, multi-user, Z.AI, management API) that must be preserved.
   - Flag any breaking changes (for example Node, Claude Code, MCP config format) that affect our current image.

## 2. High-Level Architecture After Upgrade

Target state overview:

- Single unified CachyOS image providing:
  - Hyprland Wayland desktop served over wayvnc at 3840x2160.
  - A tmux `workspace` session whose windows are represented as tiled terminals on the Hyprland desktop.
  - Per user shells (devuser, gemini-user, openai-user, zai-user) ready to use.
  - Chromium with Chrome DevTools MCP running and connected to Claude Code.
  - Blender and QGIS auto started (minimized) with their MCP servers running.
  - All existing skills wired into Claude MCP configuration.
- The host `~/.claude` is mounted read write directly at `/home/devuser/.claude` and is the single source of truth for Claude Code configuration.

We will continue to drive long running services via supervisord and will keep SSH, code-server, management API, and Z.AI as in the current image.

## 3. Hyprland + wayvnc Desktop Design

### 3.1 Packages and system setup

1. Extend  to install a minimal Wayland stack:
   - hyprland, xdg-desktop-portal-hyprland, wayvnc, kitty (or alacritty), wl-clipboard, grim, slurp, swaylock or equivalent.
   - Any additional fonts required for large, high contrast rendering (for example FiraCode Nerd Font, high legibility sans fonts).
2. Keep Xorg libraries where needed for apps like Blender and QGIS but do not start Xvnc or XFCE.
3. Ensure NVIDIA or other GPU drivers work for Wayland:
   - Reuse the existing CUDA and NVIDIA env vars if compatible.
   - Verify that `/dev/dri` devices and `nvidia*` devices are present in  for Wayland compositing.

### 3.2 Wayland VNC

1. Replace the `xvnc` + `startxfce4` program pair in  with:
   - A `hyprland` supervisor program that starts a seat for `devuser` on for example `WAYLAND_DISPLAY=wayland-1` and `XDG_RUNTIME_DIR=/run/user/1000`.
   - A `wayvnc` supervisor program bound to that Wayland display, serving 3840x2160 with a strong password or token.
2. Ensure dbus and dbus user sessions are initialized before Hyprland (reuse the existing `dbus` programs but confirm ordering via supervisor priorities).
3. Keep the VNC port mapping in  and adjust comments to describe that it is now wayvnc over Wayland.

### 3.3 High contrast, large font defaults

1. Choose a terminal (kitty or alacritty) and configure:
   - Font size suitable for 4K (for example 18 to 22).
   - High contrast theme (light or dark) with clear cursor.
2. Configure Chromium system wide flags in  or in a wrapper script to use:
   - Large default zoom (for example 150 percent).
   - GPU acceleration enabled for Wayland where stable.
3. Store Hyprland config (keybindings, tiling, workspace rules) under `/home/devuser/.config/hypr` and ensure ownership is `devuser`.

## 4. Tmux, Users, and Desktop Tiling

### 4.1 Tmux session layout

1. Keep the `workspace` tmux session defined in  but extend it to:
   - Optionally add windows dedicated to `gemini-user`, `openai-user`, and `zai-user` shells (using the existing `as-*` scripts).
   - Set tmux status bar and history limit as currently configured.

2. Ensure tmux is started after dbus and before or alongside Hyprland so that terminals can attach immediately.

### 4.2 Hyprland terminal tiling

1. Design a Hyprland config that:
   - Starts N terminals on login, each running `tmux attach-session -t workspace` and selecting a different window.
   - Tiles them across the 4K desktop (for example a 2 by 2 or 3 by 3 grid depending on how many you want visible).
   - Leaves space for Chromium, Blender, and QGIS windows.
2. Add autostart scripts (for example `exec-once` entries in `hyprland.conf`) to launch:
   - `kitty --class tmux-main` for the main Claude window.
   - Additional kitty instances for Services, Logs, System etc, using `tmux select-window` to pin them.
3. Verify that when the VNC client connects, the user sees the tmux windows already attached and arranged.

### 4.3 Per user shells

1. Extend  to create extra windows that run:
   - `as-gemini`
   - `as-openai`
   - `as-zai`
   each dropping into an interactive zsh session for the target user.
2. Confirm sudoers configuration in  still allows `devuser` to `sudo` into those users without a password.
3. Optionally add shortcuts in Hyprland to switch focus directly to those tmux windows.

## 5. Browser and Chrome DevTools MCP

1. Ensure Chromium remains installed via pacman and adjust `CHROME_PATH` where needed:
   - Update Chrome DevTools MCP config under  if binary paths or flags change.
2. Confirm MCP registration in devuser `mcp_settings.json` generated by  or move this configuration fully into version controlled files.
3. Decide how Chromium should be launched on Hyprland:
   - Autostart via Hyprland `exec-once` with a persistent profile directory.
   - Optionally pre open a DevTools target page if needed for Chrome DevTools MCP.
4. Make sure the Chrome DevTools MCP supervisor program stays headless and talks to the Chromium instance via a remote debugging port.

## 6. Blender and QGIS Integration

1. Keep Blender and QGIS installed from pacman in .
2. Ensure their MCP tools are executable and correctly configured:
   - Blender MCP:  and corresponding supervisor program `blender-mcp`.
   - QGIS MCP:  and supervisor program `qgis-mcp`.
3. Update supervisord programs so that:
   - `qgis-mcp` and `blender-mcp` autostart by default.
   - Their working directories match the paths used in `mcp_settings.json`.
4. Add Hyprland autostart commands to launch Blender and QGIS GUI instances after Hyprland starts, using:
   - Command line flags or Hyprland window rules to start minimized or send them to a background workspace.
5. Validate that the MCP servers can successfully connect to the running GUI instances via their configured ports.

## 7. Claude Credentials and Host Mounts

1. Change  so that:
   - `${HOME}/.claude` is mounted read write directly at `/home/devuser/.claude`.
   - The older read only `/mnt/host-claude` mount is removed unless needed as a fallback.
2. Simplify  credential logic:
   - Stop copying from `/mnt/host-claude` when the direct mount is present.
   - Retain a safe fallback where, if `/home/devuser/.claude` is empty and `/mnt/host-claude` exists, we perform a one time copy.
3. Verify that Claude Code, Claude Flow hooks, and MCP settings all use the mounted directory as their source of truth.

## 8. Gemini and OpenAI User Tooling

1. Audit the `gemini-user` setup:
   - Confirm `gemini-flow` is installed globally and that the `gemini-flow` supervisord program uses a stable MCP socket path.
   - Verify that `GOOGLE_GEMINI_API_KEY` and related environment variables are available when that program starts.
2. Audit the `openai-user` setup:
   - Confirm OpenAI Python or CLI tooling is installed in the shared venv or via npm.
   - Ensure `OPENAI_API_KEY` and `OPENAI_ORG_ID` are wired through from  into the OpenAI config created in the entrypoint.
3. Add simple verification commands to the documentation to test each user environment (for example `gf-health` for Gemini, a simple OpenAI API curl for `openai-user`).

## 9. Skills and MCP Configuration Audit

1. Recursively audit skills under  and `/home/devuser/.claude/skills` to ensure:
   - All tools are executable (`chmod +x`) and have correct shebang lines for Python or Node.
   - Paths in `mcp_settings.json` match the on disk layout.
   - Any environment variables required by skills (for example `QGIS_HOST`, `PBR_HOST`) are defined either in the entrypoint or in supervisord programs.
2. Update MCP configuration for `devuser`:
   - Prefer generating `mcp_settings.json` from a template or checked in file instead of constructing it inline in the entrypoint where possible.
   - Ensure `chrome-devtools`, `blender`, `qgis`, `web-summary`, `imagemagick`, `kicad`, `ngspice`, `pbr-rendering`, and `playwright` are all present and enabled.

## 10. Logging, Monitoring, and Health Checks

1. Extend  with explicit log files for new programs:
   - Hyprland, wayvnc, plus any autostart wrapper scripts.
   - `blender-mcp` and `qgis-mcp` if logs are currently missing.
2. Add or extend health checks:
   - For wayvnc, consider a simple TCP port check or a small script that verifies the process is running.
   - For MCP servers, optionally add lightweight scripts that make a trivial MCP request and log the result.
3. Ensure logs are written under `/var/log` and that the existing `logs` volume remains mounted so host side inspection is easy.

## 11. Iterative Validation Workflow

1. Rebuild the image:
   - `docker compose -f docker-compose.unified.yml build agentic-workstation`
2. Start the container and attach:
   - `docker compose -f docker-compose.unified.yml up -d agentic-workstation`
   - `docker exec -it agentic-workstation zsh`
3. Validate in phases:
   - Desktop: confirm Hyprland and wayvnc are running, connect via VNC at 3840x2160 and verify tiling and fonts.
   - Tmux: `tmux attach-session -t workspace` and check all windows including per user shells.
   - MCP: from Claude Code inside the container, exercise Chrome DevTools, Blender, QGIS and other skills.
   - Services: use `supervisorctl status` and the existing Management API health checks to confirm all core services are up.
4. Iterate on configuration until logs are clean (no crash loops, clear MCP startup messages, stable desktop).

## 12. Documentation and Onboarding

1. Use this file  as the living technical plan and update it as decisions solidify.
2. Update or create user facing docs under  to cover:
   - How to connect to the Hyprland desktop via VNC.
   - How to use the tmux windows and per user shells.
   - How to trigger and debug MCP servers and skills.
3. Capture a troubleshooting checklist for common issues (VNC blank screen, MCP failing to connect, missing credentials, skill path mismatches).
4. Once the system is stable, tag the codebase or branch and note the exact upstream turbo-flow-claude commit that we are aligned with.
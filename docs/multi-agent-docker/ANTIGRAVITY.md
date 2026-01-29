---
title: Google Antigravity IDE Integration
description: The Turbo Flow Unified Container now includes the Google Antigravity IDE, accessible directly from the virtual desktop.
category: explanation
tags:
  - docker
  - ai
  - ide
related-docs:
  - multi-agent-docker/SKILLS.md
  - multi-agent-docker/TERMINAL_GRID.md
  - multi-agent-docker/comfyui-sam3d-setup.md
updated-date: 2025-12-18
difficulty-level: intermediate
---

# Google Antigravity IDE Integration

The Turbo Flow Unified Container now includes the Google Antigravity IDE, accessible directly from the virtual desktop.

## Installation Details

- **Version**: 1.11.3
- **Installation Path**: `/opt/antigravity`
- **Binary**: `/usr/bin/antigravity` (symlinked)

## How to Launch

### 1. Keyboard Shortcut (Recommended)
Press `SUPER + A` (Windows/Command key + A) anywhere in the virtual desktop to launch Antigravity.

### 2. Terminal
Run `antigravity` from any terminal window.

### 3. Application Menu
Right-click on the desktop background to access the application menu (if configured in Openbox/Hyprland menu).

## Features
- Integrated Python development environment
- Virtual desktop optimization
- Seamless integration with the container's file system

---

## Related Documentation

- [Terminal Grid Configuration](TERMINAL_GRID.md)
- [Hyprland Migration Summary](hyprland-migration-summary.md)
- [GPU Access Fix Summary](../gpu-fix-summary.md)
- [GPU-Only Build Fix Summary](fixes/SUMMARY.md)
- [Upstream Turbo-Flow-Claude Analysis](upstream-analysis.md)

## Troubleshooting
If the IDE fails to launch, check the logs or try launching from a terminal to see error output:
```bash
antigravity --verbose
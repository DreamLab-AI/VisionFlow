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

## Troubleshooting
If the IDE fails to launch, check the logs or try launching from a terminal to see error output:
```bash
antigravity --verbose
# VNC Popup Blocking Access - Quick Fixes

## 🔧 Quick Fixes to Try

### Fix 1: Kill the popup-causing processes
```bash
docker exec multi-agent-container bash -c "DISPLAY=:1 killall xfce4-notifyd Thunar"
```

### Fix 2: Open a terminal (bypass popup)
```bash
docker exec multi-agent-container bash -c "DISPLAY=:1 xfce4-terminal &"
```

### Fix 3: Reset XFCE session
```bash
docker exec multi-agent-container bash -c "
rm -rf /home/dev/.cache/sessions
rm -rf /home/dev/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-session.xml
"
docker exec multi-agent-container supervisorctl restart xfce
```

### Fix 4: Send Escape key (if popup has focus)
```bash
docker exec multi-agent-container bash -c "DISPLAY=:1 xdotool key Escape"
```

### Fix 5: Disable autostart items
```bash
docker exec multi-agent-container bash -c "
mkdir -p /home/dev/.config/autostart
cat > /home/dev/.config/autostart/thunar.desktop << 'EOF'
[Desktop Entry]
Hidden=true
EOF
"
```

## 🎯 Likely Culprits

Based on processes running:
1. **Thunar** (file manager) - PID 1753 - Often shows "Removable Media" dialog
2. **xfce4-notifyd** (notifications) - PID 1885
3. **xfce4-power-manager** - PID 1889 - Battery/power dialogs

## 📋 Alternative: Use CLI Instead

If popup persists, work from terminal:
```bash
docker exec -it multi-agent-container bash
claude --dangerously-skip-permissions
# Use Claude from CLI instead of GUI
```

## 🔍 Identify the Popup

Check X session logs:
```bash
docker exec multi-agent-container tail -100 /home/dev/.xsession-errors
```

## 🚀 Permanent Fix (for rebuild)

Add to Dockerfile:
```dockerfile
# Disable Thunar autostart
RUN mkdir -p /home/dev/.config/autostart && \
    echo "[Desktop Entry]\nHidden=true" > /home/dev/.config/autostart/thunar.desktop && \
    chown -R dev:dev /home/dev/.config
```

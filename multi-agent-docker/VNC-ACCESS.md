# VNC/GUI Access Instructions

## 🔑 Credentials

**Username:** `dev`
**Password:** `password`

## 🌐 Access Methods

### Method 1: Web Browser (noVNC) - RECOMMENDED
```
URL: http://localhost:6901
Password: password
```

### Method 2: VNC Client (TigerVNC, RealVNC, etc.)
```
Host: localhost:5901
Password: password
```

## 🔧 Troubleshooting

### Check VNC is running:
```bash
docker exec multi-agent-container supervisorctl status vnc novnc
```

Should show:
```
vnc      RUNNING
novnc    RUNNING
```

### Check ports are accessible:
```bash
# Web VNC
curl -I http://localhost:6901

# Direct VNC
nc -zv localhost 5901
```

### Reset VNC password:
```bash
docker exec -u dev multi-agent-container bash -c '
echo "password" | vncpasswd -f > ~/.vnc/passwd
chmod 600 ~/.vnc/passwd
'

docker exec multi-agent-container supervisorctl restart vnc novnc
```

### Check VNC logs:
```bash
docker exec multi-agent-container tail -50 /home/dev/.vnc/*.log
```

## 🖥️ Display Info

- **Display:** `:1`
- **Resolution:** 1600x1200
- **Depth:** 24-bit color
- **Desktop:** XFCE4

## 🔗 Ports Exposed

- `5901` - VNC Direct Connection
- `6901` - noVNC Web Interface (HTTP)

## 📝 Alternative: Use SSH with X11 Forwarding

If VNC doesn't work:
```bash
docker exec -it multi-agent-container bash
export DISPLAY=:1
xfce4-terminal &
```

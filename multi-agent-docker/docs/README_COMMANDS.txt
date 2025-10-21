╔═══════════════════════════════════════════════════════════════════════════╗
║                      CONTAINER LOGIN & COMMANDS                           ║
║                        Quick Reference Card                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

When you run: docker exec -it turbo-flow-unified zsh

DEFAULT: You login as ROOT at /home/zai-user

RECOMMENDED: Switch to devuser immediately:
  └─> su - devuser    (or use: docker exec -u devuser -it ... zsh)

───────────────────────────────────────────────────────────────────────────

ENTRYPOINT SCRIPT SETUP (9 Phases on Container Start)
═══════════════════════════════════════════════════════════════════════════

Phase 1: Directory Creation
├─ /home/devuser/workspace          (your main working directory)
├─ /home/devuser/agents             (610+ agent templates)
├─ /home/devuser/.claude/skills     (6 skills)
├─ /home/devuser/.config            (all config files)
├─ /run/user/1000                   (XDG runtime)
└─ /tmp/.X11-unix, /tmp/.ICE-unix  (X11 sockets, mode 1777)

Phase 2: Credential Distribution (from .env to user configs)
├─ ~/.config/claude/config.json     → devuser (ANTHROPIC_API_KEY)
├─ ~/.config/gemini/config.json     → gemini-user (GOOGLE_GEMINI_API_KEY)
├─ ~/.config/openai/config.json     → openai-user (OPENAI_API_KEY)
├─ ~/.config/zai/config.json        → zai-user (Z.AI config)
└─ ~/.config/gh/config.yml          → all users (GITHUB_TOKEN)

Phase 3: Host Claude Config Copy
└─ Copies /mnt/host-claude/* to ~/.claude/ if mounted

Phase 4: DBus Cleanup
└─ Removes stale PID files (managed by supervisord)

Phase 5: Claude Code Skills Setup
├─ Makes all *.py, *.js, *.sh executable in ~/.claude/skills/
└─ Reports: "6 Claude Code skills available"

Phase 6: Agent Templates Setup
└─ Reports: "610 agent templates available"

Phase 7: SSH Host Keys
└─ Generates SSH host keys if not present

Phase 8: Display Connection Banner
└─ Shows SSH, VNC, API, Z.AI connection details

Phase 9: Start Supervisord
└─ Launches all 9 services (dbus, dbus-user, sshd, xvnc, xfce4, 
   management-api, code-server, claude-zai, gemini-flow, tmux-autostart)

───────────────────────────────────────────────────────────────────────────

COMMANDS AVAILABLE (as devuser)
═══════════════════════════════════════════════════════════════════════════

CLAUDE CODE
───────────
claude                  Start Claude Code CLI
dsp                     Alias for: claude --dangerously-skip-permissions
claude-monitor          Usage monitoring (if installed)

DEVELOPMENT TOOLS
─────────────────
node, npm, npx          Node.js v24.9.0 + TypeScript, pm2, playwright
python, pip             Python 3.13 + ML stack (PyTorch, scikit-learn)
cargo, rustc, rustup    Rust stable + complete toolchain
nvcc, nvidia-smi        CUDA development environment
git, gh                 Version control + GitHub CLI

USER SWITCHING
──────────────
as-gemini               Switch to gemini-user (Google Gemini credentials)
as-openai               Switch to openai-user (OpenAI credentials)
as-zai                  Switch to zai-user (Z.AI service user)

DESKTOP APPS (via VNC)
──────────────────────
chromium, firefox       Web browsers
blender                 3D modeling and animation
qgis                    GIS operations
kicad                   PCB design and schematics
thunar                  XFCE4 file manager
xfce4-terminal          Terminal emulator

UTILITIES
─────────
tmux                    Terminal multiplexer (workspace auto-started)
vim, nano               Text editors
bat, exa                Modern cat/ls replacements
ripgrep (rg), fd        Fast grep/find
htop, btop              System monitors

───────────────────────────────────────────────────────────────────────────

ENVIRONMENT VARIABLES (devuser)
═══════════════════════════════════════════════════════════════════════════

WORKSPACE=/home/devuser/workspace
AGENTS_DIR=/home/devuser/agents
DEVPOD_WORKSPACE_FOLDER=/home/devuser/workspace
PATH includes: ~/.cargo/bin, ~/.local/bin, /opt/cuda/bin
LD_LIBRARY_PATH=/opt/cuda/lib64
DISPLAY=:1 (for GUI applications)

───────────────────────────────────────────────────────────────────────────

DIRECTORY LAYOUT
═══════════════════════════════════════════════════════════════════════════

/home/devuser/
├── workspace/              ← Your main development directory
├── agents/                 ← 610+ agent template files (*.md)
│   ├── doc-planner.md
│   ├── microtask-breakdown.md
│   └── ... (608 more)
├── .claude/
│   └── skills/            ← 6 Claude Code skills
│       ├── blender/
│       ├── imagemagick/
│       ├── kicad/
│       ├── pbr-rendering/
│       ├── qgis/
│       └── web-summary/
├── .config/
│   ├── claude/            ← Claude API config
│   ├── gh/                ← GitHub CLI config
│   └── tmux-autostart.sh  ← tmux startup script
├── .cache/                ← Cache directory
└── logs/                  ← User logs

───────────────────────────────────────────────────────────────────────────

TMUX WORKSPACE (8 windows - auto-started)
═══════════════════════════════════════════════════════════════════════════

Access: tmux attach-session -t workspace

Window 0: Claude-Main       ← Primary Claude Code workspace
Window 1: Claude-Agent      ← Agent execution and testing
Window 2: Services          ← Supervisord service monitoring
Window 3: Development       ← Python/Rust/CUDA development
Window 4: Logs              ← Service logs (split panes)
Window 5: System            ← htop resource monitoring
Window 6: VNC-Status        ← VNC connection information
Window 7: SSH-Shell         ← General purpose shell

───────────────────────────────────────────────────────────────────────────

QUICK START
═══════════════════════════════════════════════════════════════════════════

Step 1: Login as devuser
  docker exec -u devuser -it turbo-flow-unified zsh

Step 2: Navigate to workspace
  cd $WORKSPACE

Step 3: Start Claude Code
  claude

───────────────────────────────────────────────────────────────────────────

ALIASES CONFIGURED
═══════════════════════════════════════════════════════════════════════════

as-gemini               sudo -u gemini-user -i
as-openai               sudo -u openai-user -i
as-zai                  sudo -u zai-user -i
dsp                     claude --dangerously-skip-permissions
claude-monitor          ~/.cargo/bin/claude-monitor

───────────────────────────────────────────────────────────────────────────

CREDENTIAL FILES CREATED BY ENTRYPOINT
═══════════════════════════════════════════════════════════════════════════

~/.config/claude/config.json    (devuser - Anthropic API)
~/.config/gemini/config.json    (gemini-user - Google Gemini API)
~/.config/openai/config.json    (openai-user - OpenAI API)
~/.config/zai/config.json       (zai-user - Z.AI service config)
~/.config/gh/config.yml         (All users - GitHub token)

───────────────────────────────────────────────────────────────────────────

SERVICES RUNNING
═══════════════════════════════════════════════════════════════════════════

dbus (system)           System message bus
dbus-user (session)     User session bus for devuser
sshd                    SSH server (port 22 → host 2222)
xvnc                    VNC server (port 5901)
xfce4                   Full XFCE4 desktop environment
management-api          REST API (port 9090)
claude-zai              Z.AI service (port 9600, internal)
tmux-autostart          8-window workspace auto-started

───────────────────────────────────────────────────────────────────────────

FULL DOCUMENTATION
═══════════════════════════════════════════════════════════════════════════

See: CONTAINER_COMMANDS.md (514 lines - comprehensive reference)

Includes:
  • Detailed command reference for all tools
  • Service management commands
  • API endpoint documentation
  • Common workflows and examples
  • Configuration file locations
  • Troubleshooting tips

───────────────────────────────────────────────────────────────────────────

SUMMARY
═══════════════════════════════════════════════════════════════════════════

✅ Entrypoint configures: Directories, credentials, skills, agents, SSH
✅ You have access to: Claude Code, Node.js, Python, Rust, CUDA, Git
✅ User switching: as-gemini, as-openai, as-zai
✅ Desktop environment: Full XFCE4 via VNC (localhost:5901)
✅ 610+ AI agents + 6 Claude Code skills ready to use
✅ tmux workspace with 8 windows auto-started

Recommended login: docker exec -u devuser -it turbo-flow-unified zsh

═══════════════════════════════════════════════════════════════════════════

---
name: tmux-ops
version: 1.0.0
description: Terminal multiplexer for persistent background sessions - run long tasks, detach, check later
author: agentic-workstation
tags: [tmux, terminal, sessions, background-tasks, multiplexer]
mcp_server: true
---

# tmux-ops Skill

Terminal multiplexer operations for persistent background sessions. Run long-running builds, tests, or processes, detach from them, and check results later without blocking agent execution.

## Overview

tmux-ops enables non-blocking execution of long-running commands:

- **Spawn & Detach**: Start builds/tests in background sessions
- **Check Later**: Poll session output without blocking
- **Persistent**: Sessions survive disconnections
- **Multiplexed**: Multiple windows/panes per session

**Critical Use Cases**:
- Long compilation/build processes (Rust, C++, large projects)
- Extended test suites that take minutes/hours
- Background monitoring tasks
- Interactive applications that need persistent state

## Tools

### Session Management

**`session_list()`** - List all tmux sessions
```python
# Returns: [{"name": "build-session", "created": "2025-01-01", "attached": false}, ...]
```

**`session_new(name: str, command: str = None, detached: bool = True)`** - Create new session
```python
# Create session and optionally run command
session_new("rust-build", "cargo build --release", detached=True)
```

**`session_kill(name: str)`** - Terminate session
```python
session_kill("rust-build")
```

**`session_exists(name: str)`** - Check if session exists
```python
if session_exists("test-runner"):
    # Session is still alive
```

### Window Management

**`window_list(session: str)`** - List windows in session
```python
# Returns: [{"index": 0, "name": "main", "active": true}, ...]
```

**`window_new(session: str, name: str = None, command: str = None)`** - Create window
```python
window_new("dev-session", "tests", "npm test")
```

### Pane Operations

**`pane_list(session: str, window: int = None)`** - List panes in session/window
```python
# Returns: [{"pane_id": "%0", "active": true, "width": 120, "height": 40}, ...]
```

**`send_keys(target: str, keys: str, enter: bool = True)`** - Send keystrokes to pane
```python
send_keys("build-session:0.0", "cargo test", enter=True)
```

**`capture_pane(target: str, lines: int = 100, start: int = None)`** - Capture pane output
```python
output = capture_pane("build-session:0.0", lines=50)
# Returns last 50 lines of output
```

**`get_pane_pid(target: str)`** - Get process PID in pane
```python
pid = get_pane_pid("build-session:0.0")
# Check if process is still running
```

### High-Level Workflows

**`run_detached(name: str, command: str)`** - Create session, run command, return immediately
```python
# Start long-running build without blocking
run_detached("rust-build", "cargo build --release --all-features")
# Returns immediately with session name
```

**`wait_and_capture(name: str, timeout: int = 30, poll_interval: int = 2)`** - Poll until complete
```python
# Wait for build to finish (up to 30s), polling every 2s
result = wait_and_capture("rust-build", timeout=30, poll_interval=2)
# Returns: {"completed": true, "output": "...", "exit_code": 0}
```

## Patterns

### Non-Blocking Build Pattern
```python
# 1. Start build in background
run_detached("my-build", "cargo build --release")

# 2. Do other work (analyze code, run linter, etc.)
# ... agent continues with other tasks ...

# 3. Check if build finished
if session_exists("my-build"):
    output = capture_pane("my-build:0.0", lines=20)
    if "Finished release" in output:
        session_kill("my-build")
        # Build complete!
```

### Long Test Suite Pattern
```python
# Start test suite (might take 5+ minutes)
run_detached("test-suite", "npm test -- --coverage")

# Poll every 10s until complete or timeout (5 min)
result = wait_and_capture("test-suite", timeout=300, poll_interval=10)

if result["completed"]:
    print(f"Tests finished: {result['exit_code']}")
    print(result["output"])
else:
    print("Tests still running, check later")
```

### Multi-Stage Build Pattern
```python
# Create session with multiple windows for parallel builds
session_new("multi-build")
window_new("multi-build", "backend", "cd backend && npm run build")
window_new("multi-build", "frontend", "cd frontend && npm run build")
window_new("multi-build", "tests", "npm test")

# Check each window independently
for window in ["backend", "frontend", "tests"]:
    output = capture_pane(f"multi-build:{window}.0")
    print(f"{window}: {output[-5:]}")  # Last 5 lines
```

### Interactive Application Pattern
```python
# Start persistent app (server, REPL, etc.)
session_new("dev-server", "python manage.py runserver", detached=True)

# Send commands to running app
send_keys("dev-server:0.0", "reload", enter=True)

# Check server logs
logs = capture_pane("dev-server:0.0", lines=50)
```

## Examples

### Example 1: Rust Build Monitoring
```python
# Start large Rust project build
run_detached("rust-build", "cargo build --release --workspace")

# Check every 5s for 2 minutes
for _ in range(24):  # 2 min = 24 * 5s
    if not session_exists("rust-build"):
        print("Build completed!")
        break

    output = capture_pane("rust-build:0.0", lines=10)
    print(f"Building: {output[-1]}")  # Last line

    time.sleep(5)
```

### Example 2: Parallel Test Execution
```python
# Create test session with 3 parallel test runners
session_new("parallel-tests")
window_new("parallel-tests", "unit", "npm run test:unit")
window_new("parallel-tests", "integration", "npm run test:integration")
window_new("parallel-tests", "e2e", "npm run test:e2e")

# Wait for all to complete (10 min max)
start_time = time.time()
while time.time() - start_time < 600:
    windows = window_list("parallel-tests")

    all_done = True
    for win in windows:
        output = capture_pane(f"parallel-tests:{win['index']}.0", lines=5)
        if "Test run complete" not in output:
            all_done = False

    if all_done:
        print("All tests complete!")
        session_kill("parallel-tests")
        break

    time.sleep(10)
```

### Example 3: Long Docker Build
```python
# Build large Docker image (might take 20+ minutes)
run_detached("docker-build", "docker build -t myapp:latest . --no-cache")

# Check progress every 30s
result = wait_and_capture("docker-build", timeout=1800, poll_interval=30)

if result["completed"] and result["exit_code"] == 0:
    print("Docker build successful!")
    # Now safe to run: docker run myapp:latest
else:
    print(f"Build status: {result['output'][-10:]}")
```

### Example 4: Development Server with Hot Reload
```python
# Start persistent dev server
session_new("dev-env")
window_new("dev-env", "api", "cd api && npm run dev")
window_new("dev-env", "web", "cd web && npm run dev")
window_new("dev-env", "db", "docker-compose up postgres")

# Later: Send reload command to API window
send_keys("dev-env:api.0", "rs", enter=True)  # Restart server

# Check server logs
api_logs = capture_pane("dev-env:api.0", lines=30)
web_logs = capture_pane("dev-env:web.0", lines=30)
```

## Target Format

tmux uses targets to specify sessions/windows/panes:
- `session-name` - Entire session
- `session-name:window-index` - Specific window (e.g., `dev:0`)
- `session-name:window-name` - Window by name (e.g., `dev:tests`)
- `session-name:window.pane` - Specific pane (e.g., `dev:0.1`)

## Best Practices

1. **Unique Session Names**: Use descriptive names to avoid conflicts
2. **Check Existence**: Always verify session exists before operations
3. **Cleanup**: Kill sessions when done to free resources
4. **Polling Strategy**: Use reasonable intervals (2-30s) based on task duration
5. **Timeout Handling**: Set appropriate timeouts for wait_and_capture
6. **Output Capture**: Limit lines to avoid memory issues with large outputs
7. **Process Monitoring**: Use get_pane_pid to check if process is still running

## Integration with Other Skills

- **rust-development**: Non-blocking cargo builds
- **pytorch-ml**: Background model training
- **jupyter-notebooks**: Persistent kernel sessions
- **ffmpeg-processing**: Long video encoding jobs
- **latex-documents**: Large document compilation

## Notes

- tmux must be installed in environment
- Sessions persist until explicitly killed or system restart
- Each session consumes system resources (plan accordingly)
- Output capture is limited by tmux scrollback buffer (default 2000 lines)
- Use detached sessions for true non-blocking execution

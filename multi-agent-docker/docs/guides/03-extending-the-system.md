# Guide: Extending the System

This guide explains how to add a new custom MCP tool to the environment.

### The MCP Tool Protocol

An MCP tool is any executable that communicates over `stdio` (standard input/output) using JSON.

-   **Input**: The tool reads single-line JSON objects from `stdin`.
-   **Output**: For each input object, the tool writes a single-line JSON response to `stdout`.
-   **Error Handling**: Errors should be reported in the JSON response, typically under an `error` key.

### Example Python Tool Structure

Here is a basic skeleton for a Python-based MCP tool.

```python
#!/usr/bin/env python3
# Use -u flag in mcp.json for unbuffered output
import sys
import json

def process_request(request):
    # Your tool's logic goes here
    # Example:
    tool = request.get('tool')
    params = request.get('params', {})
    # ... do something with tool and params ...
    return {"status": "success", "result": f"Processed {tool} with {params}"}

def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = process_request(request)
        except Exception as e:
            response = {"error": str(e)}

        # Write the JSON response to stdout, followed by a newline
        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()

if __name__ == "__main__":
    main()
```

### Steps to Add a New Tool

Let's add a new tool called `file-lister-mcp` that lists files in a directory.

**Step 1: Create the Tool Script**

Create a new Python file in `/app/core-assets/mcp-tools/file_lister_mcp.py`.

```python
#!/usr/bin/env python3
import sys
import json
import os

def list_files(params):
    path = params.get('path', '.')
    try:
        files = os.listdir(path)
        return {"success": True, "path": path, "files": files}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = {"result": list_files(request.get('params', {}))}
        except Exception as e:
            response = {"error": str(e)}

        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()

if __name__ == "__main__":
    main()
```

**Step 2: Make the Script Executable**

Inside the container, or by changing permissions on your host, make the new script executable.

```bash
chmod +x /app/core-assets/mcp-tools/file_lister_mcp.py
```

**Step 3: Register the Tool in `mcp.json`**

Edit `/app/core-assets/mcp.json` and add a new entry for your tool.

```json
{
  "mcpServers": {
    "...": "...",
    "file-lister-mcp": {
      "command": "python3",
      "args": ["-u", "./mcp-tools/file_lister_mcp.py"],
      "type": "stdio"
    },
    "...": "..."
  }
}
```
**Note**: The `-u` flag for Python is important. It ensures that the output is unbuffered, which is necessary for `stdio` communication with `claude-flow`.

**Step 4: Update the Workspace**

The tools and configurations in `/app/core-assets` are the source of truth. To make your new tool available in the active workspace, you need to sync the changes.

From inside the `multi-agent-container`:
```bash
/app/setup-workspace.sh --force
```
The `--force` flag ensures that existing files are overwritten with the new versions from `/app/core-assets`.

**Step 5: Test Your New Tool**

Use the `mcp-helper.sh` script to verify that your tool is registered and working.

```bash
# Check if the tool is listed
./mcp-helper.sh list-tools

# Run the tool
./mcp-helper.sh run-tool file-lister-mcp '{"params": {"path": "/app"}}'
```

You have now successfully extended the environment with a new MCP tool!

### Advanced Tool Development

#### Node.js Tools

For JavaScript/TypeScript tools, the structure is similar:

```javascript
#!/usr/bin/env node

const readline = require('readline');

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

rl.on('line', (line) => {
    try {
        const request = JSON.parse(line);
        const response = processRequest(request);
        console.log(JSON.stringify(response));
    } catch (error) {
        console.log(JSON.stringify({ error: error.message }));
    }
});

function processRequest(request) {
    const { tool, params } = request;
    // Your tool logic here
    return { status: 'success', result: 'processed' };
}
```

#### Best Practices

1. **Error Handling**: Always wrap your main logic in try-catch blocks
2. **Validation**: Validate input parameters before processing
3. **Logging**: Write debug logs to stderr, not stdout
4. **Buffering**: Use unbuffered output for real-time communication
5. **Documentation**: Include clear documentation of your tool's API

#### Tool Configuration Options

The `mcp.json` supports various configuration options:

```json
{
  "mcpServers": {
    "my-advanced-tool": {
      "command": "python3",
      "args": ["-u", "./mcp-tools/my_tool.py"],
      "type": "stdio",
      "env": {
        "TOOL_CONFIG": "/workspace/config/tool.json",
        "DEBUG": "true"
      },
      "cwd": "/workspace",
      "timeout": 30000
    }
  }
}
```

- `env`: Environment variables specific to this tool
- `cwd`: Working directory for the tool process
- `timeout`: Maximum execution time in milliseconds

### Integration with the Automated Setup

To ensure your custom tools are automatically set up:

1. Add tool dependencies to `/app/core-assets/requirements.txt` (Python) or `package.json` (Node.js)
2. Update `/app/core-assets/scripts/automated-setup.sh` if special setup is needed
3. Document your tool in `/app/core-assets/docs/custom-tools.md`

The automated setup process will handle installing dependencies and configuring your tools on container startup.
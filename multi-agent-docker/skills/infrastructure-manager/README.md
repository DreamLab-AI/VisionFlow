# Infrastructure Manager Skill

Unified Infrastructure as Code management with Ansible, Terraform, and Pulumi.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test the server
python3 server.py

# Add to Claude Desktop
cat mcp-config.json >> ~/.config/claude-desktop/config.json
```

## Features

### Ansible
- Playbook execution with check mode
- Inventory management
- Host connectivity testing
- Facts gathering
- Playbook linting

### Terraform
- Infrastructure planning and applying
- State management
- Output inspection
- Configuration validation
- Formatting checks

### Safety Controls
- Check/dry-run modes enabled by default
- Secret scanning before operations
- Approval gates for destructive operations
- Pattern-based secret detection

## Usage Examples

### Ansible Deployment
```javascript
// Test connectivity
mcp__infrastructure-manager__ansible_ping({
  inventory: "hosts.ini",
  pattern: "webservers"
})

// Dry-run deployment
mcp__infrastructure-manager__ansible_playbook({
  playbook: "deploy.yml",
  inventory: "hosts.ini",
  check: true
})
```

### Terraform Workflow
```javascript
// Initialize
mcp__infrastructure-manager__tf_init({ path: "./infra" })

// Plan changes
mcp__infrastructure-manager__tf_plan({
  path: "./infra",
  out: "plan.tfplan"
})

// Apply changes
mcp__infrastructure-manager__tf_apply({
  path: "./infra",
  plan: "plan.tfplan"
})
```

### Secret Scanning
```javascript
// Scan for secrets
mcp__infrastructure-manager__secret_scan({ path: "./infra" })
```

## Requirements

- Python 3.8+
- ansible >= 2.9
- terraform >= 1.0
- ansible-lint (optional)

## Safety Features

1. **Check Modes**: Ansible check mode enabled by default
2. **Approval Gates**: Terraform apply/destroy require confirmation
3. **Secret Scanning**: Pattern-based secret detection
4. **Validation**: Configuration validation before operations
5. **State Protection**: Safe state management operations

## Configuration

The MCP server runs as a subprocess managed by Claude Desktop. Configuration is in `mcp-config.json`.

## Troubleshooting

### Command Not Found
Ensure Ansible/Terraform are in PATH:
```bash
which ansible
which terraform
```

### Permission Issues
Check file permissions and SSH keys for Ansible:
```bash
ansible all -m ping -i hosts.ini -vvv
```

### State Lock Issues
For Terraform state locks:
```bash
terraform force-unlock <lock-id>
```

## Best Practices

1. Always scan for secrets before commits
2. Use check/dry-run modes for validation
3. Save Terraform plans for review
4. Use variable files instead of inline vars
5. Enable state locking for Terraform
6. Use Ansible Vault for secrets
7. Separate environments with different configs

## Development

```bash
# Run tests
python3 -m pytest tests/

# Lint code
pylint server.py

# Format code
black server.py
```

## License

MIT License - see LICENSE file for details

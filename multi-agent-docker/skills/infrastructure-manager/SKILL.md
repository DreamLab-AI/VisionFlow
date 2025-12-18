---
name: infrastructure-manager
version: 1.0.0
description: Infrastructure as Code - Ansible playbooks, Terraform plans, Pulumi programs, unified IaC management with safety controls
author: agentic-workstation
tags: [ansible, terraform, pulumi, iac, infrastructure, devops, automation, cloud]
mcp_server: true
runtime: python
---

# Infrastructure Manager

Unified Infrastructure as Code (IaC) management with Ansible, Terraform, and Pulumi support. Execute infrastructure operations with built-in safety controls and secret scanning.

## Overview

Manage infrastructure declaratively across multiple tools:

- **Ansible**: Configuration management, provisioning, orchestration
- **Terraform**: Infrastructure provisioning with state management
- **Pulumi**: Modern IaC with programming languages
- **Safety**: Check modes, dry-runs, secret scanning, approval gates

## Ansible Tools

### ansible_ping
Test connectivity to inventory hosts.

```javascript
mcp__infrastructure-manager__ansible_ping({
  inventory: "/path/to/inventory.ini",
  pattern: "webservers"  // optional, default "all"
})
```

### ansible_playbook
Execute Ansible playbooks with safety controls.

```javascript
mcp__infrastructure-manager__ansible_playbook({
  playbook: "deploy.yml",
  inventory: "inventory.ini",
  check: true,        // Check mode (dry-run), default true
  diff: true,         // Show diffs, default true
  extra_vars: {       // optional
    app_version: "2.1.0",
    env: "production"
  }
})
```

**Safety**: Check mode enabled by default for dry-run validation.

### ansible_inventory
Display inventory structure and hosts.

```javascript
mcp__infrastructure-manager__ansible_inventory({
  inventory: "inventory.ini",
  graph: false  // true for visual graph
})
```

### ansible_facts
Gather system facts from a host.

```javascript
mcp__infrastructure-manager__ansible_facts({
  inventory: "inventory.ini",
  host: "web01.example.com"
})
```

### ansible_lint
Lint Ansible playbooks for best practices.

```javascript
mcp__infrastructure-manager__ansible_lint({
  playbook: "playbook.yml"
})
```

## Terraform Tools

### tf_init
Initialize Terraform working directory.

```javascript
mcp__infrastructure-manager__tf_init({
  path: "/path/to/terraform/project"
})
```

### tf_plan
Generate and show execution plan.

```javascript
mcp__infrastructure-manager__tf_plan({
  path: "./infrastructure",
  out: "plan.tfplan",  // optional, save plan
  var_file: "prod.tfvars"  // optional
})
```

### tf_apply
Apply Terraform changes.

```javascript
mcp__infrastructure-manager__tf_apply({
  path: "./infrastructure",
  plan: "plan.tfplan",  // optional, use saved plan
  auto_approve: false   // default false, requires confirmation
})
```

**Safety**: Auto-approve disabled by default, requires manual confirmation.

### tf_destroy
Destroy Terraform-managed infrastructure.

```javascript
mcp__infrastructure-manager__tf_destroy({
  path: "./infrastructure",
  auto_approve: false  // default false, requires confirmation
})
```

**Safety**: Auto-approve disabled by default, requires manual confirmation.

### tf_state_list
List resources in Terraform state.

```javascript
mcp__infrastructure-manager__tf_state_list({
  path: "./infrastructure"
})
```

### tf_output
Show Terraform output values.

```javascript
mcp__infrastructure-manager__tf_output({
  path: "./infrastructure",
  name: "load_balancer_ip"  // optional, specific output
})
```

### tf_validate
Validate Terraform configuration.

```javascript
mcp__infrastructure-manager__tf_validate({
  path: "./infrastructure"
})
```

### tf_fmt
Format Terraform files.

```javascript
mcp__infrastructure-manager__tf_fmt({
  path: "./infrastructure",
  check: true  // default true, check only without modifying
})
```

## General Tools

### iac_detect
Detect IaC tool type in directory.

```javascript
mcp__infrastructure-manager__iac_detect({
  path: "./infrastructure"
})
```

Returns detected tool (ansible, terraform, pulumi, cloudformation) and relevant files.

### secret_scan
Scan for exposed secrets using pattern matching.

```javascript
mcp__infrastructure-manager__secret_scan({
  path: "./infrastructure"
})
```

Detects:
- AWS keys, secrets
- API tokens
- Private keys
- Database credentials
- Generic secrets in config files

## Examples

### Ansible Deployment

```javascript
// 1. Test connectivity
mcp__infrastructure-manager__ansible_ping({
  inventory: "production.ini",
  pattern: "webservers"
})

// 2. Lint playbook
mcp__infrastructure-manager__ansible_lint({
  playbook: "deploy.yml"
})

// 3. Dry-run deployment
mcp__infrastructure-manager__ansible_playbook({
  playbook: "deploy.yml",
  inventory: "production.ini",
  check: true,
  diff: true
})

// 4. Execute actual deployment
mcp__infrastructure-manager__ansible_playbook({
  playbook: "deploy.yml",
  inventory: "production.ini",
  check: false,
  extra_vars: { version: "2.1.0" }
})
```

### Terraform Workflow

```javascript
// 1. Detect IaC type
mcp__infrastructure-manager__iac_detect({
  path: "./infra"
})

// 2. Scan for secrets
mcp__infrastructure-manager__secret_scan({
  path: "./infra"
})

// 3. Initialize
mcp__infrastructure-manager__tf_init({
  path: "./infra"
})

// 4. Validate configuration
mcp__infrastructure-manager__tf_validate({
  path: "./infra"
})

// 5. Format check
mcp__infrastructure-manager__tf_fmt({
  path: "./infra",
  check: true
})

// 6. Plan changes
mcp__infrastructure-manager__tf_plan({
  path: "./infra",
  out: "production.tfplan",
  var_file: "prod.tfvars"
})

// 7. Apply changes (requires confirmation)
mcp__infrastructure-manager__tf_apply({
  path: "./infra",
  plan: "production.tfplan"
})

// 8. View outputs
mcp__infrastructure-manager__tf_output({
  path: "./infra"
})
```

### Multi-Environment Management

```javascript
// Development environment
mcp__infrastructure-manager__tf_plan({
  path: "./infra",
  var_file: "dev.tfvars"
})

// Staging environment
mcp__infrastructure-manager__tf_plan({
  path: "./infra",
  var_file: "staging.tfvars"
})

// Production environment (with saved plan)
mcp__infrastructure-manager__tf_plan({
  path: "./infra",
  var_file: "prod.tfvars",
  out: "prod.tfplan"
})
```

## Safety Features

### Check/Dry-Run Modes
- **Ansible**: Check mode (`--check`) enabled by default
- **Terraform**: Plans show changes before apply
- **Approval Gates**: Auto-approve disabled for destructive operations

### Secret Scanning
- Detects hardcoded credentials before execution
- Prevents accidental secret commits
- Pattern-based detection for common secret types

### State Management
- Terraform state inspection tools
- Non-destructive operations by default
- Clear separation of plan and apply phases

## Best Practices

1. **Always scan for secrets** before operations
2. **Use check/dry-run modes** for validation
3. **Save Terraform plans** for review before apply
4. **Use variable files** instead of inline vars
5. **Lint playbooks** before execution
6. **Test inventory connectivity** before deployments
7. **Review diffs carefully** in check mode
8. **Use version control** for all IaC files
9. **Separate environments** with different var files
10. **Enable state locking** for Terraform (S3/DynamoDB)

## Common Patterns

### Pre-Flight Checks
```javascript
// 1. Secret scan
secret_scan({ path: "." })

// 2. IaC detection
iac_detect({ path: "." })

// 3. Validation
tf_validate({ path: "." }) // or ansible_lint()

// 4. Connectivity
ansible_ping({ inventory: "hosts.ini" })
```

### Safe Deployment
```javascript
// 1. Dry-run
ansible_playbook({ playbook: "deploy.yml", check: true })

// 2. Review output
// 3. Execute
ansible_playbook({ playbook: "deploy.yml", check: false })
```

### Infrastructure Changes
```javascript
// 1. Format check
tf_fmt({ path: ".", check: true })

// 2. Plan
tf_plan({ path: ".", out: "plan.tfplan" })

// 3. Review plan
// 4. Apply
tf_apply({ path: ".", plan: "plan.tfplan" })

// 5. Verify outputs
tf_output({ path: "." })
```

## Error Handling

All tools return structured output:
```json
{
  "success": true,
  "output": "command output",
  "exit_code": 0,
  "warnings": [],
  "secrets_found": 0
}
```

Failed operations return:
```json
{
  "success": false,
  "error": "error message",
  "exit_code": 1,
  "stderr": "detailed error"
}
```

## Integration

### With CI/CD
```yaml
# Example GitLab CI
infrastructure-deploy:
  script:
    - claude-code --skill infrastructure-manager --mcp
    - # MCP calls for validation and deployment
```

### With Other Skills
- **docker-compose**: Deploy containerized infrastructure
- **kubernetes-manager**: K8s resource management
- **cloud-platforms**: Cloud-native IaC operations

## Requirements

- ansible >= 2.9
- terraform >= 1.0
- pulumi >= 3.0 (optional)
- ansible-lint (optional, for linting)
- Python 3.8+

## Notes

- **State Files**: Terraform state files may contain sensitive data
- **Inventory Security**: Protect inventory files with SSH keys
- **Remote State**: Use S3/GCS/Azure for Terraform remote state
- **Vault Integration**: Use Ansible Vault for encrypted secrets
- **Workspace Isolation**: Use Terraform workspaces for environments
- **Idempotency**: Ansible and Terraform are idempotent by design

## Troubleshooting

### Ansible
- Check SSH connectivity with `ansible_ping`
- Review inventory with `ansible_inventory --graph`
- Use `ansible_facts` to gather host information
- Enable verbose mode in playbook execution

### Terraform
- Run `tf_init` if providers are missing
- Use `tf_validate` to check configuration syntax
- Review state with `tf_state_list`
- Check for state lock issues

### Secrets
- Run `secret_scan` before commits
- Use environment variables for sensitive values
- Enable Ansible Vault for encrypted vars
- Use Terraform variables with `.tfvars` files

---

**Safety First**: This skill defaults to safe operations (check modes, approval gates, secret scanning). Always review plans and diffs before applying changes to production infrastructure.

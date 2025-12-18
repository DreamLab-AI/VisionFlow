#!/usr/bin/env python3
"""
Infrastructure Manager MCP Server
Unified Ansible, Terraform, and Pulumi IaC management with safety controls
"""

import os
import json
import subprocess
import re
from pathlib import Path
from typing import Optional, Dict, List, Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("infrastructure-manager")

# Secret detection patterns
SECRET_PATTERNS = [
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
    (r'(?i)aws_secret_access_key\s*=\s*["\']?([A-Za-z0-9/+=]{40})["\']?', 'AWS Secret Key'),
    (r'-----BEGIN (?:RSA |DSA )?PRIVATE KEY-----', 'Private Key'),
    (r'(?i)password\s*[:=]\s*["\']([^"\']{8,})["\']', 'Password'),
    (r'(?i)api[_-]?key\s*[:=]\s*["\']([^"\']{16,})["\']', 'API Key'),
    (r'(?i)secret\s*[:=]\s*["\']([^"\']{16,})["\']', 'Secret'),
    (r'(?i)token\s*[:=]\s*["\']([^"\']{16,})["\']', 'Token'),
    (r'mongodb(\+srv)?://[^\s:]+:[^\s@]+@', 'MongoDB Connection String'),
    (r'postgres://[^\s:]+:[^\s@]+@', 'PostgreSQL Connection String'),
]

def run_command(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict] = None) -> Dict[str, Any]:
    """Execute command and return structured result"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=300
        )

        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Command timeout after 5 minutes',
            'exit_code': -1
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'exit_code': -1
        }

def scan_for_secrets(path: str) -> List[Dict[str, Any]]:
    """Scan files for potential secrets"""
    secrets = []
    path_obj = Path(path)

    if not path_obj.exists():
        return []

    files_to_scan = []
    if path_obj.is_file():
        files_to_scan = [path_obj]
    else:
        # Scan common IaC and config files
        extensions = ['.yml', '.yaml', '.tf', '.tfvars', '.py', '.json', '.ini', '.env', '.conf']
        for ext in extensions:
            files_to_scan.extend(path_obj.rglob(f'*{ext}'))

    for file_path in files_to_scan:
        try:
            content = file_path.read_text()
            for pattern, secret_type in SECRET_PATTERNS:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1
                    secrets.append({
                        'file': str(file_path),
                        'line': line_num,
                        'type': secret_type,
                        'pattern': pattern
                    })
        except Exception:
            continue

    return secrets

# ============================================================================
# ANSIBLE TOOLS
# ============================================================================

@mcp.tool()
def ansible_ping(inventory: str, pattern: str = "all") -> Dict[str, Any]:
    """
    Test connectivity to Ansible inventory hosts.

    Args:
        inventory: Path to inventory file or directory
        pattern: Host pattern to ping (default: all)

    Returns:
        Dict with ping results and host status
    """
    cmd = ['ansible', pattern, '-i', inventory, '-m', 'ping']
    result = run_command(cmd)

    if result['success']:
        # Parse ping results
        lines = result['output'].split('\n')
        hosts_up = len([l for l in lines if 'SUCCESS' in l])
        hosts_unreachable = len([l for l in lines if 'UNREACHABLE' in l])

        result['hosts_up'] = hosts_up
        result['hosts_unreachable'] = hosts_unreachable

    return result

@mcp.tool()
def ansible_playbook(
    playbook: str,
    inventory: str,
    check: bool = True,
    diff: bool = True,
    extra_vars: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute Ansible playbook with safety controls.

    Args:
        playbook: Path to playbook file
        inventory: Path to inventory file
        check: Enable check mode (dry-run), default True
        diff: Show diffs, default True
        extra_vars: Extra variables as dictionary

    Returns:
        Dict with playbook execution results
    """
    cmd = ['ansible-playbook', playbook, '-i', inventory]

    if check:
        cmd.append('--check')
    if diff:
        cmd.append('--diff')
    if extra_vars:
        cmd.extend(['--extra-vars', json.dumps(extra_vars)])

    result = run_command(cmd)

    if result['success']:
        # Parse playbook results
        output = result['output']
        result['changed'] = output.count('changed=')
        result['ok'] = output.count('ok=')
        result['failed'] = output.count('failed=')
        result['check_mode'] = check

    return result

@mcp.tool()
def ansible_inventory(inventory: str, graph: bool = False) -> Dict[str, Any]:
    """
    Display Ansible inventory structure.

    Args:
        inventory: Path to inventory file or directory
        graph: Show as visual graph (default: JSON list)

    Returns:
        Dict with inventory structure
    """
    cmd = ['ansible-inventory', '-i', inventory]
    cmd.append('--graph' if graph else '--list')

    result = run_command(cmd)

    if result['success'] and not graph:
        try:
            result['inventory_data'] = json.loads(result['output'])
        except json.JSONDecodeError:
            pass

    return result

@mcp.tool()
def ansible_facts(inventory: str, host: str) -> Dict[str, Any]:
    """
    Gather system facts from an Ansible host.

    Args:
        inventory: Path to inventory file
        host: Hostname or IP address

    Returns:
        Dict with gathered facts
    """
    cmd = ['ansible', host, '-i', inventory, '-m', 'setup']
    result = run_command(cmd)

    if result['success']:
        try:
            # Parse facts from output
            output = result['output']
            # Extract JSON from ansible output
            json_start = output.find('{')
            if json_start != -1:
                facts_json = output[json_start:]
                result['facts'] = json.loads(facts_json)
        except Exception:
            pass

    return result

@mcp.tool()
def ansible_lint(playbook: str) -> Dict[str, Any]:
    """
    Lint Ansible playbook for best practices.

    Args:
        playbook: Path to playbook file

    Returns:
        Dict with linting results
    """
    cmd = ['ansible-lint', playbook]
    result = run_command(cmd)

    # ansible-lint returns non-zero on warnings, but that's not an error
    if not result['success'] and result['exit_code'] == 2:
        result['warnings'] = result['output']
        result['success'] = True

    return result

# ============================================================================
# TERRAFORM TOOLS
# ============================================================================

@mcp.tool()
def tf_init(path: str) -> Dict[str, Any]:
    """
    Initialize Terraform working directory.

    Args:
        path: Path to Terraform project directory

    Returns:
        Dict with initialization results
    """
    cmd = ['terraform', 'init']
    result = run_command(cmd, cwd=path)

    if result['success']:
        result['initialized'] = 'Terraform has been successfully initialized' in result['output']

    return result

@mcp.tool()
def tf_plan(
    path: str,
    out: Optional[str] = None,
    var_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate Terraform execution plan.

    Args:
        path: Path to Terraform project directory
        out: Save plan to file
        var_file: Variable file to use

    Returns:
        Dict with plan details
    """
    cmd = ['terraform', 'plan']

    if out:
        cmd.extend(['-out', out])
    if var_file:
        cmd.extend(['-var-file', var_file])

    result = run_command(cmd, cwd=path)

    if result['success']:
        output = result['output']
        result['changes'] = {
            'add': output.count(' will be created'),
            'change': output.count(' will be updated'),
            'destroy': output.count(' will be destroyed')
        }

    return result

@mcp.tool()
def tf_apply(
    path: str,
    plan: Optional[str] = None,
    auto_approve: bool = False
) -> Dict[str, Any]:
    """
    Apply Terraform changes.

    Args:
        path: Path to Terraform project directory
        plan: Use saved plan file
        auto_approve: Skip interactive approval (default: False for safety)

    Returns:
        Dict with apply results
    """
    cmd = ['terraform', 'apply']

    if plan:
        cmd.append(plan)
    if auto_approve:
        cmd.append('-auto-approve')

    result = run_command(cmd, cwd=path)

    if result['success']:
        output = result['output']
        result['resources'] = {
            'added': output.count('created'),
            'changed': output.count('updated'),
            'destroyed': output.count('destroyed')
        }

    return result

@mcp.tool()
def tf_destroy(path: str, auto_approve: bool = False) -> Dict[str, Any]:
    """
    Destroy Terraform-managed infrastructure.

    Args:
        path: Path to Terraform project directory
        auto_approve: Skip interactive approval (default: False for safety)

    Returns:
        Dict with destroy results
    """
    cmd = ['terraform', 'destroy']

    if auto_approve:
        cmd.append('-auto-approve')

    result = run_command(cmd, cwd=path)

    if result['success']:
        result['destroyed'] = 'Destroy complete!' in result['output']

    return result

@mcp.tool()
def tf_state_list(path: str) -> Dict[str, Any]:
    """
    List resources in Terraform state.

    Args:
        path: Path to Terraform project directory

    Returns:
        Dict with state resources
    """
    cmd = ['terraform', 'state', 'list']
    result = run_command(cmd, cwd=path)

    if result['success']:
        result['resources'] = [r.strip() for r in result['output'].split('\n') if r.strip()]

    return result

@mcp.tool()
def tf_output(path: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Show Terraform output values.

    Args:
        path: Path to Terraform project directory
        name: Specific output name (optional)

    Returns:
        Dict with output values
    """
    cmd = ['terraform', 'output', '-json']
    if name:
        cmd.append(name)

    result = run_command(cmd, cwd=path)

    if result['success']:
        try:
            result['outputs'] = json.loads(result['output'])
        except json.JSONDecodeError:
            pass

    return result

@mcp.tool()
def tf_validate(path: str) -> Dict[str, Any]:
    """
    Validate Terraform configuration.

    Args:
        path: Path to Terraform project directory

    Returns:
        Dict with validation results
    """
    cmd = ['terraform', 'validate', '-json']
    result = run_command(cmd, cwd=path)

    if result['success']:
        try:
            validation = json.loads(result['output'])
            result['valid'] = validation.get('valid', False)
            result['diagnostics'] = validation.get('diagnostics', [])
        except json.JSONDecodeError:
            pass

    return result

@mcp.tool()
def tf_fmt(path: str, check: bool = True) -> Dict[str, Any]:
    """
    Format Terraform configuration files.

    Args:
        path: Path to Terraform project directory
        check: Check only without modifying (default: True)

    Returns:
        Dict with format results
    """
    cmd = ['terraform', 'fmt', '-recursive']
    if check:
        cmd.append('-check')

    result = run_command(cmd, cwd=path)

    if result['success']:
        result['files_formatted'] = [f.strip() for f in result['output'].split('\n') if f.strip()]

    return result

# ============================================================================
# GENERAL TOOLS
# ============================================================================

@mcp.tool()
def iac_detect(path: str) -> Dict[str, Any]:
    """
    Detect Infrastructure as Code tool type in directory.

    Args:
        path: Path to scan for IaC files

    Returns:
        Dict with detected IaC types and files
    """
    path_obj = Path(path)
    detected = {
        'terraform': [],
        'ansible': [],
        'pulumi': [],
        'cloudformation': []
    }

    if not path_obj.exists():
        return {'success': False, 'error': 'Path does not exist'}

    # Terraform
    detected['terraform'] = list(path_obj.rglob('*.tf'))

    # Ansible
    ansible_files = list(path_obj.rglob('*.yml')) + list(path_obj.rglob('*.yaml'))
    for f in ansible_files:
        content = f.read_text()
        if 'hosts:' in content or 'tasks:' in content or 'roles:' in content:
            detected['ansible'].append(f)

    # Pulumi
    detected['pulumi'] = list(path_obj.rglob('Pulumi.yaml'))

    # CloudFormation
    for f in ansible_files:  # Check YAML files
        content = f.read_text()
        if 'AWSTemplateFormatVersion' in content or 'Resources:' in content:
            detected['cloudformation'].append(f)

    # Convert Path objects to strings
    for key in detected:
        detected[key] = [str(f) for f in detected[key]]

    # Determine primary tool
    primary = None
    for tool, files in detected.items():
        if files:
            primary = tool
            break

    return {
        'success': True,
        'primary_tool': primary,
        'detected': detected
    }

@mcp.tool()
def secret_scan(path: str) -> Dict[str, Any]:
    """
    Scan for exposed secrets in IaC files.

    Args:
        path: Path to scan (file or directory)

    Returns:
        Dict with found secrets and locations
    """
    secrets = scan_for_secrets(path)

    return {
        'success': True,
        'secrets_found': len(secrets),
        'secrets': secrets,
        'warning': 'Found potential secrets! Review before committing.' if secrets else None
    }

if __name__ == "__main__":
    mcp.run()

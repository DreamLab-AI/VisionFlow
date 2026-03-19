---
name: "Skill Display Name"
description: "Brief description of what this skill does. Use when [specific condition or trigger]."
version: "0.1.0"
status: "experimental"
tags: [primary-tag, secondary-tag]
depends_on: []
supersedes: []
---

# Skill Display Name

One to three sentences summarising the skill. State the core capability and the
primary use case. Avoid marketing language; be specific and verifiable.

## Prerequisites

- **Tool Name** >= X.Y.Z -- `tool --version` to verify
- **Service Name** running on `localhost:PORT` -- `curl http://localhost:PORT/health` to verify
- **Configuration**: `~/.config/tool/config.yaml` must exist with `key: value`

Verify all prerequisites before proceeding:

```bash
tool --version    # Expected: X.Y.Z or higher
curl -s http://localhost:PORT/health | jq .status  # Expected: "ok"
```

## Quick Start

The simplest working invocation. A new user should be able to copy-paste this
block and see a result.

```bash
# Step 1: Initialise
tool init --project my-project

# Step 2: Run the primary command
tool run --input example.txt --output result.txt

# Step 3: Verify output
cat result.txt
# Expected output:
# Processing complete. 42 items processed.
```

## Usage

### Basic Commands

| Command | Description |
|---------|-------------|
| `tool init` | Create a new project |
| `tool run` | Execute the primary workflow |
| `tool status` | Check current state |
| `tool clean` | Remove generated artefacts |

### Flags and Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | (required) | Input file or directory |
| `--output PATH` | `./output` | Output destination |
| `--verbose` | `false` | Enable detailed logging |
| `--format FORMAT` | `json` | Output format: `json`, `csv`, `text` |

### Examples

#### Process a single file

```bash
tool run --input data.csv --output results.json --format json
```

#### Batch processing

```bash
tool run --input ./data/ --output ./results/ --verbose
```

## When to Use

- Processing structured data files that conform to the X format
- Integrating with the Y pipeline as a transformation step
- Generating reports from raw Z telemetry data
- Automating repetitive Q tasks during development

## When Not to Use

- For real-time streaming data -- use the `stream-chain` skill instead
- For binary file processing -- use the `ffmpeg-processing` or `imagemagick`
  skill depending on the file type
- When the input exceeds 10 GB -- use the batch-processing variant documented
  in `docs/large-scale.md`

## Troubleshooting

### "Connection refused" on startup

**Cause**: The prerequisite service is not running.

**Fix**:
```bash
# Start the service
sudo systemctl start tool-service

# Verify
curl -s http://localhost:PORT/health
```

### "Invalid input format" error

**Cause**: Input file does not match the expected schema.

**Fix**: Validate the input first:
```bash
tool validate --input data.csv
# Follow the error messages to correct the file
```

### Command hangs with no output

**Cause**: Large input without `--verbose` gives no progress indication.

**Fix**: Re-run with `--verbose` to see progress:
```bash
tool run --input large-data/ --output results/ --verbose
```

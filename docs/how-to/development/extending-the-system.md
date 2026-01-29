---
title: Extending the System
description: > [Guides](./index.md) > Extending the System
category: how-to
tags:
  - tutorial
  - api
  - api
  - docker
  - backend
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Extending the System

 > [Guides](./index.md) > Extending the System

This guide covers how to extend VisionFlow with custom functionality, including creating new MCP tools, agent types, plugins, and API extensions. Learn how to tailor the system to your specific requirements and integrate with external services.

## Table of Contents

1. [Extension Points Overview](#extension-points-overview)
2. [Creating Custom Agents](#creating-custom-agents)
3. [Creating Custom MCP Tools](#creating-custom-mcp-tools)
4. [Building Plugins](#building-plugins)
5. [API Extensions](#api-extensions)
6. [Integration with External Services](#integration-with-external-services)
7. [Custom Visualisations](#custom-visualisations)
8. [Publishing Extensions](#publishing-extensions)
9. [Best Practices](#best-practices)
10. [Troubleshooting Extensions](#troubleshooting-extensions)

## Extension Points Overview

VisionFlow provides multiple extension points for customisation:

```mermaid
graph TB
    subgraph "Extension Points"
        MCP[MCP Tools]
        AGT[Custom Agents]
        PLG[Plugins]
        API[API Extensions]
        VIZ[Visualisations]
        INT[Integrations]
    end

    subgraph "Core System"
        CORE[VisionFlow Core]
        ORCH[Agent Orchestrator]
        GFX[Graph Engine]
        WS[WebSocket Layer]
    end

    MCP --> CORE
    AGT --> ORCH
    PLG --> CORE
    API --> CORE
    VIZ --> GFX
    INT --> WS
```

### Extension Architecture

| Extension Type | Purpose | Language | Interface |
|----------------|---------|----------|-----------|
| MCP Tools | External tool integration | Python/Node.js | Stdio JSON |
| Custom Agents | Specialised AI agents | Python | Agent API |
| Plugins | System functionality | Rust/TypeScript | Plugin API |
| API Extensions | New endpoints | Rust | REST/GraphQL |
| Visualisations | Custom 3D views | TypeScript/GLSL | React Three Fiber |
| Integrations | External services | Any | WebSocket/HTTP |

## Creating Custom Agents

Custom agents extend VisionFlow's capabilities by adding specialised behaviours and workflows. The system provides comprehensive agent templates to accelerate development.

### Understanding Agent Templates

Agent templates provide reusable patterns for common agent types. Browse the complete collection at .

Key template categories:
- **Automation Templates**: Intelligent automation and workflow management
- **Coordination Templates**: Multi-agent interaction patterns
- **Specialisation Templates**: Domain-specific implementations
- **Methodology Templates**: Development methodology patterns (SPARC, TDD)

### Step-by-Step: Create a Custom Agent from Template

#### 1. Choose Your Template

Review available templates in :

-  - Intelligent agent coordination
-  - SPARC methodology implementation
-  - Task orchestration patterns
-  - Memory management
-  - GitHub PR workflows

#### 2. Define Agent Structure

Create your agent definition file:

```yaml
# agents/my-custom-agent.yaml
---
name: data-processor
colour: "purple"
type: processing
description: Intelligent data processing and transformation specialist
capabilities:
  - data-extraction
  - data-transformation
  - data-validation
  - format-conversion
  - schema-inference
priority: high
hooks:
  pre: |
    echo "🔄 Data Processor initialising..."
    echo "📊 Loading transformation rules"
    memory-retrieve "transformation-config" || echo "Using default configuration"
  post: |
    echo "✅ Processing complete"
    memory-store "last-processing-$(date +%s)" "Data processing task completed"
    echo "📈 Metrics stored for analysis"
---
```

#### 3. Implement Agent Base Class

Based on the  template:

```python
# agents/base-agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
import logging

class BaseCustomAgent(ABC):
    """Base class for custom agents."""

    def --init--(self, agent-id: str, config: Dict[str, Any]):
        self.agent-id = agent-id
        self.config = config
        self.logger = logging.getLogger(f"agent.{agent-id}")

        # Agent state
        self.status = "initialising"
        self.capabilities = []
        self.active-tasks = []
        self.metrics = {
            "tasks-completed": 0,
            "tasks-failed": 0,
            "avg-task-duration": 0
        }

    @abstractmethod
    async def initialise(self):
        """Initialise agent resources."""
        pass

    @abstractmethod
    async def process-task(self, task: Dict[str, Any]) -> Any:
        """Process a single task."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Clean up agent resources."""
        pass

    async def run(self):
        """Main agent execution loop."""
        try:
            await self.initialise()
            self.status = "ready"

            while self.status == "ready":
                task = await self.get-next-task()

                if task:
                    await self.execute-task(task)
                else:
                    await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Agent error: {e}", exc-info=True)
            self.status = "error"
        finally:
            await self.cleanup()

    async def execute-task(self, task: Dict[str, Any]):
        """Execute task with metrics tracking."""
        start-time = asyncio.get-event-loop().time()

        try:
            self.active-tasks.append(task['id'])
            result = await self.process-task(task)

            # Update metrics
            duration = asyncio.get-event-loop().time() - start-time
            self.update-metrics('success', duration)

            # Report result
            await self.report-result(task['id'], result)

        except Exception as e:
            self.logger.error(f"Task {task['id']} failed: {e}")
            self.update-metrics('failure', 0)
            await self.report-error(task['id'], str(e))

        finally:
            self.active-tasks.remove(task['id'])

    def update-metrics(self, status: str, duration: float):
        """Update agent metrics."""
        if status == 'success':
            self.metrics['tasks-completed'] += 1
            # Update average duration
            total-tasks = self.metrics['tasks-completed']
            current-avg = self.metrics['avg-task-duration']
            self.metrics['avg-task-duration'] = (
                (current-avg * (total-tasks - 1) + duration) / total-tasks
            )
        else:
            self.metrics['tasks-failed'] += 1

    async def get-next-task(self) -> Optional[Dict[str, Any]]:
        """Retrieve next task from queue."""
        # Implementation depends on your task queue system
        pass

    async def report-result(self, task-id: str, result: Any):
        """Report task result to orchestrator."""
        # Implementation depends on your communication protocol
        pass

    async def report-error(self, task-id: str, error: str):
        """Report task error to orchestrator."""
        # Implementation depends on your communication protocol
        pass
```

#### 4. Create Specialised Agent

Following the  pattern:

```python
# agents/data-processor-agent.py
import pandas as pd
from typing import Dict, Any
import aiofiles
import json

class DataProcessorAgent(BaseCustomAgent):
    """Agent specialised in data processing operations."""

    def --init--(self, agent-id: str, config: Dict[str, Any]):
        super().--init--(agent-id, config)
        self.capabilities = [
            'data-extraction',
            'data-transformation',
            'data-validation',
            'format-conversion',
            'schema-inference'
        ]
        self.transformation-rules = config.get('transformation-rules', {})
        self.validators = {}

    async def initialise(self):
        """Initialise data processing resources."""
        # Load transformation rules
        self.transformation-rules = await self.load-transformation-rules()

        # Initialise validators
        self.validators = {
            'schema': SchemaValidator(),
            'quality': DataQualityValidator(),
            'business': BusinessRuleValidator()
        }

        # Setup data cache
        self.cache = {}

        self.logger.info("Data processor agent initialised")

    async def process-task(self, task: Dict[str, Any]) -> Any:
        """Process data processing tasks."""
        task-type = task.get('type')

        handlers = {
            'extract': self.extract-data,
            'transform': self.transform-data,
            'validate': self.validate-data,
            'convert': self.convert-format,
            'infer-schema': self.infer-schema
        }

        handler = handlers.get(task-type)
        if not handler:
            raise ValueError(f"Unknown task type: {task-type}")

        return await handler(task.get('params', {}))

    async def extract-data(self, params: Dict) -> pd.DataFrame:
        """Extract data from various sources."""
        source-type = params.get('source-type')

        if source-type == 'file':
            file-path = params.get('file-path')
            file-format = params.get('format', 'csv')

            if file-format == 'csv':
                return pd.read-csv(file-path)
            elif file-format == 'json':
                async with aiofiles.open(file-path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    return pd.DataFrame(data)
            elif file-format == 'parquet':
                return pd.read-parquet(file-path)
            else:
                raise ValueError(f"Unsupported format: {file-format}")

        elif source-type == 'database':
            # Database extraction logic
            query = params.get('query')
            connection = await self.get-db-connection(params.get('connection-string'))
            return pd.read-sql(query, connection)

        else:
            raise ValueError(f"Unsupported source type: {source-type}")

    async def transform-data(self, params: Dict) -> pd.DataFrame:
        """Apply transformations to data."""
        data = params.get('data')
        rules = params.get('transformation-rules', [])

        df = pd.DataFrame(data)

        for rule in rules:
            rule-type = rule.get('type')

            if rule-type == 'filter':
                column = rule['column']
                operator = rule['operator']
                value = rule['value']

                if operator == 'gt':
                    df = df[df[column] > value]
                elif operator == 'lt':
                    df = df[df[column] < value]
                elif operator == 'eq':
                    df = df[df[column] == value]

            elif rule-type == 'aggregate':
                group-by = rule['group-by']
                aggregations = rule['aggregations']
                df = df.groupby(group-by).agg(aggregations)

            elif rule-type == 'join':
                other-data = await self.get-cached-data(rule['dataset'])
                join-key = rule['join-key']
                join-type = rule.get('join-type', 'inner')
                df = df.merge(other-data, on=join-key, how=join-type)

            elif rule-type == 'compute':
                column-name = rule['column']
                expression = rule['expression']
                df[column-name] = df.eval(expression)

        return df

    async def validate-data(self, params: Dict) -> Dict[str, Any]:
        """Validate data against rules."""
        data = params.get('data')
        validation-rules = params.get('rules', [])

        df = pd.DataFrame(data)
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        for rule in validation-rules:
            validator-type = rule.get('type')
            validator = self.validators.get(validator-type)

            if validator:
                validation-result = await validator.validate(df, rule)

                if not validation-result['valid']:
                    results['valid'] = False
                    results['errors'].extend(validation-result['errors'])

                results['warnings'].extend(validation-result.get('warnings', []))

        return results

    async def convert-format(self, params: Dict) -> str:
        """Convert data between formats."""
        data = params.get('data')
        source-format = params.get('source-format')
        target-format = params.get('target-format')
        output-path = params.get('output-path')

        df = pd.DataFrame(data)

        if target-format == 'csv':
            df.to-csv(output-path, index=False)
        elif target-format == 'json':
            df.to-json(output-path, orient='records', indent=2)
        elif target-format == 'parquet':
            df.to-parquet(output-path, index=False)
        elif target-format == 'excel':
            df.to-excel(output-path, index=False)
        else:
            raise ValueError(f"Unsupported target format: {target-format}")

        return output-path

    async def infer-schema(self, params: Dict) -> Dict[str, Any]:
        """Infer schema from data."""
        data = params.get('data')
        df = pd.DataFrame(data)

        schema = {
            'columns': [],
            'row-count': len(df),
            'nullable-columns': []
        }

        for column in df.columns:
            column-info = {
                'name': column,
                'type': str(df[column].dtype),
                'nullable': df[column].isnull().any(),
                'unique-count': df[column].nunique(),
                'sample-values': df[column].head(5).tolist()
            }

            schema['columns'].append(column-info)

            if column-info['nullable']:
                schema['nullable-columns'].append(column)

        return schema

    async def cleanup(self):
        """Clean up resources."""
        self.cache.clear()
        self.validators.clear()
        self.logger.info("Data processor agent cleaned up")

    async def load-transformation-rules(self) -> Dict:
        """Load transformation rules from configuration."""
        rules-path = self.config.get('rules-path')
        if rules-path:
            async with aiofiles.open(rules-path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        return {}

    async def get-cached-data(self, dataset-id: str) -> pd.DataFrame:
        """Retrieve cached dataset."""
        if dataset-id in self.cache:
            return self.cache[dataset-id]
        else:
            raise ValueError(f"Dataset {dataset-id} not found in cache")

    async def get-db-connection(self, connection-string: str):
        """Get database connection."""
        # Implementation depends on your database system
        pass


# Validator implementations
class SchemaValidator:
    """Validates data against schema definitions."""

    async def validate(self, df: pd.DataFrame, rule: Dict) -> Dict:
        """Validate dataframe against schema."""
        schema = rule.get('schema', {})
        errors = []
        warnings = []

        # Check required columns
        required-columns = schema.get('required-columns', [])
        missing-columns = set(required-columns) - set(df.columns)
        if missing-columns:
            errors.append(f"Missing required columns: {missing-columns}")

        # Check data types
        for column, expected-type in schema.get('column-types', {}).items():
            if column in df.columns:
                actual-type = str(df[column].dtype)
                if actual-type != expected-type:
                    warnings.append(
                        f"Column '{column}' has type {actual-type}, expected {expected-type}"
                    )

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


class DataQualityValidator:
    """Validates data quality metrics."""

    async def validate(self, df: pd.DataFrame, rule: Dict) -> Dict:
        """Validate data quality."""
        errors = []
        warnings = []

        # Check for null values
        max-null-percentage = rule.get('max-null-percentage', 0.1)
        for column in df.columns:
            null-percentage = df[column].isnull().sum() / len(df)
            if null-percentage > max-null-percentage:
                warnings.append(
                    f"Column '{column}' has {null-percentage:.2%} null values"
                )

        # Check for duplicates
        if rule.get('check-duplicates', False):
            duplicate-count = df.duplicated().sum()
            if duplicate-count > 0:
                warnings.append(f"Found {duplicate-count} duplicate rows")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


class BusinessRuleValidator:
    """Validates business rules."""

    async def validate(self, df: pd.DataFrame, rule: Dict) -> Dict:
        """Validate business rules."""
        errors = []
        warnings = []

        # Custom business rule validation logic
        business-rules = rule.get('rules', [])

        for br in business-rules:
            rule-type = br.get('type')

            if rule-type == 'range':
                column = br['column']
                min-val = br.get('min')
                max-val = br.get('max')

                if min-val is not None:
                    violations = df[df[column] < min-val]
                    if not violations.empty:
                        errors.append(
                            f"Column '{column}' has {len(violations)} values below minimum {min-val}"
                        )

                if max-val is not None:
                    violations = df[df[column] > max-val]
                    if not violations.empty:
                        errors.append(
                            f"Column '{column}' has {len(violations)} values above maximum {max-val}"
                        )

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
```

#### 5. Register Your Agent

```yaml
# config/agents.yaml
custom-agents:
  data-processor:
    class: agents.data-processor-agent.DataProcessorAgent
    config:
      max-concurrent-tasks: 10
      rules-path: /workspace/config/transformation-rules.json
      cache-size: 1000
      transformation-rules:
        default:
          - type: filter
            column: status
            operator: eq
            value: active
          - type: compute
            column: full-name
            expression: "first-name + ' ' + last-name"
    resources:
      memory: "4Gi"
      cpu: "2.0"
    capabilities:
      - data-extraction
      - data-transformation
      - data-validation
      - format-conversion
      - schema-inference
```

#### 6. Test Your Agent

Create comprehensive tests following TDD principles from :

```python
# tests/test-data-processor-agent.py
import pytest
import pandas as pd
from agents.data-processor-agent import DataProcessorAgent

@pytest.fixture
async def agent():
    """Create test agent instance."""
    config = {
        'transformation-rules': {},
        'cache-size': 100
    }
    agent = DataProcessorAgent('test-agent', config)
    await agent.initialise()
    yield agent
    await agent.cleanup()

@pytest.mark.asyncio
async def test-extract-csv-data(agent, tmp-path):
    """Test CSV data extraction."""
    # Arrange
    test-file = tmp-path / "test.csv"
    test-data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    test-data.to-csv(test-file, index=False)

    params = {
        'source-type': 'file',
        'file-path': str(test-file),
        'format': 'csv'
    }

    # Act
    result = await agent.extract-data(params)

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ['col1', 'col2']

@pytest.mark.asyncio
async def test-transform-filter(agent):
    """Test data filtering transformation."""
    # Arrange
    test-data = pd.DataFrame({
        'value': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'B', 'A']
    })

    params = {
        'data': test-data.to-dict('records'),
        'transformation-rules': [
            {
                'type': 'filter',
                'column': 'value',
                'operator': 'gt',
                'value': 2
            }
        ]
    }

    # Act
    result = await agent.transform-data(params)

    # Assert
    assert len(result) == 3
    assert all(result['value'] > 2)

@pytest.mark.asyncio
async def test-validate-schema(agent):
    """Test schema validation."""
    # Arrange
    test-data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })

    params = {
        'data': test-data.to-dict('records'),
        'rules': [
            {
                'type': 'schema',
                'schema': {
                    'required-columns': ['id', 'name', 'email'],
                    'column-types': {
                        'id': 'int64',
                        'name': 'object'
                    }
                }
            }
        ]
    }

    # Act
    result = await agent.validate-data(params)

    # Assert
    assert not result['valid']
    assert any('email' in error for error in result['errors'])

@pytest.mark.asyncio
async def test-infer-schema(agent):
    """Test schema inference."""
    # Arrange
    test-data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', None],
        'score': [95.5, 87.3, 91.2]
    })

    params = {
        'data': test-data.to-dict('records')
    }

    # Act
    result = await agent.infer-schema(params)

    # Assert
    assert result['row-count'] == 3
    assert len(result['columns']) == 3
    assert 'name' in result['nullable-columns']
```

### Agent Template Resources

For more examples and patterns, explore:

-  - Complete template catalogue
-  - Intelligent coordination patterns
-  - Implementation methodology
-  - Orchestration patterns
-  - Memory management
-  - GitHub workflows

### Contributing New Templates

Help expand the template library! See [Contributing Guide](./contributing.md) for:

- Template submission guidelines
- Documentation standards
- Testing requirements
- Review process

## Creating Custom MCP Tools

MCP (Model Context Protocol) tools enable external integrations through standardised stdio communication.

### MCP Tool Anatomy

MCP tools communicate via stdio using JSON messages:

```python
#!/usr/bin/env python3
"""
Custom MCP Tool Template
"""
import sys
import json
import logging
from typing import Dict, Any, Optional

class CustomMCPTool:
    def --init--(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('/app/logs/custom-tool.log')]
        )
        self.logger = logging.getLogger(--name--)

        # Initialise tool state
        self.config = self.load-config()
        self.capabilities = ['process', 'analyse', 'transform']

    def load-config(self) -> Dict:
        """Load tool configuration."""
        try:
            with open('/app/config/custom-tool.json', 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def process-request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming MCP request."""
        try:
            method = request.get('method', 'default')
            params = request.get('params', {})

            # Route to appropriate handler
            handlers = {
                'process': self.handle-process,
                'analyse': self.handle-analyse,
                'transform': self.handle-transform,
                'capabilities': self.get-capabilities
            }

            handler = handlers.get(method, self.handle-unknown)
            result = handler(params)

            return {'result': result}

        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc-info=True)
            return {'error': str(e)}

    def handle-process(self, params: Dict) -> Any:
        """Handle process requests."""
        data = params.get('data')
        options = params.get('options', {})

        # Your processing logic here
        processed = self.process-data(data, options)

        return {
            'status': 'success',
            'processed': processed,
            'metadata': {
                'items-processed': len(processed) if hasattr(processed, '--len--') else 1,
                'options-used': options
            }
        }

    def handle-analyse(self, params: Dict) -> Any:
        """Handle analysis requests."""
        input-data = params.get('input')
        depth = params.get('depth', 1)

        # Analysis implementation
        analysis-result = {
            'summary': f"Analysed {len(input-data)} items at depth {depth}",
            'insights': [],
            'recommendations': []
        }

        return analysis-result

    def handle-transform(self, params: Dict) -> Any:
        """Handle transformation requests."""
        data = params.get('data')
        transformation = params.get('transformation')

        # Transformation implementation
        transformed-data = self.apply-transformation(data, transformation)

        return transformed-data

    def handle-unknown(self, params: Dict) -> Any:
        """Handle unknown methods."""
        return {'error': 'Unknown method'}

    def get-capabilities(self, params: Dict) -> list:
        """Return tool capabilities."""
        return self.capabilities

    def process-data(self, data: Any, options: Dict) -> Any:
        """Process data with options."""
        # Implementation
        return data

    def apply-transformation(self, data: Any, transformation: str) -> Any:
        """Apply transformation to data."""
        # Implementation
        return data

    def run(self):
        """Main execution loop."""
        self.logger.info("Custom MCP Tool started")

        # Read from stdin, write to stdout
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = self.process-request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                error-response = {'error': f'Invalid JSON: {e}'}
                print(json.dumps(error-response), flush=True)
            except Exception as e:
                error-response = {'error': str(e)}
                print(json.dumps(error-response), flush=True)

if --name-- == '--main--':
    tool = CustomMCPTool()
    tool.run()
```

### Registering the Tool

1. **Add to MCP Configuration**
```json
// .mcp.json
{
  "tools": {
    "custom-tool": {
      "command": "python3",
      "args": ["-u", "./mcp-tools/custom-tool.py"],
      "description": "Custom tool for specialised processing",
      "schema": {
        "methods": {
          "process": {
            "params": {
              "data": "array",
              "options": "object"
            }
          },
          "analyse": {
            "params": {
              "input": "string",
              "depth": "number"
            }
          }
        }
      }
    }
  }
}
```

2. **Create Tool Wrapper Script**
```bash
#!/bin/bash
# mcp-tools/custom-tool-wrapper.sh

# Set environment
export PYTHONUNBUFFERED=1
export TOOL-CONFIG-PATH=/workspace/config/custom-tool.json

# Run tool with proper error handling
exec python3 -u /workspace/mcp-tools/custom-tool.py 2>>/app/logs/custom-tool.error.log
```

### Advanced MCP Tool Features

#### Asynchronous Operations

```python
import asyncio
import aiohttp

class AsyncMCPTool(CustomMCPTool):
    async def handle-fetch(self, params: Dict) -> Any:
        """Handle async fetch operations."""
        url = params.get('url')
        timeout = params.get('timeout', 30)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                data = await response.json()

        return {
            'status': response.status,
            'data': data,
            'headers': dict(response.headers)
        }

    def process-request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request with async support."""
        method = request.get('method')

        if method == 'fetch':
            # Run async method in event loop
            loop = asyncio.new-event-loop()
            asyncio.set-event-loop(loop)
            result = loop.run-until-complete(
                self.handle-fetch(request.get('params', {}))
            )
            loop.close()
            return {'result': result}

        return super().process-request(request)
```

#### State Management

```python
from datetime import datetime
import uuid

class StatefulMCPTool(CustomMCPTool):
    def --init--(self):
        super().--init--()
        self.sessions = {}

    def handle-create-session(self, params: Dict) -> str:
        """Create new session."""
        session-id = str(uuid.uuid4())

        self.sessions[session-id] = {
            'created': datetime.utcnow().isoformat(),
            'data': {},
            'history': []
        }

        return session-id

    def handle-session-operation(self, params: Dict) -> Any:
        """Handle session-based operations."""
        session-id = params.get('session-id')
        operation = params.get('operation')

        if session-id not in self.sessions:
            raise ValueError("Invalid session ID")

        session = self.sessions[session-id]
        session['history'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation
        })

        # Process operation
        result = self.process-session-operation(session, operation)

        return result

    def handle-destroy-session(self, params: Dict) -> Dict:
        """Destroy session and clean up."""
        session-id = params.get('session-id')

        if session-id in self.sessions:
            session = self.sessions.pop(session-id)
            return {
                'status': 'destroyed',
                'operations-count': len(session['history'])
            }

        return {'status': 'not-found'}

    def process-session-operation(self, session: Dict, operation: Dict) -> Any:
        """Process operation within session context."""
        # Implementation
        return {'status': 'completed'}
```

## Building Plugins

### Plugin Architecture

```rust
// src/plugins/plugin-interface.rs
use async-trait::async-trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[async-trait]
pub trait Plugin: Send + Sync {
    /// Plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Initialise plugin
    async fn initialise(&mut self, config: PluginConfig) -> Result<(), PluginError>;

    /// Handle plugin events
    async fn handle-event(&self, event: PluginEvent) -> Result<PluginResponse, PluginError>;

    /// Cleanup plugin
    async fn cleanup(&mut self) -> Result<(), PluginError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub settings: HashMap<String, serde-json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEvent {
    pub event-type: String,
    pub payload: serde-json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginResponse {
    pub success: bool,
    pub data: Option<serde-json::Value>,
    pub error: Option<String>,
}

#[derive(Debug)]
pub enum PluginError {
    InitialisationError(String),
    InvalidData(String),
    UnknownEvent(String),
    ProcessingError(String),
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PluginError::InitialisationError(msg) => write!(f, "Initialisation error: {}", msg),
            PluginError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            PluginError::UnknownEvent(msg) => write!(f, "Unknown event: {}", msg),
            PluginError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl std::error::Error for PluginError {}
```

### Example Plugin Implementation

```rust
// src/plugins/custom-analytics.rs
use super::{Plugin, PluginMetadata, PluginConfig, PluginEvent, PluginResponse, PluginError};
use async-trait::async-trait;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Node {
    id: String,
    label: String,
    properties: HashMap<String, serde-json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Edge {
    source: String,
    target: String,
    weight: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct GraphMetrics {
    node-count: usize,
    edge-count: usize,
    average-degree: f64,
    clustering-coefficient: f64,
    connected-components: usize,
}

pub struct CustomAnalyticsPlugin {
    config: Option<PluginConfig>,
    metrics: HashMap<String, f64>,
}

impl CustomAnalyticsPlugin {
    pub fn new() -> Self {
        Self {
            config: None,
            metrics: HashMap::new(),
        }
    }

    fn setup-analytics-engine(&mut self) -> Result<(), PluginError> {
        // Initialise analytics engine
        self.metrics.insert("total-analyses".to-string(), 0.0);
        Ok(())
    }

    async fn analyse-graph(&self, data: serde-json::Value) -> Result<PluginResponse, PluginError> {
        // Parse graph data
        let graph: Graph = serde-json::from-value(data)
            .map-err(|e| PluginError::InvalidData(e.to-string()))?;

        // Perform analysis
        let metrics = self.calculate-graph-metrics(&graph)?;

        Ok(PluginResponse {
            success: true,
            data: Some(serde-json::to-value(metrics)
                .map-err(|e| PluginError::ProcessingError(e.to-string()))?),
            error: None,
        })
    }

    async fn detect-patterns(&self, data: serde-json::Value) -> Result<PluginResponse, PluginError> {
        let graph: Graph = serde-json::from-value(data)
            .map-err(|e| PluginError::InvalidData(e.to-string()))?;

        // Pattern detection logic
        let patterns = self.find-patterns(&graph)?;

        Ok(PluginResponse {
            success: true,
            data: Some(serde-json::to-value(patterns)
                .map-err(|e| PluginError::ProcessingError(e.to-string()))?),
            error: None,
        })
    }

    async fn find-anomalies(&self, data: serde-json::Value) -> Result<PluginResponse, PluginError> {
        let graph: Graph = serde-json::from-value(data)
            .map-err(|e| PluginError::InvalidData(e.to-string()))?;

        // Anomaly detection logic
        let anomalies = self.detect-anomalies-in-graph(&graph)?;

        Ok(PluginResponse {
            success: true,
            data: Some(serde-json::to-value(anomalies)
                .map-err(|e| PluginError::ProcessingError(e.to-string()))?),
            error: None,
        })
    }

    fn calculate-graph-metrics(&self, graph: &Graph) -> Result<GraphMetrics, PluginError> {
        let metrics = GraphMetrics {
            node-count: graph.nodes.len(),
            edge-count: graph.edges.len(),
            average-degree: self.calculate-average-degree(graph),
            clustering-coefficient: self.calculate-clustering-coefficient(graph),
            connected-components: self.find-connected-components(graph),
        };

        Ok(metrics)
    }

    fn calculate-average-degree(&self, graph: &Graph) -> f64 {
        if graph.nodes.is-empty() {
            return 0.0;
        }
        (2.0 * graph.edges.len() as f64) / graph.nodes.len() as f64
    }

    fn calculate-clustering-coefficient(&self, graph: &Graph) -> f64 {
        // Clustering coefficient calculation
        0.0 // Placeholder
    }

    fn find-connected-components(&self, graph: &Graph) -> usize {
        // Connected components algorithm
        1 // Placeholder
    }

    fn find-patterns(&self, graph: &Graph) -> Result<Vec<String>, PluginError> {
        // Pattern detection
        Ok(vec!["pattern1".to-string(), "pattern2".to-string()])
    }

    fn detect-anomalies-in-graph(&self, graph: &Graph) -> Result<Vec<String>, PluginError> {
        // Anomaly detection
        Ok(vec![])
    }
}

#[async-trait]
impl Plugin for CustomAnalyticsPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "Custom Analytics".to-string(),
            version: "1.0.0".to-string(),
            author: "Your Name".to-string(),
            description: "Advanced analytics for graph data".to-string(),
            capabilities: vec![
                "graph-analysis".to-string(),
                "pattern-detection".to-string(),
                "anomaly-detection".to-string(),
            ],
        }
    }

    async fn initialise(&mut self, config: PluginConfig) -> Result<(), PluginError> {
        self.config = Some(config);

        // Initialise analytics engine
        self.setup-analytics-engine()?;

        Ok(())
    }

    async fn handle-event(&self, event: PluginEvent) -> Result<PluginResponse, PluginError> {
        match event.event-type.as-str() {
            "analyse-graph" => self.analyse-graph(event.payload).await,
            "detect-patterns" => self.detect-patterns(event.payload).await,
            "find-anomalies" => self.find-anomalies(event.payload).await,
            - => Err(PluginError::UnknownEvent(event.event-type)),
        }
    }

    async fn cleanup(&mut self) -> Result<(), PluginError> {
        self.metrics.clear();
        Ok(())
    }
}
```

### Plugin Registration

```toml
# plugins.toml
[[plugins]]
name = "custom-analytics"
path = "./plugins/custom-analytics.so"
enabled = true
config = { buffer-size = 1000, cache-ttl = 300 }

[[plugins]]
name = "external-integration"
path = "./plugins/external-integration.wasm"
enabled = true
config = { api-endpoint = "https://api.example.com" }
```

## API Extensions

### Creating Custom Endpoints

```rust
// src/api/extensions/custom-endpoints.rs
use actix-web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Serialize, Deserialize)]
struct CustomAnalysisRequest {
    graph-id: String,
    analysis-type: String,
    parameters: serde-json::Value,
}

#[derive(Serialize)]
struct CustomAnalysisResponse {
    result: serde-json::Value,
    metadata: AnalysisMetadata,
}

#[derive(Serialize)]
struct AnalysisMetadata {
    timestamp: DateTime<Utc>,
    duration-ms: u64,
}

pub fn configure-custom-routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/custom")
            .route("/analyse", web::post().to(analyse-graph))
            .route("/visualise/{id}", web::get().to(get-visualisation))
            .route("/export", web::post().to(export-data))
    );
}

async fn analyse-graph(
    req: web::Json<CustomAnalysisRequest>,
    app-state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let start-time = std::time::Instant::now();

    let analysis-result = app-state
        .analyser
        .perform-analysis(&req.graph-id, &req.analysis-type, &req.parameters)
        .await?;

    let duration-ms = start-time.elapsed().as-millis() as u64;

    Ok(HttpResponse::Ok().json(CustomAnalysisResponse {
        result: analysis-result,
        metadata: AnalysisMetadata {
            timestamp: Utc::now(),
            duration-ms,
        },
    }))
}

async fn get-visualisation(
    path: web::Path<String>,
    app-state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let graph-id = path.into-inner();

    let visualisation = app-state
        .visualisation-engine
        .generate(&graph-id)
        .await?;

    Ok(HttpResponse::Ok().json(visualisation))
}

async fn export-data(
    req: web::Json<ExportRequest>,
    app-state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let export-result = app-state
        .exporter
        .export(&req.graph-id, &req.format)
        .await?;

    Ok(HttpResponse::Ok().json(export-result))
}

#[derive(Deserialize)]
struct ExportRequest {
    graph-id: String,
    format: String,
}

// Application state
pub struct AppState {
    pub analyser: Box<dyn GraphAnalyser>,
    pub visualisation-engine: Box<dyn VisualisationEngine>,
    pub exporter: Box<dyn DataExporter>,
}

// Traits for dependency injection
#[async-trait::async-trait]
pub trait GraphAnalyser: Send + Sync {
    async fn perform-analysis(
        &self,
        graph-id: &str,
        analysis-type: &str,
        parameters: &serde-json::Value,
    ) -> Result<serde-json::Value, Box<dyn std::error::Error>>;
}

#[async-trait::async-trait]
pub trait VisualisationEngine: Send + Sync {
    async fn generate(
        &self,
        graph-id: &str,
    ) -> Result<serde-json::Value, Box<dyn std::error::Error>>;
}

#[async-trait::async-trait]
pub trait DataExporter: Send + Sync {
    async fn export(
        &self,
        graph-id: &str,
        format: &str,
    ) -> Result<serde-json::Value, Box<dyn std::error::Error>>;
}
```

### GraphQL Extensions

```rust
// src/graphql/custom-schema.rs
use juniper::{FieldResult, RootNode, graphql-object};
use serde::{Deserialize, Serialize};

pub struct Context {
    pub search-engine: Box<dyn SearchEngine>,
    pub pattern-detector: Box<dyn PatternDetector>,
    pub visualisation-engine: Box<dyn VisualisationEngine>,
}

impl juniper::Context for Context {}

pub struct CustomQuery;

#[graphql-object(context = Context)]
impl CustomQuery {
    async fn advanced-search(
        ctx: &Context,
        query: String,
        filters: SearchFilters,
        limit: Option<i32>,
    ) -> FieldResult<SearchResults> {
        let results = ctx
            .search-engine
            .search(&query, filters, limit.unwrap-or(10))
            .await?;

        Ok(results)
    }

    async fn pattern-analysis(
        ctx: &Context,
        graph-id: String,
        pattern-type: PatternType,
    ) -> FieldResult<Vec<Pattern>> {
        let patterns = ctx
            .pattern-detector
            .find-patterns(&graph-id, pattern-type)
            .await?;

        Ok(patterns)
    }
}

pub struct CustomMutation;

#[graphql-object(context = Context)]
impl CustomMutation {
    async fn create-custom-visualisation(
        ctx: &Context,
        input: VisualisationInput,
    ) -> FieldResult<Visualisation> {
        let viz = ctx
            .visualisation-engine
            .create-custom(input)
            .await?;

        Ok(viz)
    }
}

#[derive(juniper::GraphQLInputObject)]
struct SearchFilters {
    category: Option<String>,
    tags: Option<Vec<String>>,
    date-range: Option<DateRange>,
}

#[derive(juniper::GraphQLInputObject)]
struct DateRange {
    start: String,
    end: String,
}

#[derive(juniper::GraphQLObject)]
struct SearchResults {
    total: i32,
    items: Vec<SearchResult>,
}

#[derive(juniper::GraphQLObject)]
struct SearchResult {
    id: String,
    title: String,
    score: f64,
}

#[derive(juniper::GraphQLEnum)]
enum PatternType {
    Cycle,
    Tree,
    Hub,
    Community,
}

#[derive(juniper::GraphQLObject)]
struct Pattern {
    id: String,
    pattern-type: String,
    nodes: Vec<String>,
    confidence: f64,
}

#[derive(juniper::GraphQLInputObject)]
struct VisualisationInput {
    graph-id: String,
    layout: String,
    colour-scheme: String,
}

#[derive(juniper::GraphQLObject)]
struct Visualisation {
    id: String,
    graph-id: String,
    layout: String,
    created-at: String,
}

// Trait definitions
#[async-trait::async-trait]
pub trait SearchEngine: Send + Sync {
    async fn search(
        &self,
        query: &str,
        filters: SearchFilters,
        limit: i32,
    ) -> Result<SearchResults, Box<dyn std::error::Error>>;
}

#[async-trait::async-trait]
pub trait PatternDetector: Send + Sync {
    async fn find-patterns(
        &self,
        graph-id: &str,
        pattern-type: PatternType,
    ) -> Result<Vec<Pattern>, Box<dyn std::error::Error>>;
}

#[async-trait::async-trait]
pub trait VisualisationEngine: Send + Sync {
    async fn create-custom(
        &self,
        input: VisualisationInput,
    ) -> Result<Visualisation, Box<dyn std::error::Error>>;
}
```

## Integration with External Services

### Webhook Integration

```python
# integrations/webhook-handler.py
from aiohttp import web
import aiohttp
import hmac
import hashlib
import logging
from typing import Callable, Dict, Any

logger = logging.getLogger(--name--)

class WebhookIntegration:
    def --init--(self, config: Dict[str, Any]):
        self.config = config
        self.secret = config.get('webhook-secret', '').encode()
        self.handlers: Dict[str, Callable] = {}

    def register-handler(self, event-type: str, handler: Callable):
        """Register webhook handler for event type."""
        self.handlers[event-type] = handler
        logger.info(f"Registered handler for event type: {event-type}")

    async def handle-webhook(self, request: web.Request) -> web.Response:
        """Handle incoming webhook."""
        # Verify webhook signature
        signature = request.headers.get('X-Webhook-Signature', '')
        body = await request.read()

        if not self.verify-signature(body, signature):
            logger.warning("Invalid webhook signature")
            return web.Response(status=401, text="Invalid signature")

        # Parse webhook data
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse webhook data: {e}")
            return web.json-response(
                {'status': 'error', 'error': 'Invalid JSON'},
                status=400
            )

        event-type = data.get('event-type')

        # Route to handler
        handler = self.handlers.get(event-type)
        if handler:
            try:
                result = await handler(data)
                logger.info(f"Successfully handled webhook: {event-type}")
                return web.json-response({'status': 'success', 'result': result})
            except Exception as e:
                logger.error(f"Handler error for {event-type}: {e}", exc-info=True)
                return web.json-response(
                    {'status': 'error', 'error': str(e)},
                    status=500
                )

        logger.warning(f"No handler registered for event type: {event-type}")
        return web.json-response(
            {'status': 'error', 'error': 'Unknown event type'},
            status=400
        )

    def verify-signature(self, body: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        if not self.secret:
            logger.warning("No webhook secret configured")
            return False

        expected = hmac.new(
            self.secret,
            body,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare-digest(expected, signature)
```

### External Service Client

```typescript
// src/integrations/ExternalServiceClient.ts
import axios, { AxiosInstance, AxiosError } from 'axios';
import { v4 as uuidv4 } from 'uuid';

interface ExternalServiceConfig {
  apiKey: string;
  baseUrl: string;
  timeout?: number;
  retryAttempts?: number;
}

interface SyncResult {
  status: string;
  syncedCount: number;
  errors: string[];
}

export class ExternalServiceClient {
  private apiKey: string;
  private baseUrl: string;
  private httpClient: AxiosInstance;
  private retryAttempts: number;

  constructor(config: ExternalServiceConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl;
    this.retryAttempts = config.retryAttempts || 3;

    this.httpClient = axios.create({
      baseURL: this.baseUrl,
      headers: {
        'Authorisation': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: config.timeout || 30000
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.httpClient.interceptors.request.use(
      (config) => {
        config.headers['X-Request-ID'] = uuidv4();
        config.headers['X-Client-Version'] = '1.0.0';
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.httpClient.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          await this.refreshToken();
          return this.httpClient.request(error.config!);
        }

        if (error.response?.status === 429) {
          // Rate limit handling
          const retryAfter = error.response.headers['retry-after'];
          await this.delay(parseInt(retryAfter) * 1000 || 5000);
          return this.httpClient.request(error.config!);
        }

        return Promise.reject(error);
      }
    );
  }

  async syncData(data: any): Promise<SyncResult> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const response = await this.httpClient.post('/sync', {
          data,
          timestamp: new Date().toISOString()
        });

        return response.data;
      } catch (error) {
        lastError = error as Error;

        if (attempt < this.retryAttempts) {
          await this.delay(Math.pow(2, attempt) * 1000);
        }
      }
    }

    throw lastError;
  }

  async streamUpdates(onUpdate: (update: any) => void): Promise<void> {
    const eventSource = new EventSource(
      `${this.baseUrl}/stream?token=${this.apiKey}`
    );

    eventSource.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        onUpdate(update);
      } catch (error) {
        console.error('Failed to parse update:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('Stream error:', error);
      eventSource.close();

      // Attempt reconnection
      setTimeout(() => this.streamUpdates(onUpdate), 5000);
    };
  }

  async batchRequest<T>(requests: Array<() => Promise<T>>): Promise<T[]> {
    const results = await Promise.allSettled(
      requests.map(req => req())
    );

    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        console.error(`Request ${index} failed:`, result.reason);
        throw result.reason;
      }
    });
  }

  private async refreshToken(): Promise<void> {
    // Token refresh implementation
    const response = await axios.post(`${this.baseUrl}/auth/refresh`, {
      apiKey: this.apiKey
    });

    this.apiKey = response.data.token;
    this.httpClient.defaults.headers['Authorisation'] = `Bearer ${this.apiKey}`;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

## Custom Visualisations

### Three.js Custom Component

```typescript
// src/components/CustomVisualisation.tsx
import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { BufferGeometry, Float32BufferAttribute, ShaderMaterial, Points } from 'three';
import * as THREE from 'three';

interface VisualisationData {
  points: Array<{
    x: number;
    y: number;
    z: number;
    colour: { r: number; g: number; b: number };
    size: number;
  }>;
}

interface VisualisationConfig {
  animationSpeed: number;
  colourScheme: string;
  particleSize: number;
}

interface CustomVisualisationProps {
  data: VisualisationData;
  config: VisualisationConfig;
}

export const CustomVisualisation: React.FC<CustomVisualisationProps> = ({
  data,
  config
}) => {
  const meshRef = useRef<Points>(null);
  const { camera } = useThree();

  // Generate geometry from data
  const geometry = useMemo(() => {
    const geo = new BufferGeometry();

    const positions = new Float32Array(data.points.length * 3);
    const colours = new Float32Array(data.points.length * 3);
    const sizes = new Float32Array(data.points.length);

    data.points.forEach((point, i) => {
      positions[i * 3] = point.x;
      positions[i * 3 + 1] = point.y;
      positions[i * 3 + 2] = point.z;

      colours[i * 3] = point.colour.r;
      colours[i * 3 + 1] = point.colour.g;
      colours[i * 3 + 2] = point.colour.b;

      sizes[i] = point.size * config.particleSize;
    });

    geo.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geo.setAttribute('colour', new Float32BufferAttribute(colours, 3));
    geo.setAttribute('size', new Float32BufferAttribute(sizes, 1));

    return geo;
  }, [data, config.particleSize]);

  // Custom shader material
  const material = useMemo(() => {
    return new ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        cameraPosition: { value: camera.position }
      },
      vertexShader: `
        attribute float size;
        attribute vec3 colour;

        varying vec3 vColor;
        varying float vDistance;

        uniform vec3 cameraPosition;

        void main() {
          vColor = colour;

          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vDistance = length(cameraPosition - position);

          gl-PointSize = size * (300.0 / -mvPosition.z);
          gl-Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vDistance;

        uniform float time;

        void main() {
          vec2 uv = gl-PointCoord - vec2(0.5);
          float dist = length(uv);

          if (dist > 0.5) discard;

          float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
          alpha *= 1.0 - smoothstep(100.0, 500.0, vDistance);

          // Pulsing effect
          alpha *= 0.8 + 0.2 * sin(time * 2.0);

          gl-FragColor = vec4(vColor, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending
    });
  }, [camera]);

  // Animation
  useFrame((state, delta) => {
    if (meshRef.current) {
      material.uniforms.time.value += delta * config.animationSpeed;
      material.uniforms.cameraPosition.value = state.camera.position;
    }
  });

  // Cleanup
  useEffect(() => {
    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  return (
    <points ref={meshRef} geometry={geometry} material={material} />
  );
};
```

## Publishing Extensions

### Package Structure

```
my-visionflow-extension/
├── package.json
├── readme.md
├── LICENSE
├── src/
│   ├── index.ts
│   ├── agent/
│   │   └── MyCustomAgent.ts
│   ├── tools/
│   │   └── my-tool.py
│   ├── components/
│   │   └── MyVisualisation.tsx
│   └── api/
│       └── endpoints.ts
├── dist/
├── config/
│   ├── agent.yaml
│   └── tool.json
├── tests/
│   ├── agent.test.ts
│   └── tool.test.py
└── examples/
    └── usage.md
```

### Extension Manifest

```json
// extension.json
{
  "name": "my-visionflow-extension",
  "version": "1.0.0",
  "description": "Custom extension for VisionFlow",
  "author": "Your Name",
  "license": "MIT",
  "visionflow": {
    "minVersion": "1.0.0",
    "maxVersion": "2.0.0"
  },
  "components": {
    "agents": [
      {
        "type": "data-processor",
        "class": "MyCustomAgent",
        "config": "./config/agent.yaml"
      }
    ],
    "tools": [
      {
        "name": "my-tool",
        "command": "python3 -u ./src/tools/my-tool.py",
        "config": "./config/tool.json"
      }
    ],
    "visualisations": [
      {
        "name": "MyVisualisation",
        "component": "./dist/components/MyVisualisation.js"
      }
    ],
    "api": {
      "routes": "./dist/api/endpoints.js"
    }
  },
  "dependencies": {
    "visionflow-sdk": "^1.0.0"
  }
}
```

### Publishing Process

```bash
# Build extension
npm run build

# Test locally
visionflow-cli test-extension ./

# Validate extension
visionflow-cli validate-extension ./

# Package extension
visionflow-cli package-extension ./

# Publish to registry
visionflow-cli publish-extension ./dist/my-extension-1.0.0.vfx
```

### Extension Installation

```bash
# Install from registry
visionflow-cli install-extension my-visionflow-extension

# Install from file
visionflow-cli install-extension ./my-extension-1.0.0.vfx

# Install from GitHub
visionflow-cli install-extension github:username/repo

# List installed extensions
visionflow-cli list-extensions

# Update extension
visionflow-cli update-extension my-visionflow-extension

# Remove extension
visionflow-cli remove-extension my-visionflow-extension
```

## Best Practices

### Extension Development

1. **Modularity**: Keep extensions focused on specific functionality
2. **Documentation**: Provide clear documentation and examples
3. **Error Handling**: Implement robust error handling
4. **Performance**: Optimise for performance and resource usage
5. **Testing**: Include comprehensive tests

### Security Considerations

```python
# Validate inputs
import jsonschema

def validate-input(data: Dict) -> bool:
    """Validate and sanitise input data."""
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["process", "analyse"]},
            "data": {"type": "array", "maxItems": 1000}
        },
        "required": ["action", "data"]
    }

    try:
        jsonschema.validate(data, schema)
        return True
    except jsonschema.ValidationError:
        return False

# Sandbox execution
def execute-in-sandbox(code: str) -> Any:
    """Execute code in sandboxed environment."""
    restricted-globals = {
        "--builtins--": {
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            # Limited built-ins only
        }
    }

    return exec(code, restricted-globals, {})
```

### Version Compatibility

```typescript
// Check version compatibility
export function checkCompatibility(
  requiredVersion: string,
  currentVersion: string
): boolean {
  const required = parseVersion(requiredVersion);
  const current = parseVersion(currentVersion);

  return (
    current.major === required.major &&
    current.minor >= required.minor
  );
}

function parseVersion(version: string) {
  const [major, minor, patch] = version.split('.').map(Number);
  return { major, minor, patch };
}

// Provide compatibility layer
export class CompatibilityAdapter {
  constructor(private version: string) {}

  async callAPI(method: string, params: any): Promise<any> {
    if (this.version.startsWith('1.')) {
      return this.callV1API(method, params);
    } else if (this.version.startsWith('2.')) {
      return this.callV2API(method, params);
    }

    throw new Error(`Unsupported version: ${this.version}`);
  }

  private async callV1API(method: string, params: any): Promise<any> {
    // V1 implementation
    return {};
  }

  private async callV2API(method: string, params: any): Promise<any> {
    // V2 implementation
    return {};
  }
}
```

## Troubleshooting Extensions

### Common Issues

1. **Extension Not Loading**
```bash
# Check extension status
visionflow-cli status my-extension

# View extension logs
docker logs visionflow-container | grep my-extension

# Validate configuration
visionflow-cli validate-config ./extension.json
```

2. **Performance Issues**
```python
# Profile extension performance
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your extension code
result = process-data(large-dataset)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort-stats('cumulative')
stats.print-stats(10)
```

3. **Debugging Tips**
- Enable debug logging in extension config
- Use remote debugging for complex issues
- Monitor resource usage
- Test with minimal configuration first

## Related Documentation

-  - Complete template catalogue
- [Contributing Guide](./contributing.md) - Contribution guidelines
-  - Common issues
-  - API details
-  - Development practices

---

* | [Back to Guides](README.md) | *

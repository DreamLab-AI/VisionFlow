---
title: VisionFlow Error Codes Reference
description: This document provides a comprehensive reference of all error codes used throughout the VisionFlow system. Error codes follow a hierarchical naming scheme: `[SYSTEM]-[SEVERITY]-[NUMBER]`.
category: reference
tags:
  - rest
  - websocket
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# VisionFlow Error Codes Reference

## Overview

This document provides a comprehensive reference of all error codes used throughout the VisionFlow system. Error codes follow a hierarchical naming scheme: `[SYSTEM]-[SEVERITY]-[NUMBER]`.

## Error Code Format

**Format Pattern:** `[SYSTEM][SEVERITY][NUMBER]`

### System Identifiers (2-char)
- **AP**: API/Application Layer
- **DB**: Database Layer
- **GR**: Graph/Ontology Reasoning
- **GP**: GPU/Physics Computing
- **WS**: WebSocket/Network
- **AU**: Authentication/Authorization
- **ST**: Storage/File Management

### Severity Levels (1-char)
- **E**: Error (operation failed, can recover)
- **F**: Fatal (unrecoverable, requires restart)
- **W**: Warning (degraded performance, operation continues)
- **I**: Info (informational, no action required)

### Error Number (3-digit)
- Range: 000-999

## API Layer Errors (AP)

### AP-E-001 to AP-E-099: Request Validation

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `AP-E-001` | Invalid Request Format | Request body is malformed JSON | Verify JSON syntax and structure |
| `AP-E-002` | Missing Required Field | Required field '{field}' is missing | Add missing required field |
| `AP-E-003` | Invalid Field Type | Field '{field}' has wrong type (expected {type}) | Correct field type |
| `AP-E-004` | Invalid Enum Value | '{value}' is not valid for {enum} | Use valid enum value |
| `AP-E-005` | String Too Long | Field '{field}' exceeds max length ({max}) | Shorten string value |
| `AP-E-006` | String Too Short | Field '{field}' is below min length ({min}) | Extend string value |
| `AP-E-007` | Number Out of Range | Field '{field}' ({value}) outside range ({min}-{max}) | Use value within range |
| `AP-E-008` | Invalid URI Format | URI '{uri}' is not valid | Provide properly formatted URI |
| `AP-E-009` | Constraint Violation | Value violates constraint: {constraint} | Ensure value meets constraint |
| `AP-E-010` | Duplicate Value | Value '{value}' already exists for {field} | Use unique value |

### AP-E-100 to AP-E-199: Authentication/Authorization

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `AP-E-101` | Missing Auth Token | Authorization header missing | Provide valid JWT token |
| `AP-E-102` | Invalid Token | Token is invalid or expired | Refresh authentication token |
| `AP-E-103` | Token Expired | Authentication token has expired | Refresh token or re-authenticate |
| `AP-E-104` | Insufficient Permissions | User lacks permission for {action} | Request elevated permissions |
| `AP-E-105` | Resource Forbidden | Access to {resource} is forbidden | Contact administrator |
| `AP-E-106` | User Not Found | User '{id}' does not exist | Verify user ID |
| `AP-E-107` | User Disabled | User '{id}' account is disabled | Contact support |

### AP-E-200 to AP-E-299: Resource Not Found

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `AP-E-201` | Resource Not Found | {resource} with ID '{id}' not found | Verify resource ID exists |
| `AP-E-202` | Project Not Found | Project '{id}' not found | Create project or use correct ID |
| `AP-E-203` | Asset Not Found | Asset '{id}' not found in project | Upload asset or verify ID |
| `AP-E-204` | Graph Not Found | Knowledge graph '{id}' not found | Create graph or verify ID |
| `AP-E-205` | Job Not Found | Job '{id}' not found | Verify job ID exists |

### AP-E-300 to AP-E-399: Business Logic Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `AP-E-301` | Invalid State Transition | Cannot transition from {state} to {new-state} | Check valid state transitions |
| `AP-E-302` | Duplicate Resource | {resource} with this identifier already exists | Use unique identifier |
| `AP-E-303` | Resource In Use | Cannot delete {resource}, it is still in use | Remove all dependencies first |
| `AP-E-304` | Quota Exceeded | Storage quota exceeded ({used}/{limit}) | Delete unused resources or upgrade |
| `AP-E-305` | Rate Limit Exceeded | Too many requests, retry after {seconds}s | Slow down request rate |
| `AP-E-306` | Invalid Configuration | Configuration is invalid: {reason} | Correct configuration |
| `AP-E-307` | Operation Timeout | Operation timed out after {seconds}s | Retry or contact support |

## Database Layer Errors (DB)

### DB-E-001 to DB-E-099: Connection Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `DB-E-001` | Connection Failed | Cannot connect to database | Check database is running and accessible |
| `DB-E-002` | Connection Timeout | Database connection timeout after {seconds}s | Increase timeout or check database health |
| `DB-E-003` | Connection Pool Exhausted | No available connections in pool | Increase pool size or reduce concurrent operations |
| `DB-E-004` | Authentication Failed | Invalid database credentials | Verify database authentication |

### DB-E-100 to DB-E-199: Query Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `DB-E-101` | Query Failed | Database query failed: {error} | Check query syntax and database state |
| `DB-E-102` | Transaction Aborted | Transaction aborted: {reason} | Retry operation |
| `DB-E-103` | Deadlock Detected | Query deadlock, retrying... | Retry operation |
| `DB-E-104` | Query Timeout | Query exceeded timeout ({seconds}s) | Optimize query or increase timeout |
| `DB-E-105` | Invalid Query | Query syntax error: {error} | Fix query syntax |

### DB-E-200 to DB-E-299: Data Integrity Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `DB-E-201` | Foreign Key Violation | Referenced {resource} does not exist | Create referenced resource first |
| `DB-E-202` | Unique Constraint Violation | Value must be unique for {field} | Use unique value |
| `DB-E-203` | Check Constraint Violation | Value violates check constraint | Provide valid value |
| `DB-E-204` | Not Null Violation | Field '{field}' cannot be null | Provide value for field |
| `DB-E-205` | Data Type Mismatch | Value type mismatch for {field} | Provide correct type |

## Graph/Ontology Reasoning Errors (GR)

### GR-E-001 to GR-E-099: Parsing & Validation

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `GR-E-001` | Invalid OWL Syntax | OWL file has syntax error: {error} | Fix OWL/RDF syntax |
| `GR-E-002` | Unsupported Profile | OWL profile {profile} not supported | Use OWL 2 EL |
| `GR-E-003` | Undefined Class | Class '{uri}' is undefined | Define class or fix reference |
| `GR-E-004` | Undefined Property | Property '{uri}' is undefined | Define property or fix reference |
| `GR-E-005` | Circular Definition | Circular definition detected in {classes} | Remove circular dependencies |
| `GR-E-006` | Invalid Axiom | Axiom is invalid: {reason} | Correct axiom |

### GR-E-100 to GR-E-199: Reasoning Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `GR-E-101` | Reasoning Failed | Ontology reasoning failed: {error} | Check ontology consistency |
| `GR-E-102` | Inconsistent Ontology | Ontology is inconsistent | Fix logical contradictions |
| `GR-E-103` | Reasoning Timeout | Reasoning exceeded timeout ({seconds}s) | Simplify ontology or increase timeout |
| `GR-E-104` | Unsatisfiable Class | Class '{class}' is unsatisfiable | Fix class definitions |
| `GR-E-105` | Memory Exhausted | Reasoning ran out of memory | Reduce ontology size |

## GPU/Physics Computing Errors (GP)

### GP-E-001 to GP-E-099: GPU Initialization

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `GP-E-001` | No GPU Found | No CUDA-capable GPU detected | Install NVIDIA GPU or use CPU fallback |
| `GP-E-002` | GPU Memory Insufficient | GPU memory insufficient ({available}/{required}) | Reduce batch size or use smaller model |
| `GP-E-003` | CUDA Error | CUDA error: {error} | Check GPU drivers and CUDA toolkit |
| `GP-E-004` | GPU Initialization Failed | Failed to initialize GPU: {error} | Verify GPU setup and drivers |
| `GP-E-005` | GPU Driver Mismatch | CUDA version mismatch ({required} vs {available}) | Update GPU drivers |

### GP-E-100 to GP-E-199: Computation Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `GP-E-101` | Kernel Launch Failed | CUDA kernel launch failed: {error} | Check kernel parameters |
| `GP-E-102` | Memory Copy Failed | GPU memory copy failed: {error} | Check memory availability |
| `GP-E-103` | Computation Timeout | GPU computation timeout ({seconds}s) | Increase timeout or optimize code |
| `GP-E-104` | Invalid Buffer | Invalid GPU buffer: {error} | Reallocate buffer |
| `GP-E-105` | Synchronization Failed | GPU synchronization failed: {error} | Check GPU health |

## WebSocket/Network Errors (WS)

### WS-E-001 to WS-E-099: Connection Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `WS-E-001` | Connection Refused | WebSocket connection refused | Check server is running |
| `WS-E-002` | Connection Timeout | WebSocket connection timeout | Check network connectivity |
| `WS-E-003` | Connection Closed | WebSocket connection closed unexpectedly | Reconnect to server |
| `WS-E-004` | Invalid Protocol Version | Protocol version mismatch | Use compatible client version |
| `WS-E-005` | Handshake Failed | WebSocket handshake failed: {error} | Check server configuration |

### WS-E-100 to WS-E-199: Message Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `WS-E-101` | Invalid Message Format | Received invalid message format | Check message encoding |
| `WS-E-102` | Message Too Large | Message size ({size}) exceeds limit ({limit}) | Send smaller messages |
| `WS-E-103` | Message Parsing Failed | Failed to parse message: {error} | Verify message format |
| `WS-E-104` | Unknown Message Type | Unknown message type: {type} | Use valid message type |

## Storage/File Management Errors (ST)

### ST-E-001 to ST-E-099: File Operation Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `ST-E-001` | File Not Found | File '{path}' not found | Verify file path |
| `ST-E-002` | File Access Denied | Access denied for '{path}' | Check file permissions |
| `ST-E-003` | File Already Exists | File '{path}' already exists | Use different filename or overwrite |
| `ST-E-004` | File Write Failed | Failed to write file '{path}': {error} | Check disk space and permissions |
| `ST-E-005` | File Read Failed | Failed to read file '{path}': {error} | Check file exists and is readable |

### ST-E-100 to ST-E-199: Storage Errors

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `ST-E-101` | Storage Full | Storage is full ({used}/{available}) | Delete unused files |
| `ST-E-102` | Storage Unavailable | Storage service unavailable | Check S3/storage backend |
| `ST-E-103` | Upload Failed | File upload failed: {error} | Retry upload |
| `ST-E-104` | Download Failed | File download failed: {error} | Retry download |

## Fatal Errors (F)

### System-Level Failures

| Code | Name | Message | Resolution |
|------|------|---------|-----------|
| `AP-F-001` | Critical Exception | Unhandled exception: {error} | Check logs and restart |
| `DB-F-001` | Database Corruption | Database corruption detected | Contact support, restore backup |
| `GP-F-001` | GPU Hardware Failure | GPU hardware failure detected | Replace GPU hardware |
| `ST-F-001` | File System Failure | File system failure detected | Check disk health, restore backup |

## Error Handling Best Practices

### Client-Side Handling

```typescript
async function apiCall(endpoint: string, data: unknown) {
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      const error = await response.json();
      const errorCode = error.code; // e.g., "AP-E-001"
      const severity = errorCode[3]; // 'E' for error

      switch (severity) {
        case 'E': // Recoverable error
          showUserMessage(error.message);
          // Retry logic
          break;
        case 'F': // Fatal error
          showFatalError(error.message);
          break;
        case 'W': // Warning
          logWarning(error.message);
          break;
      }
    }
  } catch (e) {
    console.error('Network error:', e);
  }
}
```

### Server-Side Logging

```typescript
logger.error({
  code: 'GR-E-102',
  message: 'Ontology consistency violation',
  details: {
    classes: ['Class1', 'Class2'],
    axioms: [...]
  },
  timestamp: new Date(),
  userId: request.userId
});
```

## Related Documentation

-  - Complete REST API reference
-  - Binary protocol specification
-  - Common issues and solutions
-  - System design

---

**Last Updated**: 2025-11-04
**Status**: Production Reference
**Maintenance**: Errors added/updated as system evolves

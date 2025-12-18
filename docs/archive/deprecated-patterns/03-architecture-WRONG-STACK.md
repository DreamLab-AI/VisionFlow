---
title: Architecture Overview (OBSOLETE - WRONG STACK)
description: ❌ OBSOLETE - This described a PostgreSQL/Redis/Vue.js architecture that was NEVER IMPLEMENTED
category: explanation
tags:
  - architecture
  - neo4j
  - rust
  - react
updated-date: 2025-12-18
difficulty-level: advanced
---


# ❌ OBSOLETE: Wrong Technology Stack

## ⚠️ DO NOT USE THIS DOCUMENT ⚠️

**Status**: DEPRECATED - Never Implemented
**Archived**: December 2, 2025
**Issue**: This document describes a PostgreSQL + Redis + Vue.js architecture that was **never built**

### Actual Architecture

The real system uses:
- **Database**: Neo4j (graph database), not PostgreSQL
- **Cache**: No Redis - Neo4j is the source of truth
- **Frontend**: React + TypeScript, not Vue.js
- **Backend**: Rust + Actix-web, not Node.js
- **Auth**: Nostr protocol, not JWT

### Current Documentation

For the **actual, current architecture**, see:
- **Primary**: `/docs/ARCHITECTURE_OVERVIEW.md`
- **Detailed**: `/docs/explanations/architecture/`
- **Components**: `/docs/explanations/architecture/core/`

---

## Historical Content Below (DO NOT USE)

# Architecture Overview (WRONG STACK - NEVER IMPLEMENTED)

## System Architecture

VisionFlow is built as a distributed, microservices-based application designed for scalability, maintainability, and high availability.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WebUI[Web UI]
        CLI[CLI Tool]
        SDK[SDK/API Clients]
    end

    subgraph "API Gateway"
        NGINX[NGINX/Load Balancer]
        API[API Server]
    end

    subgraph "Application Layer"
        Auth[Authentication Service]
        Project[Project Service]
        Asset[Asset Service]
        Processing[Processing Service]
        Worker[Background Workers]
    end

    subgraph "Data Layer"
        DB[(PostgreSQL)]
        Cache[(Redis)]
        Queue[(Message Queue)]
        Storage[(Object Storage)]
    end

    subgraph "External Services"
        Email[Email Service]
        Monitor[Monitoring]
        Logging[Logging]
    end

    WebUI --> NGINX
    CLI --> API
    SDK --> API
    NGINX --> API

    API --> Auth
    API --> Project
    API --> Asset
    API --> Processing

    Processing --> Worker
    Worker --> Queue

    Auth --> DB
    Project --> DB
    Asset --> DB
    Processing --> DB

    API --> Cache
    Worker --> Cache

    Asset --> Storage
    Processing --> Storage

    API --> Email
    API --> Monitor
    Worker --> Logging

    style WebUI fill:#E3F2FD
    style API fill:#C8E6C9
    style DB fill:#FFE4B5
    style Cache fill:#FFE4B5
    style Storage fill:#FFE4B5
```

## Component Architecture

### 1. API Server

**Responsibilities**:
- HTTP request handling
- Request validation
- Authentication/Authorization
- Business logic coordination
- Response formatting

**Architecture Pattern**: Layered Architecture

```mermaid
graph LR
    A[Routes] --> B[Controllers]
    B --> C[Services]
    C --> D[Models]
    C --> E[External APIs]

    F[Middleware] -.-> A
    F -.-> B

    style A fill:#E3F2FD
    style B fill:#C8E6C9
    style C fill:#FFF9C4
    style D fill:#FFE4B5
```

**Layers**:
1. **Routes**: Endpoint definitions
2. **Middleware**: Request processing pipeline
3. **Controllers**: Request/response handling
4. **Services**: Business logic
5. **Models**: Data access layer

**Example Flow**:
```javascript
// Route Definition
router.post('/projects', validateProject, ProjectController.create);

// Controller
async create(req, res) {
  const project = await ProjectService.create(req.body);
  res.json(project);
}

// Service
async create(data) {
  // Business logic
  const project = await Project.create(data);
  await this.notifyCreation(project);
  return project;
}

// Model
class Project extends Model {
  static async create(data) {
    return await db.projects.insert(data);
  }
}
```

### 2. Web UI Architecture

**Pattern**: Component-Based Architecture (Vue.js)

```mermaid
graph TD
    A[App.vue] --> B[Router]
    B --> C[Views/Pages]
    C --> D[Feature Components]
    D --> E[Common Components]

    F[Store/State] -.-> C
    F -.-> D

    G[Services/API] -.-> F
    G -.-> D

    style A fill:#E3F2FD
    style C fill:#C8E6C9
    style F fill:#FFF9C4
    style G fill:#FFE4B5
```

**Component Hierarchy**:
```
App.vue
├── Layout Components
│   ├── Header
│   ├── Sidebar
│   └── Footer
├── Page/View Components
│   ├── Dashboard
│   ├── ProjectList
│   └── AssetManager
└── Feature Components
    ├── ProjectCard
    ├── AssetUploader
    └── ProcessingStatus
```

**State Management**:
```javascript
// Store Structure
store/
├── index.js              // Root store
└── modules/
    ├── auth.js          // Authentication state
    ├── projects.js      // Projects state
    └── assets.js        // Assets state
```

### 3. Background Worker Architecture

**Pattern**: Queue-Based Processing

```mermaid
sequenceDiagram
    participant API
    participant Queue
    participant Worker
    participant Storage
    participant DB

    API->>Queue: Enqueue Job
    API->>DB: Update Status (Queued)
    Queue->>Worker: Dequeue Job
    Worker->>DB: Update Status (Processing)
    Worker->>Storage: Process Data
    Worker->>DB: Update Status (Complete)
    Worker->>Queue: Send Result Event
```

**Worker Types**:
1. **Processing Workers**: Data processing tasks
2. **Export Workers**: Data export operations
3. **Notification Workers**: Email/push notifications
4. **Cleanup Workers**: Maintenance tasks

**Job Processing**:
```javascript
class ProcessingWorker {
  async process(job) {
    try {
      await this.updateStatus(job.id, 'processing');
      const result = await this.executeJob(job);
      await this.updateStatus(job.id, 'completed');
      return result;
    } catch (error) {
      await this.handleError(job, error);
      throw error;
    }
  }
}
```

## Data Architecture

### Database Schema

```mermaid
erDiagram
    USERS ||--o{ PROJECTS : creates
    USERS ||--o{ SESSIONS : has
    PROJECTS ||--o{ ASSETS : contains
    PROJECTS ||--o{ JOBS : has
    ASSETS ||--o{ VERSIONS : has
    JOBS ||--o{ JOB_LOGS : generates

    USERS {
        uuid id PK
        string email
        string password_hash
        string role
        timestamp created_at
    }

    PROJECTS {
        uuid id PK
        uuid user_id FK
        string name
        jsonb config
        string status
        timestamp created_at
    }

    ASSETS {
        uuid id PK
        uuid project_id FK
        string name
        string type
        bigint size
        string storage_path
        timestamp created_at
    }

    JOBS {
        uuid id PK
        uuid project_id FK
        string type
        string status
        jsonb params
        jsonb result
        timestamp created_at
    }
```

### Data Flow

```mermaid
flowchart LR
    A[User Upload] --> B[Validation]
    B --> C[Temporary Storage]
    C --> D[Queue Job]
    D --> E[Process Data]
    E --> F[Permanent Storage]
    F --> G[Update Database]
    G --> H[Notify User]

    style A fill:#E3F2FD
    style E fill:#C8E6C9
    style F fill:#FFE4B5
    style H fill:#FFF9C4
```

## Security Architecture

### Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Auth
    participant DB
    participant Cache

    Client->>API: Login Request
    API->>Auth: Validate Credentials
    Auth->>DB: Check User
    DB-->>Auth: User Data
    Auth->>Auth: Generate JWT
    Auth->>Cache: Store Session
    Auth-->>API: JWT Token
    API-->>Client: Token + User Info

    Note over Client,Cache: Subsequent Requests

    Client->>API: Request + JWT
    API->>Cache: Validate Token
    Cache-->>API: Session Valid
    API->>API: Process Request
    API-->>Client: Response
```

### Authorization

```mermaid
graph TD
    A[Request] --> B{Authenticated?}
    B -->|No| C[401 Unauthorized]
    B -->|Yes| D{Has Permission?}
    D -->|No| E[403 Forbidden]
    D -->|Yes| F[Process Request]

    G[RBAC] -.-> D
    H[Resource Ownership] -.-> D

    style C fill:#FFB6C6
    style E fill:#FFB6C6
    style F fill:#90EE90
```

**Permission Model**:
```javascript
// Role-Based Access Control
const permissions = {
  admin: ['*'],
  operator: ['projects.*', 'assets.*', 'users.read'],
  user: ['projects.own', 'assets.own'],
  guest: ['*.read']
};

// Check permission
function hasPermission(user, action, resource) {
  const userPerms = permissions[user.role];
  return userPerms.includes(action) ||
         userPerms.includes(`${resource}.*`) ||
         userPerms.includes('*');
}
```

## Scalability Patterns

### Horizontal Scaling

```mermaid
graph TB
    LB[Load Balancer]

    LB --> API1[API Instance 1]
    LB --> API2[API Instance 2]
    LB --> API3[API Instance 3]

    API1 --> DB[(Database Cluster)]
    API2 --> DB
    API3 --> DB

    API1 --> Cache[(Redis Cluster)]
    API2 --> Cache
    API3 --> Cache

    W1[Worker 1] --> Queue[Message Queue]
    W2[Worker 2] --> Queue
    W3[Worker 3] --> Queue

    Queue --> Storage[(Shared Storage)]

    style LB fill:#4A90E2
    style DB fill:#90EE90
    style Cache fill:#FFE4B5
    style Storage fill:#FFF9C4
```

### Caching Strategy

**Multi-Level Caching**:
```javascript
// Level 1: In-Memory (Fast, Limited)
const memoryCache = new Map();

// Level 2: Redis (Fast, Distributed)
const redisCache = new Redis();

// Level 3: Database (Slow, Authoritative)
const database = new Database();

async function getData(key) {
  // Try memory cache
  if (memoryCache.has(key)) {
    return memoryCache.get(key);
  }

  // Try Redis cache
  const cached = await redisCache.get(key);
  if (cached) {
    memoryCache.set(key, cached);
    return cached;
  }

  // Fetch from database
  const data = await database.get(key);
  await redisCache.set(key, data, 'EX', 3600);
  memoryCache.set(key, data);
  return data;
}
```

## Integration Patterns

### External Service Integration

```mermaid
graph LR
    A[VisionFlow] --> B[Adapter Layer]
    B --> C[Service A]
    B --> D[Service B]
    B --> E[Service C]

    B --> F[Circuit Breaker]
    B --> G[Retry Logic]
    B --> H[Fallback]

    style A fill:#E3F2FD
    style B fill:#C8E6C9
    style F fill:#FFE4B5
```

**Adapter Pattern**:
```javascript
// Storage Adapter Interface
class IStorageAdapter {
  async upload(file, path) { throw new Error('Not implemented'); }
  async download(path) { throw new Error('Not implemented'); }
  async delete(path) { throw new Error('Not implemented'); }
}

// S3 Implementation
class S3Adapter extends IStorageAdapter {
  async upload(file, path) {
    return await s3.putObject({
      Bucket: this.bucket,
      Key: path,
      Body: file
    });
  }
}

// Local Implementation
class LocalAdapter extends IStorageAdapter {
  async upload(file, path) {
    return await fs.writeFile(path, file);
  }
}

// Usage
const storage = config.storage === 's3'
  ? new S3Adapter()
  : new LocalAdapter();
```

## Event-Driven Architecture

```mermaid
sequenceDiagram
    participant Service A
    participant Event Bus
    participant Service B
    participant Service C

    Service A->>Event Bus: Emit Event
    Event Bus->>Service B: Deliver Event
    Event Bus->>Service C: Deliver Event
    Service B->>Service B: Process Event
    Service C->>Service C: Process Event
```

**Event System**:
```javascript
class EventBus {
  constructor() {
    this.subscribers = new Map();
  }

  subscribe(event, handler) {
    if (!this.subscribers.has(event)) {
      this.subscribers.set(event, []);
    }
    this.subscribers.get(event).push(handler);
  }

  async emit(event, data) {
    const handlers = this.subscribers.get(event) || [];
    await Promise.all(handlers.map(h => h(data)));
  }
}

// Usage
eventBus.subscribe('project.created', async (project) => {
  await notificationService.sendEmail(project.owner, 'Project created');
});

eventBus.emit('project.created', project);
```

## Deployment Architecture

### Production Deployment

```mermaid
graph TB
    subgraph "Production Environment"
        subgraph "DMZ"
            LB[Load Balancer]
            CDN[CDN]
        end

        subgraph "Application Tier"
            API1[API Server 1]
            API2[API Server 2]
            Worker1[Worker 1]
            Worker2[Worker 2]
        end

        subgraph "Data Tier"
            DBPrimary[(Primary DB)]
            DBReplica[(Replica DB)]
            Redis[(Redis)]
            Queue[Message Queue]
        end

        subgraph "Storage Tier"
            S3[(Object Storage)]
        end
    end

    CDN --> LB
    LB --> API1
    LB --> API2

    API1 --> DBPrimary
    API2 --> DBPrimary
    API1 --> DBReplica
    API2 --> DBReplica

    API1 --> Redis
    API2 --> Redis

    Worker1 --> Queue
    Worker2 --> Queue

    API1 --> S3
    API2 --> S3
    Worker1 --> S3
    Worker2 --> S3
```

## Performance Optimization

### Request Lifecycle Optimization

```mermaid
graph LR
    A[Request] --> B[CDN Cache]
    B -->|Miss| C[API Cache]
    C -->|Miss| D[Database]

    E[Response] --> F[Compress]
    F --> G[Cache]
    G --> H[Client]

    style B fill:#90EE90
    style C fill:#FFE4B5
    style D fill:#FFB6C6
```

### Database Optimization

**Indexing Strategy**:
```sql
-- Frequently queried columns
CREATE INDEX idx-projects-user-id ON projects(user-id);
CREATE INDEX idx-assets-project-id ON assets(project-id);

-- Composite indexes for common queries
CREATE INDEX idx-projects-user-status ON projects(user-id, status);

-- Partial indexes for specific queries
CREATE INDEX idx-active-jobs ON jobs(created-at)
WHERE status = 'processing';
```

## Monitoring & Observability

### Metrics Collection

```mermaid
graph LR
    A[Application] --> B[Metrics Collector]
    B --> C[Prometheus]
    C --> D[Grafana]

    A --> E[Log Aggregator]
    E --> F[Elasticsearch]
    F --> G[Kibana]

    A --> H[Trace Collector]
    H --> I[Jaeger]

    style B fill:#E3F2FD
    style C fill:#C8E6C9
    style F fill:#FFF9C4
```

## Design Principles

1. **Single Responsibility**: Each module has one reason to change
2. **Dependency Inversion**: Depend on abstractions, not implementations
3. **Open/Closed**: Open for extension, closed for modification
4. **Interface Segregation**: Many specific interfaces vs. one general
5. **DRY**: Don't Repeat Yourself
6. **KISS**: Keep It Simple, Stupid
7. **YAGNI**: You Aren't Gonna Need It

---

---

## Related Documentation

- [Deprecated Patterns Archive](README.md)
- [Hexagonal Architecture Ports - Overview](../../explanations/architecture/ports/01-overview.md)
- [KnowledgeGraphRepository Port](../../explanations/architecture/ports/03-knowledge-graph-repository.md)
- [Settings API Authentication](../../guides/features/settings-authentication.md)
- [Ontology Storage Architecture](../../explanations/architecture/ontology-storage-architecture.md)

## Next Steps (OBSOLETE LINKS - DO NOT FOLLOW)

❌ **These links are from the old, wrong documentation. For current documentation:**
- Learn about [Current Architecture](/docs/ARCHITECTURE_OVERVIEW.md)
- Review [Testing Strategy](/docs/guides/developer/testing-guide.md)
- Understand [Contributing Guidelines](/docs/guides/contributing.md)

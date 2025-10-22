# Hexser Crate - Comprehensive Research Guide

**Version:** 0.4.7
**License:** MIT OR Apache-2.0
**Documentation Coverage:** 94.89%
**Research Date:** 2025-10-22

## Table of Contents
1. [Overview](#overview)
2. [Architecture Layers](#architecture-layers)
3. [Core Traits & Types](#core-traits--types)
4. [Derive Macros](#derive-macros)
5. [CQRS Pattern: Directives & Queries](#cqrs-pattern-directives--queries)
6. [Ports & Adapters](#ports--adapters)
7. [Dependency Injection Container](#dependency-injection-container)
8. [Graph-Based Introspection](#graph-based-introspection)
9. [Registration Macros](#registration-macros)
10. [Features & Dependencies](#features--dependencies)
11. [Implementation Examples](#implementation-examples)
12. [Best Practices](#best-practices)
13. [Common Pitfalls](#common-pitfalls)

---

## Overview

Hexser is a Rust crate that provides **"Zero-boilerplate hexagonal architecture with graph-based introspection."** It enables developers to implement clean hexagonal (ports and adapters) architecture with CQRS patterns, minimal boilerplate code, and comprehensive architectural introspection capabilities.

**Key Features:**
- Nine derive macros for automatic trait implementation
- CQRS support with Directive (Command) and Query patterns
- Compile-time component registration via `inventory` crate
- Thread-safe dependency injection container
- Graph-based architecture visualization and analysis
- CloudEvents v1.0 event bus implementation
- Optional async support with tokio

---

## Architecture Layers

Hexser organizes code across **five architectural layers**, each with distinct responsibilities:

### 1. Domain Layer
**Purpose:** Core business logic (entities, value objects, aggregates)

**Characteristics:**
- Independent from infrastructure and frameworks
- Contains pure business rules
- No external dependencies
- Immutable value objects
- Entities with unique identities

### 2. Ports Layer
**Purpose:** Interface definitions for external interactions

**Characteristics:**
- Abstract contracts that adapters implement
- Input ports (primary/driving ports)
- Output ports (secondary/driven ports)
- Repository interfaces
- Use case definitions
- Query interfaces

### 3. Adapters Layer
**Purpose:** Concrete implementations of port interfaces

**Characteristics:**
- Technology-specific implementations
- Database adapters
- API clients
- UI adapters
- Event bus implementations
- Data transformation/mapping

### 4. Application Layer
**Purpose:** Use case orchestration and command/query handling

**Characteristics:**
- Coordinates domain objects
- Implements DirectiveHandler for commands
- Implements QueryHandler for queries
- Transaction boundaries
- Application-level validation
- No business logic (delegated to domain)

### 5. Infrastructure Layer
**Purpose:** External system configuration and cross-cutting concerns

**Characteristics:**
- Configuration management
- Logging setup
- Database connection pools
- External service configuration
- Startup/shutdown orchestration

---

## Core Traits & Types

### Domain Layer Traits

#### `HexEntity`
**Purpose:** Base trait for domain entities with unique identity

```rust
pub trait HexEntity {
    type Id;
    // Identity-based equality, not attribute-based
}
```

**Characteristics:**
- Objects with unique identity
- Two entities with same ID are considered equal
- Identity persists through attribute changes
- Mutable state

**Example:**
```rust
#[derive(HexEntity)]
struct User {
    id: UserId,
    email: String,
    name: String,
}

impl HexEntity for User {
    type Id = UserId;
}
```

#### `HexValueItem`
**Purpose:** Value object trait for immutable domain concepts

```rust
pub trait HexValueItem {
    // Value-based equality
}
```

**Characteristics:**
- Immutable objects
- Equality based on all attributes
- No unique identity
- Can be freely replaced

**Example:**
```rust
#[derive(HexValueItem, Clone, PartialEq)]
struct Email(String);

#[derive(HexValueItem, Clone, PartialEq)]
struct Money {
    amount: Decimal,
    currency: Currency,
}
```

#### `Aggregate`
**Purpose:** Aggregate root pattern - consistency boundary in the domain

```rust
pub trait Aggregate: HexEntity {
    // Transactional boundary
    // Ensures consistency of entity cluster
}
```

**Characteristics:**
- Root entity that controls access to aggregate
- Enforces business invariants
- Transactional consistency boundary
- External references only via ID

**Example:**
```rust
#[derive(HexAggregate)]
struct Order {
    id: OrderId,
    customer: CustomerId, // Reference by ID only
    items: Vec<OrderItem>, // Internal entities
    total: Money,
}
```

#### `DomainEvent`
**Purpose:** Represents something that happened in the domain

**Characteristics:**
- Immutable
- Past tense naming
- Contains event data
- Timestamp included

**Example:**
```rust
#[derive(HexDomain)]
struct OrderPlaced {
    order_id: OrderId,
    customer_id: CustomerId,
    total: Money,
    occurred_at: DateTime<Utc>,
}
```

#### `DomainService`
**Purpose:** Operations that don't naturally fit on entities or value objects

**Characteristics:**
- Stateless operations
- Works with multiple domain objects
- Pure domain logic

---

### Ports Layer Traits

#### `Repository`
**Purpose:** Persistence abstraction for aggregates

```rust
pub trait Repository<T: Aggregate> {
    fn save(&self, aggregate: &T) -> HexResult<()>;
    fn find_by_id(&self, id: &T::Id) -> HexResult<Option<T>>;
    fn delete(&self, id: &T::Id) -> HexResult<()>;
}
```

**Characteristics:**
- Collection-like interface
- Aggregate-focused (not individual entities)
- Hides persistence technology
- Returns domain objects

**Example:**
```rust
trait UserRepository: Repository<User> {
    fn find_by_email(&self, email: &Email) -> HexResult<Option<User>>;
    fn find_active_users(&self) -> HexResult<Vec<User>>;
}
```

#### `InputPort`
**Purpose:** Application entry points for receiving requests

**Characteristics:**
- Primary/driving ports
- Called by external actors
- Use case interfaces
- Command/Query entry points

#### `OutputPort`
**Purpose:** Secondary adapters for external system interactions

**Characteristics:**
- Driven ports
- Called by application
- Database interfaces
- External API interfaces
- Notification interfaces

#### `UseCase`
**Purpose:** Business operation definitions

```rust
pub trait UseCase {
    type Input;
    type Output;

    fn execute(&self, input: Self::Input) -> HexResult<Self::Output>;
}
```

#### `Query`
**Purpose:** Read-only operations following CQRS patterns

**Characteristics:**
- No side effects
- Optimized for reading
- May bypass domain model
- Returns DTOs

#### `EventPublisher` / `EventSubscriber`
**Purpose:** CloudEvents v1.0-compliant event handling

**Characteristics:**
- Async event propagation
- Decoupled components
- CloudEvents standard compliance

---

### Adapters Layer Traits

#### `Adapter`
**Purpose:** Marker trait for port implementations

```rust
pub trait Adapter {
    // Marker trait identifying adapter implementations
}
```

**Characteristics:**
- Technology-specific implementations
- Implements port interfaces
- Handles data transformation
- Isolated from domain

**Example:**
```rust
struct PostgresUserRepository {
    pool: PgPool,
}

impl Adapter for PostgresUserRepository {}

impl Repository<User> for PostgresUserRepository {
    fn save(&self, user: &User) -> HexResult<()> {
        // PostgreSQL-specific implementation
    }

    fn find_by_id(&self, id: &UserId) -> HexResult<Option<User>> {
        // PostgreSQL-specific implementation
    }
}
```

#### `Mapper`
**Purpose:** Data transformation between layers

```rust
pub trait Mapper<From, To> {
    fn map(&self, from: From) -> To;
}
```

**Characteristics:**
- Converts between domain and external models
- DTO ↔ Entity mapping
- Database rows ↔ Domain objects
- API requests ↔ Domain objects

**Example:**
```rust
struct UserMapper;

impl Mapper<UserRow, User> for UserMapper {
    fn map(&self, row: UserRow) -> User {
        User {
            id: UserId::from(row.id),
            email: Email(row.email),
            name: row.name,
        }
    }
}
```

---

### Application Layer Traits

#### `Application`
**Purpose:** Main application orchestrator and entry point

```rust
pub trait Application {
    // Marks top-level entry points
    // Coordinates lifecycle management
}
```

**Characteristics:**
- System entry point
- Wires components together
- No business logic
- Delegates to use cases

#### `Directive` (formerly Command)
**Purpose:** Represents write operations in CQRS pattern

```rust
pub trait Directive {
    // Write operation request
}
```

**Characteristics:**
- Imperative naming (CreateOrder, UpdateUser)
- Represents intention to change state
- Validated before processing
- May return success/failure

**Example:**
```rust
#[derive(HexDirective)]
struct CreateUserDirective {
    email: String,
    name: String,
    password: String,
}

#[derive(HexDirective)]
struct PlaceOrderDirective {
    customer_id: Uuid,
    items: Vec<OrderItemData>,
}
```

#### `DirectiveHandler`
**Purpose:** Processes write operations triggered by directives

```rust
pub trait DirectiveHandler<D: Directive> {
    type Output;

    fn handle(&self, directive: D) -> HexResult<Self::Output>;
}
```

**Characteristics:**
- Separates request from execution
- Single responsibility per handler
- Transactional boundary
- May emit domain events

**Example:**
```rust
struct CreateUserDirectiveHandler {
    user_repo: Arc<dyn UserRepository>,
    event_bus: Arc<dyn EventPublisher>,
}

impl DirectiveHandler<CreateUserDirective> for CreateUserDirectiveHandler {
    type Output = UserId;

    fn handle(&self, directive: CreateUserDirective) -> HexResult<Self::Output> {
        // 1. Validate
        let email = Email::new(&directive.email)?;

        // 2. Create domain object
        let user = User::new(email, directive.name, directive.password)?;

        // 3. Persist
        self.user_repo.save(&user)?;

        // 4. Publish event
        self.event_bus.publish(UserCreated { user_id: user.id })?;

        Ok(user.id)
    }
}
```

#### `QueryHandler`
**Purpose:** Handles read operations in CQRS pattern

```rust
pub trait QueryHandler<Q: Query> {
    type Output;

    fn handle(&self, query: Q) -> HexResult<Self::Output>;
}
```

**Characteristics:**
- Read-only operations
- No side effects
- May bypass domain layer
- Optimized for reading
- Returns DTOs

**Example:**
```rust
#[derive(HexQuery)]
struct GetUserByEmailQuery {
    email: String,
}

struct GetUserByEmailQueryHandler {
    user_repo: Arc<dyn UserRepository>,
}

impl QueryHandler<GetUserByEmailQuery> for GetUserByEmailQueryHandler {
    type Output = Option<UserDto>;

    fn handle(&self, query: GetUserByEmailQuery) -> HexResult<Self::Output> {
        let email = Email::new(&query.email)?;
        let user = self.user_repo.find_by_email(&email)?;
        Ok(user.map(|u| UserDto::from(u)))
    }
}
```

---

### Infrastructure Layer Traits

#### `Config`
**Purpose:** Configuration abstraction for external concerns

**Characteristics:**
- Environment-based configuration
- Secret management
- Service endpoints
- Feature flags

---

## Derive Macros

Hexser provides **nine derive macros** for zero-boilerplate trait implementation:

### 1. `#[derive(HexEntity)]`
**Implements:** `HexEntity` trait

**Usage:**
```rust
#[derive(HexEntity)]
struct User {
    id: UserId,
    email: Email,
    name: String,
}
```

### 2. `#[derive(HexValueItem)]`
**Implements:** `HexValueItem` trait

**Usage:**
```rust
#[derive(HexValueItem, Clone, PartialEq)]
struct Email(String);
```

### 3. `#[derive(HexAggregate)]`
**Implements:** `Aggregate` trait (also implements `HexEntity`)

**Usage:**
```rust
#[derive(HexAggregate)]
struct Order {
    id: OrderId,
    customer_id: CustomerId,
    items: Vec<OrderItem>,
}
```

### 4. `#[derive(HexDomain)]`
**Marks:** Domain-layer components

**Usage:**
```rust
#[derive(HexDomain)]
struct OrderPlaced {
    order_id: OrderId,
    occurred_at: DateTime<Utc>,
}
```

### 5. `#[derive(HexPort)]`
**Implements:** Port interface traits

**Usage:**
```rust
#[derive(HexPort)]
trait UserRepository: Repository<User> {
    fn find_by_email(&self, email: &Email) -> HexResult<Option<User>>;
}
```

### 6. `#[derive(HexRepository)]`
**Implements:** `Repository` trait

**Usage:**
```rust
#[derive(HexRepository)]
struct InMemoryUserRepository {
    users: HashMap<UserId, User>,
}
```

### 7. `#[derive(HexAdapter)]`
**Implements:** `Adapter` trait

**Usage:**
```rust
#[derive(HexAdapter)]
struct PostgresUserRepository {
    pool: PgPool,
}
```

### 8. `#[derive(HexDirective)]`
**Implements:** `Directive` trait

**Usage:**
```rust
#[derive(HexDirective)]
struct CreateUserDirective {
    email: String,
    name: String,
}
```

### 9. `#[derive(HexQuery)]`
**Implements:** `Query` trait

**Usage:**
```rust
#[derive(HexQuery)]
struct FindUserByIdQuery {
    user_id: Uuid,
}
```

---

## CQRS Pattern: Directives & Queries

Hexser implements **Command Query Responsibility Segregation (CQRS)** using:
- **Directives** (Commands) for write operations
- **Queries** for read operations

### Why "Directive" instead of "Command"?

The framework renamed "Command" to "Directive" to **better express intent** and avoid confusion with other command patterns in software.

### Directive Pattern (Write Operations)

**Flow:**
```
External Actor → Directive → DirectiveHandler → Domain Logic → Repository → Database
                                               ↓
                                          EventBus (optional)
```

**Example: User Registration**

```rust
// 1. Define the directive
#[derive(HexDirective)]
struct RegisterUserDirective {
    email: String,
    password: String,
    name: String,
}

// 2. Implement the handler
struct RegisterUserDirectiveHandler {
    user_repo: Arc<dyn UserRepository>,
    password_hasher: Arc<dyn PasswordHasher>,
    event_bus: Arc<dyn EventPublisher>,
}

impl DirectiveHandler<RegisterUserDirective> for RegisterUserDirectiveHandler {
    type Output = UserId;

    fn handle(&self, directive: RegisterUserDirective) -> HexResult<Self::Output> {
        // Validate
        let email = Email::new(&directive.email)?;
        if self.user_repo.find_by_email(&email)?.is_some() {
            return Err(Hexserror::Conflict("Email already exists".into()));
        }

        // Create domain object
        let hashed_password = self.password_hasher.hash(&directive.password)?;
        let user = User::new(email, directive.name, hashed_password)?;

        // Persist
        self.user_repo.save(&user)?;

        // Publish event
        self.event_bus.publish(UserRegistered {
            user_id: user.id,
            email: user.email.clone(),
            occurred_at: Utc::now(),
        })?;

        Ok(user.id)
    }
}

// 3. Register the handler
hex_register_application!(RegisterUserDirectiveHandler, Role::DirectiveHandler);
```

### Query Pattern (Read Operations)

**Flow:**
```
External Actor → Query → QueryHandler → Read-Optimized Data Store → DTO
```

**Example: User Lookup**

```rust
// 1. Define the query
#[derive(HexQuery)]
struct GetUserProfileQuery {
    user_id: Uuid,
}

// 2. Define the DTO
#[derive(Serialize)]
struct UserProfileDto {
    id: Uuid,
    email: String,
    name: String,
    created_at: DateTime<Utc>,
}

// 3. Implement the handler
struct GetUserProfileQueryHandler {
    user_repo: Arc<dyn UserRepository>,
}

impl QueryHandler<GetUserProfileQuery> for GetUserProfileQueryHandler {
    type Output = UserProfileDto;

    fn handle(&self, query: GetUserProfileQuery) -> HexResult<Self::Output> {
        let user = self.user_repo
            .find_by_id(&UserId::from(query.user_id))?
            .ok_or_else(|| Hexserror::NotFound("User not found".into()))?;

        Ok(UserProfileDto {
            id: user.id.into(),
            email: user.email.0,
            name: user.name,
            created_at: user.created_at,
        })
    }
}

// 4. Register the handler
hex_register_application!(GetUserProfileQueryHandler, Role::QueryHandler);
```

### CQRS with Separate Read/Write Models

**Advanced pattern:** Use different data stores for reads and writes

```rust
// Write side: Full domain model
struct CreateOrderDirectiveHandler {
    order_repo: Arc<dyn OrderRepository>, // Write to normalized DB
    event_bus: Arc<dyn EventPublisher>,
}

// Read side: Denormalized projections
struct GetOrderDetailsQueryHandler {
    read_db: Arc<dyn ReadDatabase>, // Read from optimized view
}

impl QueryHandler<GetOrderDetailsQuery> for GetOrderDetailsQueryHandler {
    type Output = OrderDetailsDto;

    fn handle(&self, query: GetOrderDetailsQuery) -> HexResult<Self::Output> {
        // Query denormalized read model directly
        let details = self.read_db.query_order_details(query.order_id)?;
        Ok(details)
    }
}

// Event handler keeps read model in sync
struct OrderCreatedEventHandler {
    read_db: Arc<dyn ReadDatabase>,
}

impl EventSubscriber for OrderCreatedEventHandler {
    fn on_event(&self, event: OrderCreated) -> HexResult<()> {
        // Update denormalized read model
        self.read_db.insert_order_view(event.into())?;
        Ok(())
    }
}
```

---

## Ports & Adapters

### Defining Ports (Interfaces)

Ports are abstract interfaces that define contracts:

```rust
// Input Port (Primary/Driving)
trait OrderService {
    fn place_order(&self, directive: PlaceOrderDirective) -> HexResult<OrderId>;
    fn cancel_order(&self, order_id: OrderId) -> HexResult<()>;
}

// Output Port (Secondary/Driven)
trait PaymentGateway {
    fn process_payment(&self, order_id: OrderId, amount: Money) -> HexResult<PaymentId>;
    fn refund_payment(&self, payment_id: PaymentId) -> HexResult<()>;
}

trait NotificationService {
    fn send_email(&self, to: Email, subject: String, body: String) -> HexResult<()>;
}
```

### Implementing Adapters

Adapters provide concrete implementations:

```rust
// Adapter for payment processing
#[derive(HexAdapter)]
struct StripePaymentGateway {
    api_key: String,
    client: StripeClient,
}

impl PaymentGateway for StripePaymentGateway {
    fn process_payment(&self, order_id: OrderId, amount: Money) -> HexResult<PaymentId> {
        // Stripe-specific implementation
        let stripe_result = self.client.charge(
            amount.amount,
            amount.currency,
            &order_id.to_string(),
        )?;
        Ok(PaymentId::from(stripe_result.id))
    }

    fn refund_payment(&self, payment_id: PaymentId) -> HexResult<()> {
        self.client.refund(&payment_id.to_string())?;
        Ok(())
    }
}

// Alternative adapter for testing
#[derive(HexAdapter)]
struct MockPaymentGateway {
    payments: Arc<Mutex<HashMap<PaymentId, Money>>>,
}

impl PaymentGateway for MockPaymentGateway {
    fn process_payment(&self, _order_id: OrderId, amount: Money) -> HexResult<PaymentId> {
        let payment_id = PaymentId::new();
        self.payments.lock().unwrap().insert(payment_id, amount);
        Ok(payment_id)
    }

    fn refund_payment(&self, payment_id: PaymentId) -> HexResult<()> {
        self.payments.lock().unwrap().remove(&payment_id);
        Ok(())
    }
}
```

### Adapter Swapping

The power of hexagonal architecture: swap adapters without changing domain:

```rust
// Production configuration
fn build_production_app() -> Application {
    let payment_gateway: Arc<dyn PaymentGateway> = Arc::new(StripePaymentGateway::new());
    let notification_service: Arc<dyn NotificationService> = Arc::new(SendGridAdapter::new());

    Application::new(payment_gateway, notification_service)
}

// Test configuration
fn build_test_app() -> Application {
    let payment_gateway: Arc<dyn PaymentGateway> = Arc::new(MockPaymentGateway::new());
    let notification_service: Arc<dyn NotificationService> = Arc::new(InMemoryNotificationService::new());

    Application::new(payment_gateway, notification_service)
}
```

---

## Dependency Injection Container

Hexser provides a **zero-boilerplate dependency injection container** with:
- Thread-safe service resolution
- Lifetime scoping (singleton, transient)
- Compile-time circular dependency detection
- Async provider support

### Container Components

#### `Container`
Core DI resolver that manages service instances

#### `Provider`
Trait for synchronous service factory functions

```rust
pub trait Provider<T> {
    fn provide(&self) -> HexResult<T>;
}
```

#### `AsyncProvider`
Trait for asynchronous service instantiation

```rust
pub trait AsyncProvider<T> {
    async fn provide(&self) -> HexResult<T>;
}
```

#### `Scope`
Enumeration controlling service lifetime

```rust
pub enum Scope {
    Singleton,  // Single instance shared across application
    Transient,  // New instance per request
    Scoped,     // Single instance per scope (e.g., per request)
}
```

### Usage Example

```rust
use hexser::prelude::*;

// Define services
struct DatabaseConfig {
    connection_string: String,
}

struct UserRepository {
    db: Arc<DatabaseConfig>,
}

struct UserService {
    repo: Arc<UserRepository>,
}

// Create container
let mut container = Container::new();

// Register singleton database config
container.register_singleton(|| {
    DatabaseConfig {
        connection_string: env::var("DATABASE_URL").unwrap(),
    }
})?;

// Register transient repository
container.register_transient(|c: &Container| {
    let db = c.resolve::<DatabaseConfig>()?;
    Ok(UserRepository { db: Arc::new(db) })
})?;

// Register transient service
container.register_transient(|c: &Container| {
    let repo = c.resolve::<UserRepository>()?;
    Ok(UserService { repo: Arc::new(repo) })
})?;

// Resolve service
let user_service = container.resolve::<UserService>()?;
```

### Static Container

For compile-time dependency registration:

```rust
use hexser::prelude::*;

hex_static! {
    DatabaseConfig => Scope::Singleton,
    UserRepository => Scope::Transient,
    UserService => Scope::Transient,
}

fn main() {
    let container = StaticContainer::build();
    let user_service = container.resolve::<UserService>().unwrap();
}
```

---

## Graph-Based Introspection

Hexser provides a **graph-based introspection system** for analyzing hexagonal architecture:

### Core Components

#### `HexGraph`
Central immutable graph structure representing complete architecture

```rust
pub struct HexGraph {
    nodes: Vec<HexNode>,
    edges: Vec<HexEdge>,
    metadata: GraphMetadata,
}
```

#### `HexNode`
Represents individual components

```rust
pub struct HexNode {
    id: NodeId,
    layer: Layer,
    role: Role,
    type_name: String,
    metadata: NodeMetadata,
}
```

#### `HexEdge`
Defines relationships between components

```rust
pub struct HexEdge {
    from: NodeId,
    to: NodeId,
    relationship: Relationship,
}
```

#### `Layer` Enum
Categorizes architectural layers

```rust
pub enum Layer {
    Domain,
    Ports,
    Adapters,
    Application,
    Infrastructure,
}
```

#### `Role` Enum
Defines component roles

```rust
pub enum Role {
    Entity,
    ValueObject,
    Aggregate,
    Repository,
    Adapter,
    DirectiveHandler,
    QueryHandler,
    DomainService,
    // ... more roles
}
```

#### `Relationship` Enum
Specifies connection types

```rust
pub enum Relationship {
    DependsOn,
    Implements,
    Uses,
    Publishes,
    Subscribes,
    Contains,
}
```

### Building Graphs

```rust
use hexser::prelude::*;

let mut builder = GraphBuilder::new();

// Add nodes
let user_entity = builder.add_node(
    Layer::Domain,
    Role::Entity,
    "User",
)?;

let user_repo_port = builder.add_node(
    Layer::Ports,
    Role::Repository,
    "UserRepository",
)?;

let postgres_adapter = builder.add_node(
    Layer::Adapters,
    Role::Adapter,
    "PostgresUserRepository",
)?;

// Add edges
builder.add_edge(
    postgres_adapter,
    user_repo_port,
    Relationship::Implements,
)?;

builder.add_edge(
    user_repo_port,
    user_entity,
    Relationship::Uses,
)?;

// Build immutable graph
let graph = builder.build();
```

### Querying Graphs

```rust
// Find all adapters
let adapters: Vec<&HexNode> = graph
    .nodes()
    .filter(|n| n.layer == Layer::Adapters)
    .collect();

// Find dependencies of a node
let dependencies: Vec<NodeId> = graph
    .outgoing_edges(node_id)
    .map(|e| e.to)
    .collect();

// Analyze architectural violations
let violations = graph.analyze_violations();
for violation in violations {
    println!("Violation: {:?}", violation);
}
```

### Visualization

```rust
// Generate DOT format for Graphviz
let dot = graph.to_dot();
std::fs::write("architecture.dot", dot)?;

// Generate SVG
std::process::Command::new("dot")
    .args(&["-Tsvg", "architecture.dot", "-o", "architecture.svg"])
    .output()?;
```

### AI Context Integration

Hexser supports **machine-readable architecture** for AI tools:

```rust
use hexser::prelude::*;

let ai_context = graph.to_ai_context()?;
let json = serde_json::to_string_pretty(&ai_context)?;

// Feed to AI for architectural analysis
let analysis = ai_tool.analyze_architecture(&json)?;
```

---

## Registration Macros

Hexser provides **compile-time component registration** via macros:

### `hex_register_component!`
Core registration macro

**Syntax:**
```rust
hex_register_component!(Type, Layer::LayerName, Role::RoleName);
```

**Example:**
```rust
hex_register_component!(User, Layer::Domain, Role::Entity);
hex_register_component!(UserRepository, Layer::Ports, Role::Repository);
hex_register_component!(PostgresUserRepository, Layer::Adapters, Role::Adapter);
```

### Layer-Specific Convenience Macros

#### `hex_register_domain!`
```rust
hex_register_domain!(User, Role::Entity);
hex_register_domain!(Email, Role::ValueObject);
hex_register_domain!(Order, Role::Aggregate);
```

#### `hex_register_port!`
```rust
hex_register_port!(UserRepository, Role::Repository);
hex_register_port!(PaymentGateway, Role::OutputPort);
```

#### `hex_register_adapter!`
```rust
hex_register_adapter!(PostgresUserRepository);
hex_register_adapter!(StripePaymentGateway);
```

#### `hex_register_application!`
```rust
hex_register_application!(CreateUserDirectiveHandler, Role::DirectiveHandler);
hex_register_application!(GetUserQueryHandler, Role::QueryHandler);
```

#### `hex_register_infrastructure!`
```rust
hex_register_infrastructure!(DatabaseConfig);
hex_register_infrastructure!(LoggingConfig);
```

### `hex_static!`
Static dependency container construction

**Syntax:**
```rust
hex_static! {
    Type1 => Scope::Singleton,
    Type2 => Scope::Transient,
    Type3 => Scope::Scoped,
}
```

**Example:**
```rust
hex_static! {
    DatabaseConfig => Scope::Singleton,
    UserRepository => Scope::Transient,
    OrderRepository => Scope::Transient,
    PaymentGateway => Scope::Singleton,
}
```

---

## Features & Dependencies

### Cargo.toml Configuration

```toml
[dependencies]
hexser = { version = "0.4.7", features = ["full"] }
```

### Available Features

#### `full`
Enables all optional features (recommended)

```toml
hexser = { version = "0.4.7", features = ["full"] }
```

#### `async`
Enables async/await support

```toml
hexser = { version = "0.4.7", features = ["async"] }
```

**Provides:**
- `AsyncProvider` trait
- Async DirectiveHandler support
- Async QueryHandler support
- Tokio runtime integration

#### `serde`
Enables serialization/deserialization

```toml
hexser = { version = "0.4.7", features = ["serde"] }
```

**Provides:**
- Serde derives for types
- JSON serialization
- Graph export to JSON

#### `chrono`
Enables date/time handling

```toml
hexser = { version = "0.4.7", features = ["chrono"] }
```

**Provides:**
- Timestamp support for events
- Date/time value objects

### Core Dependencies

**Required:**
- `inventory` ^0.3 - Compile-time component registration

**Optional (enabled by features):**
- `async-trait` - Async trait support
- `tokio` - Async runtime
- `serde` - Serialization
- `serde_json` - JSON support
- `chrono` - Date/time handling

---

## Implementation Examples

### Complete E-Commerce Order System

```rust
use hexser::prelude::*;
use std::sync::Arc;

// ============================================================================
// DOMAIN LAYER
// ============================================================================

// Value Objects
#[derive(HexValueItem, Clone, PartialEq, Eq)]
struct CustomerId(Uuid);

#[derive(HexValueItem, Clone, PartialEq)]
struct Money {
    amount: Decimal,
    currency: Currency,
}

#[derive(HexValueItem, Clone, PartialEq, Eq)]
struct ProductId(Uuid);

// Entities
#[derive(HexEntity)]
struct OrderItem {
    id: OrderItemId,
    product_id: ProductId,
    quantity: u32,
    unit_price: Money,
}

impl OrderItem {
    fn total(&self) -> Money {
        Money {
            amount: self.unit_price.amount * Decimal::from(self.quantity),
            currency: self.unit_price.currency,
        }
    }
}

// Aggregates
#[derive(HexAggregate)]
struct Order {
    id: OrderId,
    customer_id: CustomerId,
    items: Vec<OrderItem>,
    status: OrderStatus,
    created_at: DateTime<Utc>,
}

impl Order {
    fn new(customer_id: CustomerId, items: Vec<OrderItem>) -> HexResult<Self> {
        if items.is_empty() {
            return Err(Hexserror::Validation("Order must have items".into()));
        }

        Ok(Self {
            id: OrderId::new(),
            customer_id,
            items,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
        })
    }

    fn total(&self) -> Money {
        self.items.iter().fold(
            Money { amount: Decimal::ZERO, currency: Currency::USD },
            |acc, item| Money {
                amount: acc.amount + item.total().amount,
                currency: acc.currency,
            }
        )
    }

    fn confirm(&mut self) -> HexResult<OrderConfirmed> {
        if self.status != OrderStatus::Pending {
            return Err(Hexserror::InvalidOperation("Order already processed".into()));
        }

        self.status = OrderStatus::Confirmed;
        Ok(OrderConfirmed {
            order_id: self.id,
            customer_id: self.customer_id,
            total: self.total(),
            occurred_at: Utc::now(),
        })
    }
}

// Domain Events
#[derive(HexDomain)]
struct OrderConfirmed {
    order_id: OrderId,
    customer_id: CustomerId,
    total: Money,
    occurred_at: DateTime<Utc>,
}

// ============================================================================
// PORTS LAYER
// ============================================================================

// Repository Port
#[derive(HexPort)]
trait OrderRepository: Repository<Order> {
    fn find_by_customer(&self, customer_id: &CustomerId) -> HexResult<Vec<Order>>;
    fn find_pending_orders(&self) -> HexResult<Vec<Order>>;
}

// Output Ports
#[derive(HexPort)]
trait PaymentGateway {
    fn charge(&self, customer_id: CustomerId, amount: Money) -> HexResult<PaymentId>;
}

#[derive(HexPort)]
trait NotificationService {
    fn send_order_confirmation(&self, customer_id: CustomerId, order_id: OrderId) -> HexResult<()>;
}

// ============================================================================
// ADAPTERS LAYER
// ============================================================================

// PostgreSQL Adapter
#[derive(HexAdapter)]
struct PostgresOrderRepository {
    pool: PgPool,
}

impl Repository<Order> for PostgresOrderRepository {
    fn save(&self, order: &Order) -> HexResult<()> {
        // Implementation
        Ok(())
    }

    fn find_by_id(&self, id: &OrderId) -> HexResult<Option<Order>> {
        // Implementation
        Ok(None)
    }

    fn delete(&self, id: &OrderId) -> HexResult<()> {
        // Implementation
        Ok(())
    }
}

impl OrderRepository for PostgresOrderRepository {
    fn find_by_customer(&self, customer_id: &CustomerId) -> HexResult<Vec<Order>> {
        // Implementation
        Ok(vec![])
    }

    fn find_pending_orders(&self) -> HexResult<Vec<Order>> {
        // Implementation
        Ok(vec![])
    }
}

// Stripe Adapter
#[derive(HexAdapter)]
struct StripePaymentGateway {
    api_key: String,
}

impl PaymentGateway for StripePaymentGateway {
    fn charge(&self, customer_id: CustomerId, amount: Money) -> HexResult<PaymentId> {
        // Stripe API call
        Ok(PaymentId::new())
    }
}

// Email Adapter
#[derive(HexAdapter)]
struct SendGridNotificationService {
    api_key: String,
}

impl NotificationService for SendGridNotificationService {
    fn send_order_confirmation(&self, customer_id: CustomerId, order_id: OrderId) -> HexResult<()> {
        // SendGrid API call
        Ok(())
    }
}

// ============================================================================
// APPLICATION LAYER
// ============================================================================

// Directives (Commands)
#[derive(HexDirective)]
struct PlaceOrderDirective {
    customer_id: Uuid,
    items: Vec<OrderItemData>,
}

#[derive(HexDirective)]
struct ConfirmOrderDirective {
    order_id: Uuid,
}

// Queries
#[derive(HexQuery)]
struct GetOrderDetailsQuery {
    order_id: Uuid,
}

#[derive(HexQuery)]
struct GetCustomerOrdersQuery {
    customer_id: Uuid,
}

// Directive Handlers
struct PlaceOrderDirectiveHandler {
    order_repo: Arc<dyn OrderRepository>,
    payment_gateway: Arc<dyn PaymentGateway>,
    notification_service: Arc<dyn NotificationService>,
}

impl DirectiveHandler<PlaceOrderDirective> for PlaceOrderDirectiveHandler {
    type Output = OrderId;

    fn handle(&self, directive: PlaceOrderDirective) -> HexResult<Self::Output> {
        // 1. Create order items
        let items: Vec<OrderItem> = directive.items
            .into_iter()
            .map(|data| OrderItem {
                id: OrderItemId::new(),
                product_id: ProductId(data.product_id),
                quantity: data.quantity,
                unit_price: Money {
                    amount: data.unit_price,
                    currency: Currency::USD,
                },
            })
            .collect();

        // 2. Create order aggregate
        let customer_id = CustomerId(directive.customer_id);
        let mut order = Order::new(customer_id, items)?;

        // 3. Process payment
        let payment_id = self.payment_gateway.charge(
            customer_id,
            order.total(),
        )?;

        // 4. Confirm order and get event
        let event = order.confirm()?;

        // 5. Persist order
        self.order_repo.save(&order)?;

        // 6. Send notification
        self.notification_service.send_order_confirmation(
            customer_id,
            order.id,
        )?;

        Ok(order.id)
    }
}

// Query Handlers
struct GetOrderDetailsQueryHandler {
    order_repo: Arc<dyn OrderRepository>,
}

impl QueryHandler<GetOrderDetailsQuery> for GetOrderDetailsQueryHandler {
    type Output = OrderDetailsDto;

    fn handle(&self, query: GetOrderDetailsQuery) -> HexResult<Self::Output> {
        let order = self.order_repo
            .find_by_id(&OrderId(query.order_id))?
            .ok_or_else(|| Hexserror::NotFound("Order not found".into()))?;

        Ok(OrderDetailsDto {
            id: order.id.0,
            customer_id: order.customer_id.0,
            items: order.items.iter().map(|i| OrderItemDto {
                product_id: i.product_id.0,
                quantity: i.quantity,
                unit_price: i.unit_price.amount,
            }).collect(),
            status: order.status,
            total: order.total().amount,
            created_at: order.created_at,
        })
    }
}

// ============================================================================
// REGISTRATION
// ============================================================================

hex_register_domain!(Order, Role::Aggregate);
hex_register_domain!(OrderItem, Role::Entity);
hex_register_domain!(Money, Role::ValueObject);

hex_register_port!(OrderRepository, Role::Repository);
hex_register_port!(PaymentGateway, Role::OutputPort);
hex_register_port!(NotificationService, Role::OutputPort);

hex_register_adapter!(PostgresOrderRepository);
hex_register_adapter!(StripePaymentGateway);
hex_register_adapter!(SendGridNotificationService);

hex_register_application!(PlaceOrderDirectiveHandler, Role::DirectiveHandler);
hex_register_application!(GetOrderDetailsQueryHandler, Role::QueryHandler);
```

---

## Best Practices

### 1. Domain Independence

**DO:**
```rust
// Domain has no external dependencies
#[derive(HexAggregate)]
struct Order {
    id: OrderId,
    total: Money,  // Value object, not Decimal
}
```

**DON'T:**
```rust
// Domain depending on external crate
#[derive(HexAggregate)]
struct Order {
    id: Uuid,  // Direct dependency on uuid crate
    total: rust_decimal::Decimal,  // Direct dependency
}
```

### 2. Aggregate Boundaries

**DO:**
```rust
// Reference other aggregates by ID only
#[derive(HexAggregate)]
struct Order {
    id: OrderId,
    customer_id: CustomerId,  // ID reference
    items: Vec<OrderItem>,    // Owned entities
}
```

**DON'T:**
```rust
// Don't hold references to other aggregates
struct Order {
    id: OrderId,
    customer: Customer,  // Wrong! Crosses aggregate boundary
}
```

### 3. Port Abstraction

**DO:**
```rust
// Abstract, technology-agnostic interface
trait OrderRepository: Repository<Order> {
    fn find_by_customer(&self, customer_id: &CustomerId) -> HexResult<Vec<Order>>;
}
```

**DON'T:**
```rust
// Don't leak implementation details
trait OrderRepository {
    fn execute_sql(&self, query: &str) -> SqlResult<Vec<Row>>;  // Leaks SQL
}
```

### 4. Directive Naming

**DO:**
```rust
// Imperative, intention-revealing names
#[derive(HexDirective)]
struct PlaceOrderDirective { }

#[derive(HexDirective)]
struct CancelOrderDirective { }
```

**DON'T:**
```rust
// Generic or vague names
struct OrderDirective { }
struct UpdateDirective { }
```

### 5. Query Optimization

**DO:**
```rust
// Queries can bypass domain model for performance
impl QueryHandler<GetOrderListQuery> for GetOrderListQueryHandler {
    fn handle(&self, query: GetOrderListQuery) -> HexResult<Vec<OrderSummaryDto>> {
        // Direct database query with optimized SQL
        self.read_db.query_order_summaries(query.filters)
    }
}
```

**DON'T:**
```rust
// Don't load full aggregates for read-only queries
impl QueryHandler<GetOrderListQuery> for GetOrderListQueryHandler {
    fn handle(&self, query: GetOrderListQuery) -> HexResult<Vec<OrderSummaryDto>> {
        // Inefficient: loads full aggregates
        let orders = self.order_repo.find_all()?;
        Ok(orders.into_iter().map(|o| o.to_summary()).collect())
    }
}
```

### 6. Error Handling

**DO:**
```rust
// Use HexResult for consistent error handling
fn place_order(&self, directive: PlaceOrderDirective) -> HexResult<OrderId> {
    let customer = self.customer_repo
        .find_by_id(&directive.customer_id)?
        .ok_or_else(|| Hexserror::NotFound("Customer not found".into()))?;

    // ... rest of logic
}
```

**DON'T:**
```rust
// Don't use raw Result or unwrap
fn place_order(&self, directive: PlaceOrderDirective) -> Result<OrderId, String> {
    let customer = self.customer_repo.find_by_id(&directive.customer_id).unwrap();
    // ... rest of logic
}
```

### 7. Dependency Injection

**DO:**
```rust
// Inject dependencies through constructors
struct PlaceOrderDirectiveHandler {
    order_repo: Arc<dyn OrderRepository>,
    payment_gateway: Arc<dyn PaymentGateway>,
}

impl PlaceOrderDirectiveHandler {
    fn new(
        order_repo: Arc<dyn OrderRepository>,
        payment_gateway: Arc<dyn PaymentGateway>,
    ) -> Self {
        Self { order_repo, payment_gateway }
    }
}
```

**DON'T:**
```rust
// Don't create dependencies internally
struct PlaceOrderDirectiveHandler;

impl PlaceOrderDirectiveHandler {
    fn handle(&self, directive: PlaceOrderDirective) -> HexResult<OrderId> {
        let order_repo = PostgresOrderRepository::new();  // Wrong!
        // ...
    }
}
```

### 8. Testing Strategy

**DO:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_place_order_success() {
        // Use mock adapters
        let order_repo = Arc::new(InMemoryOrderRepository::new());
        let payment_gateway = Arc::new(MockPaymentGateway::new());

        let handler = PlaceOrderDirectiveHandler::new(order_repo, payment_gateway);

        let directive = PlaceOrderDirective { /* ... */ };
        let result = handler.handle(directive);

        assert!(result.is_ok());
    }
}
```

**DON'T:**
```rust
// Don't test with real external services
#[test]
fn test_place_order() {
    let handler = PlaceOrderDirectiveHandler::new(
        Arc::new(PostgresOrderRepository::new()),  // Real DB!
        Arc::new(StripePaymentGateway::new()),    // Real API!
    );
    // ...
}
```

---

## Common Pitfalls

### 1. Leaking Domain Logic into Adapters

**Problem:**
```rust
impl OrderRepository for PostgresOrderRepository {
    fn save(&self, order: &Order) -> HexResult<()> {
        // Business logic in adapter - WRONG!
        if order.total().amount > Decimal::from(1000) {
            order.apply_discount(0.1);
        }
        // ... persist
    }
}
```

**Solution:**
```rust
// Business logic belongs in domain
impl Order {
    fn apply_bulk_discount(&mut self) {
        if self.total().amount > Decimal::from(1000) {
            self.discount = Some(Discount::percentage(10));
        }
    }
}

// Adapter only persists
impl OrderRepository for PostgresOrderRepository {
    fn save(&self, order: &Order) -> HexResult<()> {
        // Just persist, no business logic
        sqlx::query!(/* ... */).execute(&self.pool)?;
        Ok(())
    }
}
```

### 2. Fat Directives

**Problem:**
```rust
#[derive(HexDirective)]
struct CreateUserDirective {
    email: String,
    password: String,
    name: String,
    address: Address,
    preferences: UserPreferences,
    payment_methods: Vec<PaymentMethod>,
    // ... 20 more fields
}
```

**Solution:**
```rust
// Break into smaller, focused directives
#[derive(HexDirective)]
struct RegisterUserDirective {
    email: String,
    password: String,
    name: String,
}

#[derive(HexDirective)]
struct AddUserAddressDirective {
    user_id: UserId,
    address: Address,
}

#[derive(HexDirective)]
struct SetUserPreferencesDirective {
    user_id: UserId,
    preferences: UserPreferences,
}
```

### 3. Anemic Domain Model

**Problem:**
```rust
// Domain object with no behavior - just data
#[derive(HexAggregate)]
struct Order {
    id: OrderId,
    status: OrderStatus,
    items: Vec<OrderItem>,
}

// All logic in application layer
impl DirectiveHandler<ConfirmOrderDirective> for ConfirmOrderDirectiveHandler {
    fn handle(&self, directive: ConfirmOrderDirective) -> HexResult<()> {
        let mut order = self.order_repo.find_by_id(&directive.order_id)?;

        // Business logic in handler - WRONG!
        if order.items.is_empty() {
            return Err(Hexserror::Validation("No items".into()));
        }
        if order.status != OrderStatus::Pending {
            return Err(Hexserror::InvalidOperation("Already processed".into()));
        }
        order.status = OrderStatus::Confirmed;

        self.order_repo.save(&order)?;
        Ok(())
    }
}
```

**Solution:**
```rust
// Rich domain model with behavior
#[derive(HexAggregate)]
struct Order {
    id: OrderId,
    status: OrderStatus,
    items: Vec<OrderItem>,
}

impl Order {
    fn confirm(&mut self) -> HexResult<OrderConfirmed> {
        // Business rules in domain
        if self.items.is_empty() {
            return Err(Hexserror::Validation("No items".into()));
        }
        if self.status != OrderStatus::Pending {
            return Err(Hexserror::InvalidOperation("Already processed".into()));
        }

        self.status = OrderStatus::Confirmed;
        Ok(OrderConfirmed { order_id: self.id })
    }
}

// Handler delegates to domain
impl DirectiveHandler<ConfirmOrderDirective> for ConfirmOrderDirectiveHandler {
    fn handle(&self, directive: ConfirmOrderDirective) -> HexResult<()> {
        let mut order = self.order_repo.find_by_id(&directive.order_id)?;
        let event = order.confirm()?;  // Domain does the work
        self.order_repo.save(&order)?;
        Ok(())
    }
}
```

### 4. Query Side Effects

**Problem:**
```rust
impl QueryHandler<GetUserProfileQuery> for GetUserProfileQueryHandler {
    fn handle(&self, query: GetUserProfileQuery) -> HexResult<UserProfileDto> {
        let user = self.user_repo.find_by_id(&query.user_id)?;

        // Side effect in query - WRONG!
        user.last_accessed = Utc::now();
        self.user_repo.save(&user)?;

        Ok(UserProfileDto::from(user))
    }
}
```

**Solution:**
```rust
// Queries must be side-effect free
impl QueryHandler<GetUserProfileQuery> for GetUserProfileQueryHandler {
    fn handle(&self, query: GetUserProfileQuery) -> HexResult<UserProfileDto> {
        let user = self.user_repo.find_by_id(&query.user_id)?;
        Ok(UserProfileDto::from(user))
    }
}

// Use separate directive for side effects
#[derive(HexDirective)]
struct RecordUserAccessDirective {
    user_id: UserId,
}
```

### 5. Forgetting to Register Components

**Problem:**
```rust
// Component defined but not registered
#[derive(HexAdapter)]
struct PostgresOrderRepository { }

// Graph introspection won't find it!
```

**Solution:**
```rust
#[derive(HexAdapter)]
struct PostgresOrderRepository { }

// Always register components
hex_register_adapter!(PostgresOrderRepository);
```

### 6. Tight Coupling via Concrete Types

**Problem:**
```rust
struct PlaceOrderDirectiveHandler {
    order_repo: PostgresOrderRepository,  // Concrete type - WRONG!
}
```

**Solution:**
```rust
struct PlaceOrderDirectiveHandler {
    order_repo: Arc<dyn OrderRepository>,  // Interface - CORRECT!
}
```

### 7. Missing Error Context

**Problem:**
```rust
fn handle(&self, directive: PlaceOrderDirective) -> HexResult<OrderId> {
    let customer = self.customer_repo.find_by_id(&directive.customer_id)?;
    // Generic error, hard to debug
}
```

**Solution:**
```rust
fn handle(&self, directive: PlaceOrderDirective) -> HexResult<OrderId> {
    let customer = self.customer_repo
        .find_by_id(&directive.customer_id)?
        .ok_or_else(|| Hexserror::NotFound(
            format!("Customer {} not found", directive.customer_id)
        ))?;
    // Clear, actionable error
}
```

---

## Summary

Hexser provides a comprehensive framework for implementing hexagonal architecture in Rust with:

- **Zero-boilerplate** derive macros
- **CQRS** with Directive/Query patterns
- **Clean separation** of concerns across five layers
- **Dependency injection** with lifetime management
- **Graph-based introspection** for architectural analysis
- **Compile-time registration** via inventory
- **CloudEvents** support for event-driven architecture

**Key Principles:**
1. Keep domain logic in domain layer
2. Use ports to abstract external dependencies
3. Implement adapters for specific technologies
4. Separate commands (directives) from queries (CQRS)
5. Reference other aggregates by ID only
6. Inject dependencies, don't create them
7. Register all components for graph introspection

**Next Steps:**
1. Add `hexser = { version = "0.4.7", features = ["full"] }` to Cargo.toml
2. Import prelude: `use hexser::prelude::*;`
3. Define domain entities with derive macros
4. Create port interfaces
5. Implement adapters
6. Build directive/query handlers
7. Register components
8. Wire up with DI container

---

## References

- **Documentation:** https://docs.rs/hexser/0.4.7/hexser/
- **Crate:** https://crates.io/crates/hexser
- **License:** MIT OR Apache-2.0
- **Version:** 0.4.7

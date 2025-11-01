// src/cqrs/bus.rs
//! Command and Query Bus Implementation
//!
//! The bus routes commands and queries to their respective handlers.
//! It provides middleware support for cross-cutting concerns.

use crate::cqrs::types::{
    Command, CommandHandler, CommandMiddleware, Query, QueryHandler, QueryMiddleware, Result,
};
use async_trait::async_trait;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/
/
/
/
pub struct CommandBus {
    handlers: Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
    middleware: Arc<Vec<Box<dyn CommandMiddleware>>>,
}

impl CommandBus {
    
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            middleware: Arc::new(Vec::new()),
        }
    }

    
    pub fn with_middleware(middleware: Vec<Box<dyn CommandMiddleware>>) -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            middleware: Arc::new(middleware),
        }
    }

    
    pub async fn register<C: Command + 'static>(&self, handler: Box<dyn CommandHandler<C>>) {
        let type_id = TypeId::of::<C>();
        let mut handlers = self.handlers.write().await;
        handlers.insert(type_id, Box::new(handler));
    }

    
    
    
    
    
    
    
    
    pub async fn execute<C: Command + 'static>(&self, command: C) -> Result<C::Result>
    where
        C::Result: 'static,
    {
        let command_name = command.name();

        
        for mw in self.middleware.iter() {
            mw.before_execute(command_name).await?;
        }

        
        let result = async {
            let type_id = TypeId::of::<C>();
            let handlers = self.handlers.read().await;

            let handler = handlers.get(&type_id).ok_or_else(|| {
                anyhow::anyhow!("No handler registered for command: {}", command_name)
            })?;

            let handler = handler
                .downcast_ref::<Box<dyn CommandHandler<C>>>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Handler type mismatch for command: {}", command_name)
                })?;

            handler.handle(command).await
        }
        .await;

        
        match &result {
            Ok(_) => {
                for mw in self.middleware.iter() {
                    mw.after_execute(command_name).await?;
                }
            }
            Err(e) => {
                for mw in self.middleware.iter() {
                    mw.on_error(command_name, e).await?;
                }
            }
        }

        result
    }
}

impl Default for CommandBus {
    fn default() -> Self {
        Self::new()
    }
}

/
/
/
/
pub struct QueryBus {
    handlers: Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
    middleware: Arc<Vec<Box<dyn QueryMiddleware>>>,
}

impl QueryBus {
    
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            middleware: Arc::new(Vec::new()),
        }
    }

    
    pub fn with_middleware(middleware: Vec<Box<dyn QueryMiddleware>>) -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            middleware: Arc::new(middleware),
        }
    }

    
    pub async fn register<Q: Query + 'static>(&self, handler: Box<dyn QueryHandler<Q>>) {
        let type_id = TypeId::of::<Q>();
        let mut handlers = self.handlers.write().await;
        handlers.insert(type_id, Box::new(handler));
    }

    
    
    
    
    
    
    
    
    pub async fn execute<Q: Query + 'static>(&self, query: Q) -> Result<Q::Result>
    where
        Q::Result: 'static,
    {
        let query_name = query.name();

        
        for mw in self.middleware.iter() {
            mw.before_execute(query_name).await?;
        }

        
        let result = async {
            let type_id = TypeId::of::<Q>();
            let handlers = self.handlers.read().await;

            let handler = handlers.get(&type_id).ok_or_else(|| {
                anyhow::anyhow!("No handler registered for query: {}", query_name)
            })?;

            let handler = handler
                .downcast_ref::<Box<dyn QueryHandler<Q>>>()
                .ok_or_else(|| {
                    anyhow::anyhow!("Handler type mismatch for query: {}", query_name)
                })?;

            handler.handle(query).await
        }
        .await;

        
        match &result {
            Ok(_) => {
                for mw in self.middleware.iter() {
                    mw.after_execute(query_name).await?;
                }
            }
            Err(e) => {
                for mw in self.middleware.iter() {
                    mw.on_error(query_name, e).await?;
                }
            }
        }

        result
    }
}

impl Default for QueryBus {
    fn default() -> Self {
        Self::new()
    }
}

/
pub struct MetricsMiddleware {
    command_counts: Arc<RwLock<HashMap<String, u64>>>,
    query_counts: Arc<RwLock<HashMap<String, u64>>>,
}

impl MetricsMiddleware {
    pub fn new() -> Self {
        Self {
            command_counts: Arc::new(RwLock::new(HashMap::new())),
            query_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn get_command_count(&self, command_name: &str) -> u64 {
        let counts = self.command_counts.read().await;
        *counts.get(command_name).unwrap_or(&0)
    }

    pub async fn get_query_count(&self, query_name: &str) -> u64 {
        let counts = self.query_counts.read().await;
        *counts.get(query_name).unwrap_or(&0)
    }
}

impl Default for MetricsMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CommandMiddleware for MetricsMiddleware {
    async fn after_execute(&self, command_name: &str) -> Result<()> {
        let mut counts = self.command_counts.write().await;
        *counts.entry(command_name.to_string()).or_insert(0) += 1;
        Ok(())
    }
}

#[async_trait]
impl QueryMiddleware for MetricsMiddleware {
    async fn after_execute(&self, query_name: &str) -> Result<()> {
        let mut counts = self.query_counts.write().await;
        *counts.entry(query_name.to_string()).or_insert(0) += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cqrs::types::Command;

    #[derive(Debug, Clone)]
    struct TestCommand {
        value: String,
    }

    impl Command for TestCommand {
        type Result = String;

        fn name(&self) -> &'static str {
            "TestCommand"
        }

        fn validate(&self) -> Result<()> {
            if self.value.is_empty() {
                return Err(anyhow::anyhow!("Value cannot be empty"));
            }
            Ok(())
        }
    }

    struct TestCommandHandler;

    #[async_trait]
    impl CommandHandler<TestCommand> for TestCommandHandler {
        async fn handle(&self, command: TestCommand) -> Result<String> {
            command.validate()?;
            Ok(format!("Handled: {}", command.value))
        }
    }

    #[tokio::test]
    async fn test_command_bus_execute() {
        let bus = CommandBus::new();
        bus.register(Box::new(TestCommandHandler)).await;

        let command = TestCommand {
            value: "test".to_string(),
        };
        let result = bus.execute(command).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Handled: test");
    }

    #[tokio::test]
    async fn test_command_bus_validation() {
        let bus = CommandBus::new();
        bus.register(Box::new(TestCommandHandler)).await;

        let command = TestCommand {
            value: "".to_string(),
        };
        let result = bus.execute(command).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_command_bus_no_handler() {
        let bus = CommandBus::new();

        let command = TestCommand {
            value: "test".to_string(),
        };
        let result = bus.execute(command).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_metrics_middleware() {
        let metrics = Arc::new(MetricsMiddleware::new());
        let bus = CommandBus::with_middleware(vec![Box::new(metrics.clone())]);
        bus.register(Box::new(TestCommandHandler)).await;

        let command = TestCommand {
            value: "test".to_string(),
        };
        bus.execute(command).await.unwrap();

        let count = metrics.get_command_count("TestCommand").await;
        assert_eq!(count, 1);
    }
}

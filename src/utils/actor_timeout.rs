//! Actor timeout utilities
//!
//! Provides helper functions for adding timeouts to actor message calls
//! to prevent hanging requests when actors don't respond.

use actix::dev::ToEnvelope;
use actix::prelude::*;
use log::error;
use std::time::Duration;

/// Default timeout duration for actor calls (5 seconds)
pub const DEFAULT_ACTOR_TIMEOUT: Duration = Duration::from_secs(5);

/// Extended timeout for potentially long-running operations (10 seconds)
pub const EXTENDED_ACTOR_TIMEOUT: Duration = Duration::from_secs(10);

/// Short timeout for quick operations (2 seconds)
pub const SHORT_ACTOR_TIMEOUT: Duration = Duration::from_secs(2);

/// Result type for actor timeout operations
pub type ActorTimeoutResult<T> = Result<T, ActorTimeoutError>;

/// Error types for actor timeout operations
#[derive(Debug)]
pub enum ActorTimeoutError {
    /// The actor call timed out
    Timeout {
        duration: Duration,
        actor_type: &'static str,
    },
    /// The actor returned an error
    ActorError(String),
    /// Failed to send message to actor
    MailboxError(String),
}

impl std::fmt::Display for ActorTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActorTimeoutError::Timeout {
                duration,
                actor_type,
            } => {
                write!(f, "{} actor timeout after {:?}", actor_type, duration)
            }
            ActorTimeoutError::ActorError(msg) => write!(f, "Actor error: {}", msg),
            ActorTimeoutError::MailboxError(msg) => write!(f, "Mailbox error: {}", msg),
        }
    }
}

impl std::error::Error for ActorTimeoutError {}

/// Send a message to an actor with a timeout
///
/// # Arguments
/// * `addr` - The actor address
/// * `msg` - The message to send
/// * `timeout` - Timeout duration
/// * `actor_type` - Name of the actor type for error reporting
///
/// # Returns
/// Result containing the actor's response or a timeout error
pub async fn send_with_timeout<A, M>(
    addr: &Addr<A>,
    msg: M,
    timeout: Duration,
    actor_type: &'static str,
) -> ActorTimeoutResult<M::Result>
where
    A: Actor + Handler<M>,
    A::Context: ToEnvelope<A, M>,
    M: Message + Send + 'static,
    M::Result: Send,
{
    match tokio::time::timeout(timeout, addr.send(msg)).await {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(e)) => {
            error!("Failed to send message to {} actor: {}", actor_type, e);
            Err(ActorTimeoutError::MailboxError(e.to_string()))
        }
        Err(_) => {
            error!("{} actor timeout after {:?}", actor_type, timeout);
            Err(ActorTimeoutError::Timeout {
                duration: timeout,
                actor_type,
            })
        }
    }
}

/// Send a message to an actor with default timeout
pub async fn send_with_default_timeout<A, M>(
    addr: &Addr<A>,
    msg: M,
    actor_type: &'static str,
) -> ActorTimeoutResult<M::Result>
where
    A: Actor + Handler<M>,
    A::Context: ToEnvelope<A, M>,
    M: Message + Send + 'static,
    M::Result: Send,
{
    send_with_timeout(addr, msg, DEFAULT_ACTOR_TIMEOUT, actor_type).await
}

/// Send a message to an actor with extended timeout for long operations
pub async fn send_with_extended_timeout<A, M>(
    addr: &Addr<A>,
    msg: M,
    actor_type: &'static str,
) -> ActorTimeoutResult<M::Result>
where
    A: Actor + Handler<M>,
    A::Context: ToEnvelope<A, M>,
    M: Message + Send + 'static,
    M::Result: Send,
{
    send_with_timeout(addr, msg, EXTENDED_ACTOR_TIMEOUT, actor_type).await
}

/// Send a message to an actor with short timeout for quick operations
pub async fn send_with_short_timeout<A, M>(
    addr: &Addr<A>,
    msg: M,
    actor_type: &'static str,
) -> ActorTimeoutResult<M::Result>
where
    A: Actor + Handler<M>,
    A::Context: ToEnvelope<A, M>,
    M: Message + Send + 'static,
    M::Result: Send,
{
    send_with_timeout(addr, msg, SHORT_ACTOR_TIMEOUT, actor_type).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix::prelude::*;

    #[derive(Message)]
    #[rtype(result = "Result<String, String>")]
    struct TestMessage;

    struct TestActor;

    impl Actor for TestActor {
        type Context = Context<Self>;
    }

    impl Handler<TestMessage> for TestActor {
        type Result = Result<String, String>;

        fn handle(&mut self, _msg: TestMessage, _ctx: &mut Self::Context) -> Self::Result {
            Ok("success".to_string())
        }
    }

    #[actix::test]
    async fn test_send_with_timeout_success() {
        let addr = TestActor.start();
        let result = send_with_timeout(&addr, TestMessage, Duration::from_secs(1), "Test").await;
        assert!(result.is_ok());
    }

    #[actix::test]
    async fn test_send_with_default_timeout() {
        let addr = TestActor.start();
        let result = send_with_default_timeout(&addr, TestMessage, "Test").await;
        assert!(result.is_ok());
    }
}

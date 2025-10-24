/// Helper utilities for CQRS handler execution
use std::sync::mpsc;
use std::thread;

/// Execute a CQRS handler in a separate OS thread to escape the Tokio runtime context.
/// This prevents "Cannot start a runtime from within a runtime" errors.
pub async fn execute_in_thread<F, R>(handler_fn: F) -> Result<R, String>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let result = handler_fn();
            let _ = tx.send(result);
        });
        rx.recv().map_err(|e| format!("Thread communication error: {}", e))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}
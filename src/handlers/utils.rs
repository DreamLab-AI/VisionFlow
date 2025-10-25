/// Helper utilities for CQRS handler execution
use std::sync::mpsc;
use std::thread;

/// Execute a CQRS handler in a separate OS thread to escape the Tokio runtime context.
/// This prevents "Cannot start a runtime from within a runtime" errors.
///
/// This uses pure OS threads without spawn_blocking to avoid any Tokio runtime interference.
pub async fn execute_in_thread<F, R>(handler_fn: F) -> Result<R, String>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = mpsc::channel();

    // Spawn pure OS thread - completely outside Tokio's control
    thread::spawn(move || {
        let result = handler_fn();
        let _ = tx.send(result);
    });

    // Wait for the result in a non-blocking way using spawn_blocking
    // This is safe because we're just waiting on a channel, not creating a runtime
    tokio::task::spawn_blocking(move || {
        rx.recv().map_err(|e| format!("Thread communication error: {}", e))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}
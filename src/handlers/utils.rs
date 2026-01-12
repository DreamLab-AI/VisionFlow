use std::sync::mpsc;
use std::thread;

pub async fn execute_in_thread<F, R>(handler_fn: F) -> Result<R, String>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = mpsc::channel();

    
    thread::spawn(move || {
        let result = handler_fn();
        let _ = tx.send(result);
    });

    
    
    tokio::task::spawn_blocking(move || {
        rx.recv()
            .map_err(|e| format!("Thread communication error: {}", e))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

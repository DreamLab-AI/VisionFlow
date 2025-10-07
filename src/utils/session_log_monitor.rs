use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader, AsyncSeekExt};
use tokio::time::{interval, Duration};
use log::{debug, warn, error};
use crate::utils::client_message_extractor::{ClientMessage, extract_client_messages};

/// Monitor session log files for client messages
pub struct SessionLogMonitor {
    session_dir: PathBuf,
    session_id: String,
    message_sender: mpsc::UnboundedSender<ClientMessage>,
    poll_interval: Duration,
}

impl SessionLogMonitor {
    pub fn new(
        session_dir: PathBuf,
        session_id: String,
        message_sender: mpsc::UnboundedSender<ClientMessage>,
    ) -> Self {
        Self {
            session_dir,
            session_id,
            message_sender,
            poll_interval: Duration::from_millis(500), // Poll every 500ms
        }
    }

    /// Start monitoring session logs
    pub async fn start(self: Arc<Self>) {
        let log_file = self.session_dir.join("session.log");

        debug!("Starting log monitor for session {}: {:?}", self.session_id, log_file);

        let mut ticker = interval(self.poll_interval);
        let mut last_position: u64 = 0;
        let mut buffer = String::new();

        loop {
            ticker.tick().await;

            // Try to open and read the log file
            match File::open(&log_file).await {
                Ok(mut file) => {
                    // Seek to last read position
                    if let Err(e) = file.seek(std::io::SeekFrom::Start(last_position)).await {
                        warn!("Failed to seek log file for session {}: {}", self.session_id, e);
                        continue;
                    }

                    let reader = BufReader::new(file);
                    let mut lines = reader.lines();

                    // Read new lines
                    while let Ok(Some(line)) = lines.next_line().await {
                        buffer.push_str(&line);
                        buffer.push('\n');

                        // Update position
                        last_position += line.len() as u64 + 1;
                    }

                    // Extract messages from buffer if we have content
                    if !buffer.is_empty() {
                        let messages = extract_client_messages(
                            &buffer,
                            Some(self.session_id.clone()),
                            None,
                        );

                        for msg in messages {
                            debug!("Extracted client message from session {}: {}",
                                   self.session_id, msg.content);

                            if let Err(e) = self.message_sender.send(msg) {
                                error!("Failed to send client message: {}", e);
                                return; // Channel closed, stop monitoring
                            }
                        }

                        // Clear buffer after processing
                        buffer.clear();
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Log file doesn't exist yet, continue waiting
                    debug!("Log file not found for session {}, waiting...", self.session_id);
                }
                Err(e) => {
                    warn!("Error opening log file for session {}: {}", self.session_id, e);
                }
            }
        }
    }

    /// Monitor multiple sessions concurrently
    pub async fn monitor_sessions(
        sessions: Vec<(String, PathBuf)>,
        message_sender: mpsc::UnboundedSender<ClientMessage>,
    ) {
        let mut handles = Vec::new();

        for (session_id, session_dir) in sessions {
            let monitor = Arc::new(Self::new(
                session_dir,
                session_id,
                message_sender.clone(),
            ));

            let handle = tokio::spawn(async move {
                monitor.start().await;
            });

            handles.push(handle);
        }

        // Wait for all monitors (they run indefinitely)
        for handle in handles {
            if let Err(e) = handle.await {
                error!("Session monitor task failed: {}", e);
            }
        }
    }
}

/// Monitor for the TCP server instance logs specifically
pub struct TCPServerLogMonitor {
    tcp_server_dir: PathBuf,
    message_sender: mpsc::UnboundedSender<ClientMessage>,
    poll_interval: Duration,
}

impl TCPServerLogMonitor {
    pub fn new(message_sender: mpsc::UnboundedSender<ClientMessage>) -> Self {
        Self {
            tcp_server_dir: PathBuf::from("/workspace/.swarm/tcp-server-instance"),
            message_sender,
            poll_interval: Duration::from_millis(300),
        }
    }

    pub async fn start(self: Arc<Self>) {
        debug!("Starting TCP server log monitor");

        let mut ticker = interval(self.poll_interval);
        let mut last_position: u64 = 0;
        let mut buffer = String::new();

        loop {
            ticker.tick().await;

            // Look for various log sources in TCP server directory
            let log_paths = vec![
                self.tcp_server_dir.join("mcp.log"),
                self.tcp_server_dir.join("claude-flow.log"),
                PathBuf::from("/app/mcp-logs/mcp-tcp-server.log"),
            ];

            for log_file in log_paths {
                if let Ok(mut file) = File::open(&log_file).await {
                    if let Err(e) = file.seek(std::io::SeekFrom::Start(last_position)).await {
                        warn!("Failed to seek TCP server log: {}", e);
                        continue;
                    }

                    let reader = BufReader::new(file);
                    let mut lines = reader.lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        buffer.push_str(&line);
                        buffer.push('\n');
                        last_position += line.len() as u64 + 1;
                    }

                    if !buffer.is_empty() {
                        let messages = extract_client_messages(&buffer, None, Some("tcp-server".to_string()));

                        for msg in messages {
                            debug!("Extracted client message from TCP server: {}", msg.content);
                            if let Err(e) = self.message_sender.send(msg) {
                                error!("Failed to send TCP server client message: {}", e);
                                return;
                            }
                        }

                        buffer.clear();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_log_monitor_extracts_messages() {
        let temp_dir = TempDir::new().unwrap();
        let session_dir = temp_dir.path().to_path_buf();
        let log_file = session_dir.join("session.log");

        // Write test log content
        fs::write(
            &log_file,
            "Some output\n**[CLIENT_MESSAGE]** Test message **[/CLIENT_MESSAGE]**\n",
        )
        .await
        .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let monitor = Arc::new(SessionLogMonitor::new(
            session_dir,
            "test-session".to_string(),
            tx,
        ));

        // Start monitor in background
        let handle = tokio::spawn(async move {
            monitor.start().await;
        });

        // Wait for message
        tokio::select! {
            Some(msg) = rx.recv() => {
                assert_eq!(msg.content, "Test message");
                assert_eq!(msg.session_id, Some("test-session".to_string()));
            }
            _ = tokio::time::sleep(Duration::from_secs(2)) => {
                panic!("Timeout waiting for message");
            }
        }

        handle.abort();
    }
}

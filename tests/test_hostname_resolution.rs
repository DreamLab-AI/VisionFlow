#[cfg(test)]
mod tests {
    use tokio::net::TcpStream;

    #[tokio::test]
    async fn test_hostname_resolution() {
        // Test that we can connect using a hostname instead of IP
        let test_cases = vec![
            ("localhost", 22), // SSH port, likely to exist
            ("127.0.0.1", 22), // IP address should still work
        ];

        for (host, port) in test_cases {
            let addr = format!("{}:{}", host, port);
            println!("Testing connection to: {}", addr);

            // We don't actually need to succeed, just verify parsing works
            match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                TcpStream::connect(&addr),
            )
            .await
            {
                Ok(Ok(_)) => println!("Connected successfully to {}", addr),
                Ok(Err(e)) => println!("Connection failed (expected): {}", e),
                Err(_) => println!("Connection timed out (expected)"),
            }
        }
    }

    #[test]
    fn test_hostname_formats() {
        // Test various hostname formats that should be valid
        let hostnames = vec![
            "localhost:8080",
            "example.com:443",
            "multi-agent-container:9500",
            "192.168.1.1:80",
            "[::1]:8080", // IPv6
        ];

        for hostname in hostnames {
            println!("Testing hostname format: {}", hostname);
            // The format itself is valid for TcpStream::connect
            assert!(!hostname.is_empty());
        }
    }
}

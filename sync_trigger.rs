// Simple sync trigger
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 GitHub Sync Trigger - multi-ontology branch");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Set environment
    std::env::set_var("FORCE_FULL_SYNC", "1");

    println!("GITHUB_BRANCH: {}", std::env::var("GITHUB_BRANCH")?);
    println!("FORCE_FULL_SYNC: {}", std::env::var("FORCE_FULL_SYNC")?);
    println!("NEO4J_URI: {}", std::env::var("NEO4J_URI")?);
    println!("");

    // Make HTTP call to sync endpoint
    let client = reqwest::Client::new();
    let pubkey = std::env::var("POWER_USER_PUBKEYS")?;

    println!("Calling http://localhost:4000/api/admin/sync...");

    let response = client
        .post("http://localhost:4000/api/admin/sync")
        .header("Content-Type", "application/json")
        .header("X-Nostr-Pubkey", pubkey)
        .timeout(std::time::Duration::from_secs(600))
        .send()
        .await?;

    let status = response.status();
    let body = response.text().await?;

    println!("Status: {}", status);
    println!("Response: {}", body);

    if status.is_success() {
        println!("✅ Sync triggered successfully!");
    } else {
        println!("❌ Sync failed with status: {}", status);
    }

    Ok(())
}

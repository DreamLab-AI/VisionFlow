use super::config::GitHubConfig;
use crate::config::AppFullSettings; 
use crate::errors::VisionFlowResult;
use log::{debug, info};
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

// const GITHUB_API_DELAY: Duration = Duration::from_millis(500); 
// const MAX_RETRIES: u32 = 3; 
// const RETRY_DELAY: Duration = Duration::from_secs(2); 

/
pub struct GitHubClient {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
    settings: Arc<RwLock<AppFullSettings>>, 
}

impl GitHubClient {
    
    pub async fn new(
        config: GitHubConfig,
        settings: Arc<RwLock<AppFullSettings>>, 
    ) -> VisionFlowResult<Self> {
        let debug_enabled = crate::utils::logging::is_debug_enabled();

        if debug_enabled {
            debug!(
                "Initializing GitHub client - Owner: '{}', Repo: '{}', Base path: '{}'",
                config.owner, config.repo, config.base_path
            );
        }

        
        if debug_enabled {
            debug!("Configuring HTTP client - Timeout: 30s, User-Agent: github-api-client");
        }

        let client = Client::builder()
            .user_agent("github-api-client")
            .timeout(Duration::from_secs(30))
            .build()?;

        if debug_enabled {
            debug!("HTTP client configured successfully");
        }

        
        let decoded_path = urlencoding::decode(&config.base_path)
            .unwrap_or(std::borrow::Cow::Owned(config.base_path.clone()))
            .into_owned();

        if debug_enabled {
            debug!("Decoded base path: '{}'", decoded_path);
        }

        
        let base_path = decoded_path
            .trim_matches('/')
            .replace("
            .replace('\\', "/");

        if debug_enabled {
            debug!(
                "Cleaned base path: '{}' (original: '{}')",
                base_path, base_path
            );
            debug!("GitHub client initialization complete");
        }

        Ok(Self {
            client,
            token: config.token,
            owner: config.owner,
            repo: config.repo,
            base_path,
            settings: Arc::clone(&settings),
        })
    }

    

    
    pub(crate) async fn get_full_path(&self, path: &str) -> String {
        let settings = self.settings.read().await;
        let debug_enabled = crate::utils::logging::is_debug_enabled();
        drop(settings);

        if debug_enabled {
            debug!(
                "Getting full path - Base: '{}', Input path: '{}'",
                self.base_path, path
            );
        }

        let base = self.base_path.trim_matches('/');
        let path = path.trim_matches('/');

        if debug_enabled {
            log::debug!("Trimmed paths - Base: '{}', Path: '{}'", base, path);
        }

        
        let decoded_path = urlencoding::decode(path)
            .unwrap_or(std::borrow::Cow::Owned(path.to_string()))
            .into_owned();
        let decoded_base = urlencoding::decode(base)
            .unwrap_or(std::borrow::Cow::Owned(base.to_string()))
            .into_owned();

        if debug_enabled {
            log::debug!(
                "Decoded paths - Base: '{}', Path: '{}'",
                decoded_base,
                decoded_path
            );
        }

        let full_path = if decoded_base.is_empty() {
            if debug_enabled {
                log::debug!(
                    "Base path is empty, using decoded path only: '{}'",
                    decoded_path
                );
            }
            decoded_path
        } else {
            if decoded_path.is_empty() {
                if debug_enabled {
                    log::debug!("Path is empty, using base path only: '{}'", decoded_base);
                }
                decoded_base
            } else if decoded_path.starts_with(&decoded_base) {
                
                if debug_enabled {
                    log::debug!(
                        "Path already contains base path, using as-is: '{}'",
                        decoded_path
                    );
                }
                decoded_path
            } else {
                let combined = format!("{}/{}", decoded_base, decoded_path);
                if debug_enabled {
                    log::debug!("Combined path: '{}'", combined);
                }
                combined
            }
        };

        
        let encoded = urlencoding::encode(&full_path).into_owned();

        if debug_enabled {
            log::debug!("Final encoded full path: '{}'", encoded);
        }

        encoded
    }

    
    pub(crate) async fn get_contents_url(&self, path: &str) -> String {
        let settings = self.settings.read().await;
        let _debug_enabled = crate::utils::logging::is_debug_enabled();
        drop(settings);

        info!("get_contents_url: Building GitHub API URL - Owner: '{}', Repo: '{}', Base path: '{}', Input path: '{}'",
            self.owner, self.repo, self.base_path, path);

        let full_path = self.get_full_path(path).await;

        info!(
            "get_contents_url: Full path after encoding: '{}'",
            full_path
        );

        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner, self.repo, full_path
        );

        info!("get_contents_url: Final GitHub API URL: '{}'", url);

        url
    }

    
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    
    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    
    pub(crate) fn owner(&self) -> &str {
        &self.owner
    }

    
    pub(crate) fn repo(&self) -> &str {
        &self.repo
    }

    
    pub(crate) fn base_path(&self) -> &str {
        &self.base_path
    }

    
    #[allow(dead_code)]
    pub(crate) fn settings(&self) -> &Arc<RwLock<AppFullSettings>> {
        
        &self.settings
    }

    
}

//! Solid Proxy Handler
//!
//! Proxies requests to JavaScript Solid Server (JSS) with NIP-98 authentication.
//! Routes: /solid/* -> JSS
//!
//! Features:
//! - NIP-98 authentication header generation
//! - Content negotiation passthrough (JSON-LD, Turtle)
//! - LDP CRUD operations (GET, PUT, POST, DELETE, PATCH, HEAD)
//! - Pod management endpoints

use actix_web::{web, HttpRequest, HttpResponse, http::Method};
use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::models::protected_settings::NostrUser;
use crate::services::nostr_service::NostrService;
use crate::utils::nip98::{generate_nip98_token, build_auth_header, Nip98Config};
use nostr_sdk::Keys;

/// JSS configuration from environment
#[derive(Debug, Clone)]
pub struct JssConfig {
    pub base_url: String,
    pub ws_url: String,
}

impl JssConfig {
    pub fn from_env() -> Self {
        Self {
            base_url: std::env::var("JSS_URL").unwrap_or_else(|_| "http://jss:3030".to_string()),
            ws_url: std::env::var("JSS_WS_URL")
                .unwrap_or_else(|_| "ws://jss:3030/.notifications".to_string()),
        }
    }
}

/// Response from pod creation
#[derive(Debug, Serialize, Deserialize)]
pub struct PodCreationResponse {
    pub pod_url: String,
    pub webid: String,
}

/// Error response structure
#[derive(Debug, Serialize)]
pub struct SolidProxyError {
    pub error: String,
    pub details: Option<String>,
}

/// Shared state for the proxy
pub struct SolidProxyState {
    pub config: JssConfig,
    pub http_client: Client,
    /// Server-side signing key for proxied requests (optional)
    /// If None, requests are forwarded without NIP-98 headers
    pub server_keys: Option<Keys>,
}

impl SolidProxyState {
    pub fn new() -> Self {
        let server_keys = std::env::var("SOLID_PROXY_SECRET_KEY")
            .ok()
            .and_then(|hex| {
                nostr_sdk::SecretKey::from_hex(&hex)
                    .ok()
                    .map(Keys::new)
            });

        if server_keys.is_some() {
            info!("Solid proxy initialized with server-side signing key");
        } else {
            info!("Solid proxy initialized without server-side signing (passthrough mode)");
        }

        Self {
            config: JssConfig::from_env(),
            http_client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            server_keys,
        }
    }
}

impl Default for SolidProxyState {
    fn default() -> Self {
        Self::new()
    }
}

/// Main proxy handler for all /solid/* routes
pub async fn handle_solid_proxy(
    req: HttpRequest,
    body: web::Bytes,
    path: web::Path<String>,
    state: web::Data<SolidProxyState>,
    nostr_service: web::Data<NostrService>,
) -> HttpResponse {
    let target_path = path.into_inner();
    let method = req.method().clone();

    debug!(
        "Solid proxy: {} /solid/{} -> JSS",
        method, target_path
    );

    // Build target URL
    let target_url = format!("{}/{}", state.config.base_url, target_path);

    // Build the proxied request
    let mut proxy_req = match method.as_str() {
        "GET" => state.http_client.get(&target_url),
        "HEAD" => state.http_client.head(&target_url),
        "PUT" => state.http_client.put(&target_url),
        "POST" => state.http_client.post(&target_url),
        "DELETE" => state.http_client.delete(&target_url),
        "PATCH" => state.http_client.patch(&target_url),
        _ => {
            return HttpResponse::MethodNotAllowed().json(SolidProxyError {
                error: "Method not allowed".to_string(),
                details: Some(format!("Unsupported method: {}", method)),
            });
        }
    };

    // Forward relevant headers
    for (name, value) in req.headers() {
        let name_str = name.as_str().to_lowercase();
        // Forward these headers to JSS
        if matches!(
            name_str.as_str(),
            "accept" | "content-type" | "if-match" | "if-none-match" | "slug" | "link"
        ) {
            if let Ok(val) = value.to_str() {
                proxy_req = proxy_req.header(name.as_str(), val);
            }
        }
    }

    // Dual-header auth: Server identity for proxy authorization + User identity passthrough
    if let Some(keys) = &state.server_keys {
        let config = Nip98Config {
            url: target_url.clone(),
            method: method.to_string(),
            body: if body.is_empty() {
                None
            } else {
                String::from_utf8(body.to_vec()).ok()
            },
        };

        // Server identity for proxy authorization (JSS trusts this proxy)
        match generate_nip98_token(keys, &config) {
            Ok(token) => {
                proxy_req = proxy_req.header("X-Proxy-Authorization", build_auth_header(&token));
                debug!("Added X-Proxy-Authorization header for server identity");
            }
            Err(e) => {
                warn!("Failed to generate NIP-98 token for proxy: {}", e);
            }
        }

        // Pass through user's original NIP-98 for identity (ACL checks use this)
        if let Some(user_auth) = req.headers().get("Authorization") {
            if let Ok(val) = user_auth.to_str() {
                proxy_req = proxy_req.header("X-User-Authorization", val);
                debug!("Forwarded user auth as X-User-Authorization");
            }
        }
    } else {
        // Passthrough mode: forward Authorization directly
        if let Some(auth) = req.headers().get("Authorization") {
            if let Ok(val) = auth.to_str() {
                proxy_req = proxy_req.header("Authorization", val);
            }
        }
    }

    // Add body for methods that support it
    if !body.is_empty() && matches!(method.as_str(), "PUT" | "POST" | "PATCH") {
        proxy_req = proxy_req.body(body.to_vec());
    }

    // Execute the request
    match proxy_req.send().await {
        Ok(response) => {
            let status = response.status();
            let mut builder = HttpResponse::build(
                actix_web::http::StatusCode::from_u16(status.as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR),
            );

            // Forward response headers
            for (name, value) in response.headers() {
                let name_str = name.as_str().to_lowercase();
                // Forward these headers from JSS
                if matches!(
                    name_str.as_str(),
                    "content-type"
                        | "etag"
                        | "last-modified"
                        | "link"
                        | "location"
                        | "updates-via"
                        | "wac-allow"
                        | "accept-patch"
                        | "accept-post"
                        | "allow"
                        | "ms-author-via"
                ) {
                    if let Ok(val) = value.to_str() {
                        builder.insert_header((name.as_str(), val));
                    }
                }
            }

            // Get response body
            match response.bytes().await {
                Ok(bytes) => builder.body(bytes.to_vec()),
                Err(e) => {
                    error!("Failed to read JSS response body: {}", e);
                    HttpResponse::BadGateway().json(SolidProxyError {
                        error: "Failed to read response".to_string(),
                        details: Some(e.to_string()),
                    })
                }
            }
        }
        Err(e) => {
            error!("Solid proxy request failed: {}", e);
            HttpResponse::BadGateway().json(SolidProxyError {
                error: "Proxy request failed".to_string(),
                details: Some(e.to_string()),
            })
        }
    }
}

/// Create a new pod for a user based on their Nostr identity
pub async fn create_pod(
    req: HttpRequest,
    body: web::Json<CreatePodRequest>,
    state: web::Data<SolidProxyState>,
    nostr_service: web::Data<NostrService>,
) -> HttpResponse {
    // Get user from session/token
    let user = match get_user_from_request(&req, &nostr_service).await {
        Some(u) => u,
        None => {
            return HttpResponse::Unauthorized().json(SolidProxyError {
                error: "Authentication required".to_string(),
                details: Some("Valid Nostr session required to create pod".to_string()),
            });
        }
    };

    let npub = &user.npub;
    info!("Creating pod for user: {}", npub);

    // Create pod via JSS API
    let pod_url = format!("{}/.pods", state.config.base_url);
    let pod_request = serde_json::json!({
        "name": npub,
        "webId": format!("did:nostr:{}", user.pubkey)
    });

    let response = state
        .http_client
        .post(&pod_url)
        .json(&pod_request)
        .send()
        .await;

    match response {
        Ok(resp) if resp.status().is_success() => {
            let pod_url = format!("{}/pods/{}/", state.config.base_url, npub);
            HttpResponse::Created().json(PodCreationResponse {
                pod_url: pod_url.clone(),
                webid: format!("{}/profile/card#me", pod_url),
            })
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("JSS pod creation failed: {} - {}", status, body);
            HttpResponse::build(
                actix_web::http::StatusCode::from_u16(status.as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR),
            )
            .json(SolidProxyError {
                error: "Pod creation failed".to_string(),
                details: Some(body),
            })
        }
        Err(e) => {
            error!("Failed to create pod: {}", e);
            HttpResponse::BadGateway().json(SolidProxyError {
                error: "Failed to connect to Solid server".to_string(),
                details: Some(e.to_string()),
            })
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CreatePodRequest {
    /// Optional custom pod name (defaults to npub)
    pub name: Option<String>,
}

/// Check if a pod exists for the current user
pub async fn check_pod_exists(
    req: HttpRequest,
    state: web::Data<SolidProxyState>,
    nostr_service: web::Data<NostrService>,
) -> HttpResponse {
    let user = match get_user_from_request(&req, &nostr_service).await {
        Some(u) => u,
        None => {
            return HttpResponse::Unauthorized().json(SolidProxyError {
                error: "Authentication required".to_string(),
                details: None,
            });
        }
    };

    let pod_url = format!("{}/pods/{}/", state.config.base_url, user.npub);

    match state.http_client.head(&pod_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            HttpResponse::Ok().json(serde_json::json!({
                "exists": true,
                "pod_url": pod_url,
                "webid": format!("{}/profile/card#me", pod_url)
            }))
        }
        Ok(resp) if resp.status().as_u16() == 404 => {
            HttpResponse::Ok().json(serde_json::json!({
                "exists": false,
                "suggested_url": pod_url
            }))
        }
        Ok(resp) => {
            HttpResponse::build(
                actix_web::http::StatusCode::from_u16(resp.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR),
            )
            .json(SolidProxyError {
                error: "Failed to check pod".to_string(),
                details: None,
            })
        }
        Err(e) => {
            HttpResponse::BadGateway().json(SolidProxyError {
                error: "Failed to connect to Solid server".to_string(),
                details: Some(e.to_string()),
            })
        }
    }
}

/// Get user from request using session token
async fn get_user_from_request(
    req: &HttpRequest,
    nostr_service: &web::Data<NostrService>,
) -> Option<NostrUser> {
    // Try to get token from Authorization header
    let auth_header = req.headers().get("Authorization")?;
    let auth_str = auth_header.to_str().ok()?;

    // Extract Bearer token
    if auth_str.starts_with("Bearer ") {
        let token = &auth_str[7..];
        nostr_service.get_session(token).await
    } else {
        None
    }
}

/// Configure Solid proxy routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.app_data(web::Data::new(SolidProxyState::new()))
        .service(
            web::scope("/solid")
                // Pod management
                .route("/pods", web::post().to(create_pod))
                .route("/pods/check", web::get().to(check_pod_exists))
                // LDP proxy for all other paths
                .route("/{tail:.*}", web::get().to(handle_solid_proxy))
                .route("/{tail:.*}", web::put().to(handle_solid_proxy))
                .route("/{tail:.*}", web::post().to(handle_solid_proxy))
                .route("/{tail:.*}", web::delete().to(handle_solid_proxy))
                .route("/{tail:.*}", web::head().to(handle_solid_proxy))
                .route(
                    "/{tail:.*}",
                    web::method(Method::PATCH).to(handle_solid_proxy),
                ),
        );
}

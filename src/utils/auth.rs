use actix_web::{HttpRequest, HttpResponse};
use log::warn;
use tracing::{debug, info};
use uuid::Uuid;
use crate::services::nostr_service::NostrService;

#[derive(Clone, Debug)]
pub enum AccessLevel {
    Authenticated,
    PowerUser,
}

pub async fn verify_access(
    req: &HttpRequest,
    nostr_service: &NostrService,
    required_level: AccessLevel,
) -> Result<String, HttpResponse> {
    let request_id = req
        .headers()
        .get("X-Request-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&Uuid::new_v4().to_string())
        .to_string();

    // --- NIP-98 Schnorr auth (primary path) ---
    if let Some(auth_value) = req
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
    {
        if auth_value.starts_with("Nostr ") {
            // Behind a TLS-terminating proxy, connection_info returns internal
            // scheme/host; prefer X-Forwarded-* headers from the proxy.
            let conn_info = req.connection_info();
            let scheme = req.headers()
                .get("X-Forwarded-Proto")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_else(|| conn_info.scheme());
            let host = req.headers()
                .get("X-Forwarded-Host")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_else(|| conn_info.host());
            let url = format!(
                "{}://{}{}",
                scheme,
                host,
                req.uri()
                    .path_and_query()
                    .map(|pq| pq.as_str())
                    .unwrap_or("/")
            );
            let method = req.method().as_str();

            match nostr_service
                .verify_nip98_auth(auth_value, &url, method, None)
                .await
            {
                Ok(user) => {
                    info!(
                        request_id = %request_id,
                        pubkey = %user.pubkey,
                        "NIP-98 auth successful"
                    );
                    match required_level {
                        AccessLevel::Authenticated => return Ok(user.pubkey),
                        AccessLevel::PowerUser => {
                            if user.is_power_user {
                                return Ok(user.pubkey);
                            } else {
                                warn!("Non-power user {} attempted restricted operation", user.pubkey);
                                return Err(HttpResponse::Forbidden()
                                    .body("This operation requires power user access"));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("[{}] NIP-98 validation failed: {}", request_id, e);
                    return Err(
                        HttpResponse::Unauthorized().body(format!("NIP-98 auth failed: {}", e))
                    );
                }
            }
        }
    }

    // --- Legacy path: X-Nostr-Pubkey + X-Nostr-Token ---
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            debug!(
                request_id = %request_id,
                "Authentication failed - missing pubkey header"
            );
            return Err(HttpResponse::Forbidden().body("Authentication required"));
        }
    };

    let token = match req.headers().get("X-Nostr-Token") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr token in request headers");
            debug!(
                request_id = %request_id,
                has_pubkey = true,
                "Authentication failed - missing token header"
            );
            return Err(HttpResponse::Forbidden().body("Authentication required"));
        }
    };

    debug!(
        request_id = %request_id,
        has_pubkey = !pubkey.is_empty(),
        has_token = !token.is_empty(),
        pubkey_prefix = %&pubkey.chars().take(8).collect::<String>(),
        "Authentication headers extracted"
    );

    if !nostr_service.validate_session(&pubkey, &token).await {
        warn!("Invalid or expired session for user {}", pubkey);
        debug!(
            request_id = %request_id,
            pubkey = %pubkey,
            "Session validation failed"
        );
        return Err(HttpResponse::Unauthorized().body("Invalid or expired session"));
    }

    info!(
        request_id = %request_id,
        pubkey = %pubkey,
        "Session validated successfully"
    );

    match required_level {
        AccessLevel::Authenticated => {
            debug!(
                request_id = %request_id,
                pubkey = %pubkey,
                access_level = "authenticated",
                "Access granted"
            );
            Ok(pubkey)
        }
        AccessLevel::PowerUser => {
            if nostr_service.is_power_user(&pubkey).await {
                debug!(
                    request_id = %request_id,
                    pubkey = %pubkey,
                    access_level = "power_user",
                    "Power user access granted"
                );
                Ok(pubkey)
            } else {
                warn!("Non-power user {} attempted restricted operation", pubkey);
                debug!(
                    request_id = %request_id,
                    pubkey = %pubkey,
                    "Power user access denied"
                );
                Err(HttpResponse::Forbidden().body("This operation requires power user access"))
            }
        }
    }
}

// Helper function for handlers that require power user access
pub async fn verify_power_user(
    req: &HttpRequest,
    nostr_service: &NostrService,
) -> Result<String, HttpResponse> {
    verify_access(req, nostr_service, AccessLevel::PowerUser).await
}

// Helper function for handlers that require authentication
pub async fn verify_authenticated(
    req: &HttpRequest,
    nostr_service: &NostrService,
) -> Result<String, HttpResponse> {
    verify_access(req, nostr_service, AccessLevel::Authenticated).await
}
// src/settings/auth_extractor.rs
//! Authentication extractor for settings API endpoints
//! Supports dual-auth: NIP-98 Schnorr (primary) + Bearer token (legacy fallback)

use actix_web::{
    dev::Payload, error::ErrorUnauthorized, web, Error as ActixError, FromRequest, HttpRequest,
};
use futures_util::future::{ready, Ready};
use log::{debug, info, warn};

use crate::services::nostr_service::NostrService;

/// Authenticated user information extracted from NIP-98 or session token
#[derive(Clone, Debug)]
pub struct AuthenticatedUser {
    pub pubkey: String,
    pub is_power_user: bool,
}

impl AuthenticatedUser {
    /// Check if user has power user privileges
    pub fn require_power_user(&self) -> Result<(), ActixError> {
        if self.is_power_user {
            Ok(())
        } else {
            Err(ErrorUnauthorized("Power user access required"))
        }
    }
}

/// Optional authenticated user - allows both authenticated and anonymous access
pub struct OptionalAuth(pub Option<AuthenticatedUser>);

impl FromRequest for AuthenticatedUser {
    type Error = ActixError;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
        // Dev mode bypass: allow unauthenticated settings writes when explicitly enabled
        // SECURITY: Requires SETTINGS_AUTH_BYPASS=true in environment (only set in dev compose)
        if std::env::var("SETTINGS_AUTH_BYPASS").unwrap_or_default() == "true" {
            debug!("Settings auth bypass enabled (SETTINGS_AUTH_BYPASS=true) - using dev user");
            return ready(Ok(AuthenticatedUser {
                pubkey: "dev-user".to_string(),
                is_power_user: true,
            }));
        }

        // Extract NostrService from app data
        let nostr_service = match req.app_data::<web::Data<NostrService>>() {
            Some(service) => service,
            None => {
                warn!("NostrService not found in app data");
                return ready(Err(ErrorUnauthorized("Authentication service unavailable")));
            }
        };

        // Extract Authorization header
        let auth_header = match req.headers().get("Authorization") {
            Some(header) => match header.to_str() {
                Ok(s) => s,
                Err(_) => {
                    debug!("Invalid Authorization header format");
                    return ready(Err(ErrorUnauthorized("Invalid authorization header")));
                }
            },
            None => {
                debug!("Missing Authorization header");
                return ready(Err(ErrorUnauthorized("Missing authorization token")));
            }
        };

        // --- NIP-98 Schnorr auth (primary path) ---
        if auth_header.starts_with("Nostr ") {
            // Reconstruct the request URL for NIP-98 validation
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
                req.uri().path_and_query().map(|pq| pq.as_str()).unwrap_or("/")
            );
            let method = req.method().as_str().to_string();
            let auth_header_owned = auth_header.to_string();

            // verify_nip98_auth is async — need thread spawn (FromRequest is sync)
            let nostr_service = nostr_service.clone();
            let user_result = match std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
                rt.block_on(async {
                    nostr_service
                        .verify_nip98_auth(&auth_header_owned, &url, &method, None)
                        .await
                })
            })
            .join()
            {
                Ok(result) => result,
                Err(_) => {
                    warn!("NIP-98 auth thread panicked");
                    return ready(Err(ErrorUnauthorized("Authentication error")));
                }
            };

            return match user_result {
                Ok(user) => {
                    info!("NIP-98 authenticated user: {}", user.pubkey);
                    ready(Ok(AuthenticatedUser {
                        pubkey: user.pubkey,
                        is_power_user: user.is_power_user,
                    }))
                }
                Err(e) => {
                    warn!("NIP-98 auth failed: {}", e);
                    ready(Err(ErrorUnauthorized(format!("NIP-98 auth failed: {}", e))))
                }
            };
        }

        // --- Legacy Bearer token path (fallback) ---
        let token = match auth_header.strip_prefix("Bearer ") {
            Some(t) => t,
            None => {
                debug!("Authorization header has unrecognized prefix");
                return ready(Err(ErrorUnauthorized("Invalid authorization format")));
            }
        };

        // Extract pubkey from X-Nostr-Pubkey header (required for Bearer path)
        let pubkey = match req.headers().get("X-Nostr-Pubkey") {
            Some(header) => match header.to_str() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    debug!("Invalid X-Nostr-Pubkey header format");
                    return ready(Err(ErrorUnauthorized("Invalid pubkey header")));
                }
            },
            None => {
                debug!("Missing X-Nostr-Pubkey header");
                return ready(Err(ErrorUnauthorized("Missing pubkey")));
            }
        };

        // Dev-mode session bypass - requires SETTINGS_AUTH_BYPASS in environment
        if std::env::var("SETTINGS_AUTH_BYPASS").unwrap_or_default() == "true"
            && token == "dev-session-token"
        {
            debug!("Dev-mode session token accepted for pubkey: {}", pubkey);
            return ready(Ok(AuthenticatedUser {
                pubkey,
                is_power_user: true,
            }));
        }

        // Clone service for async validation
        let nostr_service = nostr_service.clone();
        let token = token.to_string();

        let pubkey_clone = pubkey.clone();
        let nostr_clone = nostr_service.clone();
        let token_clone = token.clone();
        let is_valid = match std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
            rt.block_on(async {
                nostr_clone.validate_session(&pubkey_clone, &token_clone).await
            })
        })
        .join()
        {
            Ok(valid) => valid,
            Err(_) => {
                warn!("Session validation thread panicked");
                return ready(Err(ErrorUnauthorized("Validation error")));
            }
        };

        if !is_valid {
            debug!("Session validation failed for pubkey: {}", pubkey);
            return ready(Err(ErrorUnauthorized("Invalid or expired session")));
        }

        // Get user details
        let pubkey_clone2 = pubkey.clone();
        let user_option = match std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
            rt.block_on(async { nostr_service.get_user(&pubkey_clone2).await })
        })
        .join()
        {
            Ok(user) => user,
            Err(_) => {
                warn!("User lookup thread panicked");
                return ready(Err(ErrorUnauthorized("User lookup error")));
            }
        };

        match user_option {
            Some(user) => {
                debug!("Successfully authenticated user: {}", pubkey);
                ready(Ok(AuthenticatedUser {
                    pubkey: user.pubkey,
                    is_power_user: user.is_power_user,
                }))
            }
            None => {
                warn!("User not found after successful validation: {}", pubkey);
                ready(Err(ErrorUnauthorized("User not found")))
            }
        }
    }
}

impl FromRequest for OptionalAuth {
    type Error = ActixError;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, payload: &mut Payload) -> Self::Future {
        match AuthenticatedUser::from_request(req, payload).into_inner() {
            Ok(user) => ready(Ok(OptionalAuth(Some(user)))),
            Err(_) => {
                // Authentication failed, but that's okay for optional auth
                debug!("Optional authentication: proceeding without authentication");
                ready(Ok(OptionalAuth(None)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_authenticated_user_power_check() {
        let power_user = AuthenticatedUser {
            pubkey: "test_pubkey".to_string(),
            is_power_user: true,
        };
        assert!(power_user.require_power_user().is_ok());

        let regular_user = AuthenticatedUser {
            pubkey: "test_pubkey".to_string(),
            is_power_user: false,
        };
        assert!(regular_user.require_power_user().is_err());
    }
}

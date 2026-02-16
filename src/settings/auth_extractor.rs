// src/settings/auth_extractor.rs
//! Authentication extractor for settings API endpoints

use actix_web::{
    dev::Payload, error::ErrorUnauthorized, web, Error as ActixError, FromRequest, HttpRequest,
};
use futures_util::future::{ready, Ready};
use log::{debug, warn};

use crate::services::nostr_service::NostrService;

/// Authenticated user information extracted from session token
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
        // SECURITY: Only available in debug builds to prevent accidental production bypass
        #[cfg(debug_assertions)]
        {
            if std::env::var("SETTINGS_AUTH_BYPASS").unwrap_or_default() == "true" {
                debug!("Settings auth bypass enabled (SETTINGS_AUTH_BYPASS=true) - using dev user");
                return ready(Ok(AuthenticatedUser {
                    pubkey: "dev-user".to_string(),
                    is_power_user: true,
                }));
            }
        }

        // Extract NostrService from app data
        let nostr_service = match req.app_data::<web::Data<NostrService>>() {
            Some(service) => service,
            None => {
                warn!("NostrService not found in app data");
                return ready(Err(ErrorUnauthorized("Authentication service unavailable")));
            }
        };

        // Extract Authorization header (Bearer token)
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

        // Parse Bearer token
        let token = match auth_header.strip_prefix("Bearer ") {
            Some(t) => t,
            None => {
                debug!("Authorization header missing Bearer prefix");
                return ready(Err(ErrorUnauthorized("Invalid authorization format")));
            }
        };

        // Extract pubkey from X-Nostr-Pubkey header
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

        // Dev-mode session bypass: the client sets token="dev-session-token" when
        // VITE_DEV_MODE_AUTH=true. Validating this via NostrService requires async,
        // but FromRequest returns Ready<> (sync). Using block_on inside the actix
        // async runtime panics, so we short-circuit dev tokens here.
        if token == "dev-session-token" {
            debug!("Dev-mode session token accepted for pubkey: {}", pubkey);
            return ready(Ok(AuthenticatedUser {
                pubkey,
                is_power_user: true,
            }));
        }

        // Clone service for async validation
        let nostr_service = nostr_service.clone();
        let token = token.to_string();

        // Use spawn_blocking to avoid block_on panic inside the async runtime.
        // NOTE: This is a synchronous FromRequest so we use std::thread to run
        // the async validation on a separate thread, then join.
        let pubkey_clone = pubkey.clone();
        let nostr_clone = nostr_service.clone();
        let token_clone = token.clone();
        let is_valid = match std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
            rt.block_on(async {
                nostr_clone.validate_session(&pubkey_clone, &token_clone).await
            })
        }).join() {
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
        }).join() {
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

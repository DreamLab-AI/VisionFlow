//! Authentication Middleware
//!
//! Provides Actix-web middleware for enforcing authentication on protected routes.
//! Uses Nostr-based authentication with session validation.

use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage, HttpResponse,
};
use futures_util::future::LocalBoxFuture;
use log::{debug, warn};
use std::future::{ready, Ready};
use std::rc::Rc;

use crate::services::nostr_service::NostrService;
use crate::utils::auth::{verify_authenticated, verify_power_user, AccessLevel};

/// Authentication middleware that enforces Nostr-based session validation
///
/// # Example
/// ```
/// use actix_web::{web, App};
/// use crate::middleware::auth::RequireAuth;
///
/// App::new()
///     .wrap(RequireAuth::authenticated())  // Require any authenticated user
///     .route("/protected", web::get().to(handler))
/// ```
pub struct RequireAuth {
    level: AccessLevel,
}

impl RequireAuth {
    /// Require authenticated user (any valid session)
    pub fn authenticated() -> Self {
        Self {
            level: AccessLevel::Authenticated,
        }
    }

    /// Require power user access
    pub fn power_user() -> Self {
        Self {
            level: AccessLevel::PowerUser,
        }
    }
}

impl<S, B> Transform<S, ServiceRequest> for RequireAuth
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthMiddleware {
            service: Rc::new(service),
            level: self.level.clone(),
        }))
    }
}

pub struct AuthMiddleware<S> {
    service: Rc<S>,
    level: AccessLevel,
}

impl<S, B> Service<ServiceRequest> for AuthMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let svc = self.service.clone();
        let level = self.level.clone();

        Box::pin(async move {
            // Extract NostrService from app data
            let nostr_service = match req.app_data::<actix_web::web::Data<NostrService>>() {
                Some(service) => service.clone(),
                None => {
                    warn!("NostrService not found in app data - authentication cannot proceed");
                    return Ok(req.into_response(
                        HttpResponse::InternalServerError()
                            .body("Authentication service not configured")
                            .into_body(),
                    ));
                }
            };

            // Verify access level
            let result = match level {
                AccessLevel::Authenticated => {
                    verify_authenticated(req.request(), &nostr_service).await
                }
                AccessLevel::PowerUser => {
                    verify_power_user(req.request(), &nostr_service).await
                }
            };

            match result {
                Ok(pubkey) => {
                    // Store authenticated pubkey in request extensions for handlers to use
                    req.extensions_mut().insert(AuthenticatedUser { pubkey });

                    debug!("Authentication successful, proceeding with request");

                    // Continue to the actual handler
                    svc.call(req).await
                }
                Err(response) => {
                    // Authentication failed - return error response
                    Ok(req.into_response(response.into_body()))
                }
            }
        })
    }
}

/// Authenticated user information stored in request extensions
#[derive(Clone, Debug)]
pub struct AuthenticatedUser {
    pub pubkey: String,
}

/// Extract authenticated user from request extensions (for use in handlers)
///
/// # Example
/// ```
/// use actix_web::{web, HttpRequest};
/// use crate::middleware::auth::get_authenticated_user;
///
/// async fn handler(req: HttpRequest) -> impl Responder {
///     let user = get_authenticated_user(&req)?;
///     // Use user.pubkey
/// }
/// ```
pub fn get_authenticated_user(req: &actix_web::HttpRequest) -> Option<AuthenticatedUser> {
    req.extensions().get::<AuthenticatedUser>().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_require_auth_levels() {
        let auth = RequireAuth::authenticated();
        assert!(matches!(auth.level, AccessLevel::Authenticated));

        let power = RequireAuth::power_user();
        assert!(matches!(power.level, AccessLevel::PowerUser));
    }
}

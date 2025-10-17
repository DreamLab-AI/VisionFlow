use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage, HttpResponse,
};
use futures_util::future::LocalBoxFuture;
use std::future::{ready, Ready};
use std::rc::Rc;
use log::{debug, warn};

use crate::services::nostr_service::NostrService;
use crate::services::user_service::{UserService, UserServiceError};

pub struct AuthContext {
    pub user_id: i64,
    pub nostr_pubkey: String,
    pub is_power_user: bool,
}

pub enum PermissionLevel {
    Authenticated,
    PowerUser,
}

pub struct PermissionMiddleware {
    nostr_service: Rc<NostrService>,
    user_service: Rc<UserService>,
    required_level: PermissionLevel,
}

impl PermissionMiddleware {
    pub fn new(
        nostr_service: Rc<NostrService>,
        user_service: Rc<UserService>,
        required_level: PermissionLevel,
    ) -> Self {
        Self {
            nostr_service,
            user_service,
            required_level,
        }
    }

    pub fn authenticated(nostr_service: Rc<NostrService>, user_service: Rc<UserService>) -> Self {
        Self::new(nostr_service, user_service, PermissionLevel::Authenticated)
    }

    pub fn power_user(nostr_service: Rc<NostrService>, user_service: Rc<UserService>) -> Self {
        Self::new(nostr_service, user_service, PermissionLevel::PowerUser)
    }
}

impl<S, B> Transform<S, ServiceRequest> for PermissionMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = PermissionMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(PermissionMiddlewareService {
            service: Rc::new(service),
            nostr_service: self.nostr_service.clone(),
            user_service: self.user_service.clone(),
            required_level: match self.required_level {
                PermissionLevel::Authenticated => PermissionLevel::Authenticated,
                PermissionLevel::PowerUser => PermissionLevel::PowerUser,
            },
        }))
    }
}

pub struct PermissionMiddlewareService<S> {
    service: Rc<S>,
    nostr_service: Rc<NostrService>,
    user_service: Rc<UserService>,
    required_level: PermissionLevel,
}

impl<S, B> Service<ServiceRequest> for PermissionMiddlewareService<S>
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
        let service = self.service.clone();
        let nostr_service = self.nostr_service.clone();
        let user_service = self.user_service.clone();
        let required_level = match self.required_level {
            PermissionLevel::Authenticated => PermissionLevel::Authenticated,
            PermissionLevel::PowerUser => PermissionLevel::PowerUser,
        };

        Box::pin(async move {
            let pubkey = match req.headers().get("X-Nostr-Pubkey") {
                Some(value) => value.to_str().unwrap_or("").to_string(),
                None => {
                    warn!("Missing Nostr pubkey in request headers");
                    return Err(actix_web::error::ErrorForbidden("Authentication required"));
                }
            };

            let token = match req.headers().get("X-Nostr-Token") {
                Some(value) => value.to_str().unwrap_or("").to_string(),
                None => {
                    warn!("Missing Nostr token in request headers");
                    return Err(actix_web::error::ErrorForbidden("Authentication required"));
                }
            };

            if !nostr_service.validate_session(&pubkey, &token).await {
                warn!("Invalid or expired session for user {}", pubkey);
                return Err(actix_web::error::ErrorUnauthorized(
                    "Invalid or expired session",
                ));
            }

            let user = match user_service.get_user_by_nostr_pubkey(&pubkey).await {
                Ok(u) => u,
                Err(UserServiceError::UserNotFound) => {
                    match user_service
                        .create_or_update_user(&pubkey, None)
                        .await
                    {
                        Ok(u) => u,
                        Err(e) => {
                            warn!("Failed to create user for pubkey {}: {:?}", pubkey, e);
                            return Err(actix_web::error::ErrorInternalServerError(
                                "Failed to create user",
                            ));
                        }
                    }
                }
                Err(e) => {
                    warn!("Database error fetching user {}: {:?}", pubkey, e);
                    return Err(actix_web::error::ErrorInternalServerError(
                        "Database error",
                    ));
                }
            };

            match required_level {
                PermissionLevel::Authenticated => {
                    debug!("Authenticated access granted for user_id={}", user.id);
                }
                PermissionLevel::PowerUser => {
                    if !user.is_power_user {
                        warn!(
                            "Non-power user {} attempted restricted operation",
                            pubkey
                        );
                        return Err(actix_web::error::ErrorForbidden(
                            "This operation requires power user access",
                        ));
                    }
                    debug!("Power user access granted for user_id={}", user.id);
                }
            }

            let auth_context = AuthContext {
                user_id: user.id,
                nostr_pubkey: pubkey,
                is_power_user: user.is_power_user,
            };

            req.extensions_mut().insert(auth_context);

            service.call(req).await
        })
    }
}

pub fn extract_auth_context(req: &actix_web::HttpRequest) -> Option<AuthContext> {
    req.extensions().get::<AuthContext>().map(|ctx| AuthContext {
        user_id: ctx.user_id,
        nostr_pubkey: ctx.nostr_pubkey.clone(),
        is_power_user: ctx.is_power_user,
    })
}

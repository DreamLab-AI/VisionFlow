use actix_web::{web, Error, HttpResponse, HttpRequest};
use serde::{Deserialize, Serialize};
use log::{error, info};

use crate::services::user_service::{UserService, UserServiceError, User, AuditLogEntry};
use crate::middleware::permissions::extract_auth_context;

#[derive(Debug, Serialize)]
struct UsersListResponse {
    users: Vec<User>,
    count: usize,
}

#[derive(Debug, Serialize)]
struct UserResponse {
    user: User,
}

#[derive(Debug, Serialize)]
struct MessageResponse {
    message: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Serialize)]
struct AuditLogResponse {
    entries: Vec<AuditLogEntry>,
    count: usize,
}

#[derive(Debug, Deserialize)]
struct AuditLogQuery {
    key: Option<String>,
    user_id: Option<i64>,
    #[serde(default = "default_limit")]
    limit: i64,
}

fn default_limit() -> i64 {
    100
}

pub async fn list_users(
    req: HttpRequest,
    user_service: web::Data<UserService>,
) -> Result<HttpResponse, Error> {
    let auth_context = extract_auth_context(&req)
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Authentication required"))?;

    if !auth_context.is_power_user {
        return Ok(HttpResponse::Forbidden().json(ErrorResponse {
            error: "Power user access required".to_string(),
        }));
    }

    match user_service.list_all_users().await {
        Ok(users) => {
            let count = users.len();
            info!("Listed {} users for admin {}", count, auth_context.nostr_pubkey);
            Ok(HttpResponse::Ok().json(UsersListResponse { users, count }))
        }
        Err(e) => {
            error!("Failed to list users: {:?}", e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to fetch users".to_string(),
            }))
        }
    }
}

pub async fn get_user_by_pubkey(
    req: HttpRequest,
    user_service: web::Data<UserService>,
    path: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let auth_context = extract_auth_context(&req)
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Authentication required"))?;

    if !auth_context.is_power_user {
        return Ok(HttpResponse::Forbidden().json(ErrorResponse {
            error: "Power user access required".to_string(),
        }));
    }

    let pubkey = path.into_inner();

    match user_service.get_user_by_nostr_pubkey(&pubkey).await {
        Ok(user) => {
            info!("Fetched user {} for admin {}", pubkey, auth_context.nostr_pubkey);
            Ok(HttpResponse::Ok().json(UserResponse { user }))
        }
        Err(UserServiceError::UserNotFound) => {
            Ok(HttpResponse::NotFound().json(ErrorResponse {
                error: "User not found".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to fetch user {}: {:?}", pubkey, e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to fetch user".to_string(),
            }))
        }
    }
}

pub async fn grant_power_user(
    req: HttpRequest,
    user_service: web::Data<UserService>,
    path: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let auth_context = extract_auth_context(&req)
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Authentication required"))?;

    if !auth_context.is_power_user {
        return Ok(HttpResponse::Forbidden().json(ErrorResponse {
            error: "Power user access required".to_string(),
        }));
    }

    let pubkey = path.into_inner();

    let target_user = match user_service.get_user_by_nostr_pubkey(&pubkey).await {
        Ok(user) => user,
        Err(UserServiceError::UserNotFound) => {
            return Ok(HttpResponse::NotFound().json(ErrorResponse {
                error: "User not found".to_string(),
            }));
        }
        Err(e) => {
            error!("Failed to fetch user {}: {:?}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to fetch user".to_string(),
            }));
        }
    };

    match user_service.grant_power_user(target_user.id).await {
        Ok(()) => {
            info!(
                "Admin {} granted power user to user {}",
                auth_context.nostr_pubkey, pubkey
            );
            Ok(HttpResponse::Ok().json(MessageResponse {
                message: format!("Power user granted to {}", pubkey),
            }))
        }
        Err(e) => {
            error!("Failed to grant power user to {}: {:?}", pubkey, e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to grant power user".to_string(),
            }))
        }
    }
}

pub async fn revoke_power_user(
    req: HttpRequest,
    user_service: web::Data<UserService>,
    path: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let auth_context = extract_auth_context(&req)
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Authentication required"))?;

    if !auth_context.is_power_user {
        return Ok(HttpResponse::Forbidden().json(ErrorResponse {
            error: "Power user access required".to_string(),
        }));
    }

    let pubkey = path.into_inner();

    let target_user = match user_service.get_user_by_nostr_pubkey(&pubkey).await {
        Ok(user) => user,
        Err(UserServiceError::UserNotFound) => {
            return Ok(HttpResponse::NotFound().json(ErrorResponse {
                error: "User not found".to_string(),
            }));
        }
        Err(e) => {
            error!("Failed to fetch user {}: {:?}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to fetch user".to_string(),
            }));
        }
    };

    match user_service.revoke_power_user(target_user.id).await {
        Ok(()) => {
            info!(
                "Admin {} revoked power user from user {}",
                auth_context.nostr_pubkey, pubkey
            );
            Ok(HttpResponse::Ok().json(MessageResponse {
                message: format!("Power user revoked from {}", pubkey),
            }))
        }
        Err(e) => {
            error!("Failed to revoke power user from {}: {:?}", pubkey, e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to revoke power user".to_string(),
            }))
        }
    }
}

pub async fn get_audit_log(
    req: HttpRequest,
    user_service: web::Data<UserService>,
    query: web::Query<AuditLogQuery>,
) -> Result<HttpResponse, Error> {
    let auth_context = extract_auth_context(&req)
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Authentication required"))?;

    if !auth_context.is_power_user {
        return Ok(HttpResponse::Forbidden().json(ErrorResponse {
            error: "Power user access required".to_string(),
        }));
    }

    let limit = if query.limit > 1000 { 1000 } else { query.limit };

    match user_service
        .get_audit_log(query.key.clone(), query.user_id, limit)
        .await
    {
        Ok(entries) => {
            let count = entries.len();
            info!(
                "Fetched {} audit log entries for admin {}",
                count, auth_context.nostr_pubkey
            );
            Ok(HttpResponse::Ok().json(AuditLogResponse { entries, count }))
        }
        Err(e) => {
            error!("Failed to fetch audit log: {:?}", e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to fetch audit log".to_string(),
            }))
        }
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/admin")
            .route("/users", web::get().to(list_users))
            .route("/users/{pubkey}", web::get().to(get_user_by_pubkey))
            .route(
                "/users/{pubkey}/power-user",
                web::put().to(grant_power_user),
            )
            .route(
                "/users/{pubkey}/power-user",
                web::delete().to(revoke_power_user),
            )
            .route("/settings/audit", web::get().to(get_audit_log)),
    );
}

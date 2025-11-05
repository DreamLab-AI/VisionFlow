//! Middleware modules for request processing

pub mod auth;
pub mod timeout;

pub use auth::{get_authenticated_user, AuthenticatedUser, RequireAuth};
pub use timeout::TimeoutMiddleware;

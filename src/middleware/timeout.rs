//! Request timeout middleware
//!
//! Ensures all HTTP requests complete within a reasonable timeframe
//! to prevent hanging connections and resource exhaustion.

use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error,
};
use futures::future::LocalBoxFuture;
use log::error;
use std::future::{ready, Ready};
use std::time::Duration;

/
pub struct TimeoutMiddleware {
    timeout: Duration,
}

impl TimeoutMiddleware {
    
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }

    
    pub fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

impl<S, B> Transform<S, ServiceRequest> for TimeoutMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = TimeoutMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(TimeoutMiddlewareService {
            service,
            timeout: self.timeout,
        }))
    }
}

pub struct TimeoutMiddlewareService<S> {
    service: S,
    timeout: Duration,
}

impl<S, B> Service<ServiceRequest> for TimeoutMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let timeout_duration = self.timeout;
        let fut = self.service.call(req);

        Box::pin(async move {
            match tokio::time::timeout(timeout_duration, fut).await {
                Ok(result) => result,
                Err(_) => {
                    error!(
                        "Request timeout after {:?} - request exceeded maximum processing time",
                        timeout_duration
                    );
                    
                    Err(actix_web::error::ErrorGatewayTimeout(
                        "Request processing timeout - the server took too long to respond",
                    ))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App, HttpResponse};

    #[actix_web::test]
    async fn test_timeout_middleware_success() {
        let app = test::init_service(
            App::new()
                .wrap(TimeoutMiddleware::new(Duration::from_secs(5)))
                .route(
                    "/",
                    web::get().to(|| async { HttpResponse::Ok().body("OK") }),
                ),
        )
        .await;

        let req = test::TestRequest::get().uri("/").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_timeout_middleware_timeout() {
        let app = test::init_service(
            App::new()
                .wrap(TimeoutMiddleware::new(Duration::from_millis(100)))
                .route(
                    "/slow",
                    web::get().to(|| async {
                        tokio::time::sleep(Duration::from_secs(10)).await;
                        HttpResponse::Ok().body("Never reached")
                    }),
                ),
        )
        .await;

        let req = test::TestRequest::get().uri("/slow").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::GATEWAY_TIMEOUT);
    }
}

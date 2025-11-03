use crate::AppState;
use actix_web::{post, web, HttpResponse, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};
use serde_json::json;
use crate::{
    ok_json, created_json, error_json, bad_request, not_found,
    unauthorized, forbidden, conflict, no_content, accepted,
    too_many_requests, service_unavailable, payload_too_large
};


#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PerplexityRequest {
    pub query: String,
    pub conversation_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PerplexityResponse {
    pub answer: String,
    pub conversation_id: String,
}

#[post("")]
pub async fn handle_perplexity(
    state: web::Data<AppState>,
    request: web::Json<PerplexityRequest>,
) -> impl Responder {
    info!("Received perplexity request: {:?}", request);

    let perplexity_service = match &state.perplexity_service {
        Some(service) => service,
        None => {
            return service_unavailable!("Perplexity service is not available")
        }
    };

    let conversation_id = state.ragflow_session_id.clone();
    match perplexity_service
        .query(&request.query, &conversation_id)
        .await
    {
        Ok(answer) => {
            let response = PerplexityResponse {
                answer,
                conversation_id,
            };
            ok_json!(response)
        }
        Err(e) => {
            error!("Error processing perplexity request: {}", e);
            HttpResponse::InternalServerError().json(format!("Error: {}", e))
        }
    }
}

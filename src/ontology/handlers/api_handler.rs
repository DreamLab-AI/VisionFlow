// src/ontology/handlers/api_handler.rs

//! API handlers for ontology validation, reasoning, and configuration.

use actix_web::{web, HttpResponse, Responder};

/// Configures the routes for the ontology API.
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/ontology")
            .route("/validate", web::post().to(validate_ontology))
            .route("/report", web::get().to(get_validation_report))
            // TODO: Add other routes as per ontology-api-reference.md
    );
}

/// Handles requests to validate the ontology.
async fn validate_ontology() -> impl Responder {
    // TODO: Send a message to the OntologyActor to start validation
    HttpResponse::Accepted().json("Validation job started.")
}

/// Handles requests to retrieve the latest validation report.
async fn get_validation_report() -> impl Responder {
    // TODO: Send a message to the OntologyActor to get the report
    HttpResponse::Ok().json("Validation report placeholder.")
}

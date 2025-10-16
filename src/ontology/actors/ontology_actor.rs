// src/ontology/actors/ontology_actor.rs

//! The OntologyActor manages asynchronous validation and reasoning tasks.

use actix::prelude::*;
use crate::ontology::services::owl_validator::OwlValidatorService;

/// The main actor for ontology processing.
pub struct OntologyActor {
    validator: OwlValidatorService,
}

impl OntologyActor {
    /// Creates a new OntologyActor.
    pub fn new() -> Self {
        Self {
            validator: OwlValidatorService::new(),
        }
    }
}

/// Implement the Actor trait for OntologyActor.
impl Actor for OntologyActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        log::info!("OntologyActor has started.");
    }
}

// TODO: Implement message handlers for validation, inference, etc.

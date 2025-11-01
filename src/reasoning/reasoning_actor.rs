///
///
///
///
///
///

use actix::prelude::*;
use std::sync::Arc;
use crate::reasoning::{
    custom_reasoner::{CustomReasoner, InferredAxiom, OntologyReasoner, Ontology},
    inference_cache::InferenceCache,
    ReasoningResult,
};

///
pub struct ReasoningActor {
    reasoner: Arc<dyn OntologyReasoner + Send + Sync>,
    cache: Arc<InferenceCache>,
}

impl ReasoningActor {
    
    pub fn new(cache_db_path: &str) -> ReasoningResult<Self> {
        let reasoner = Arc::new(CustomReasoner::new()) as Arc<dyn OntologyReasoner + Send + Sync>;
        let cache = Arc::new(InferenceCache::new(cache_db_path)?);

        Ok(Self { reasoner, cache })
    }

    
    pub fn with_reasoner(
        reasoner: Arc<dyn OntologyReasoner + Send + Sync>,
        cache_db_path: &str,
    ) -> ReasoningResult<Self> {
        let cache = Arc::new(InferenceCache::new(cache_db_path)?);
        Ok(Self { reasoner, cache })
    }
}

impl Actor for ReasoningActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        log::info!("ReasoningActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        log::info!("ReasoningActor stopped");
    }
}

///
#[derive(Debug, Clone)]
pub enum ReasoningMessage {
    
    TriggerReasoning {
        ontology_id: i64,
        ontology: Ontology,
    },

    
    GetInferredAxioms {
        ontology_id: i64,
    },

    
    InvalidateCache {
        ontology_id: i64,
    },

    
    GetCacheStats,
}

///
#[derive(Message)]
#[rtype(result = "ReasoningResult<Vec<InferredAxiom>>")]
pub struct TriggerReasoning {
    pub ontology_id: i64,
    pub ontology: Ontology,
}

impl Handler<TriggerReasoning> for ReasoningActor {
    type Result = ResponseFuture<ReasoningResult<Vec<InferredAxiom>>>;

    fn handle(&mut self, msg: TriggerReasoning, _ctx: &mut Self::Context) -> Self::Result {
        let reasoner = Arc::clone(&self.reasoner);
        let cache = Arc::clone(&self.cache);

        Box::pin(async move {
            log::info!("Starting reasoning for ontology {}", msg.ontology_id);

            let result = cache.get_or_compute(
                msg.ontology_id,
                reasoner.as_ref(),
                &msg.ontology,
            );

            match &result {
                Ok(axioms) => {
                    log::info!(
                        "Reasoning complete for ontology {} ({} axioms inferred)",
                        msg.ontology_id,
                        axioms.len()
                    );
                }
                Err(e) => {
                    log::error!(
                        "Reasoning failed for ontology {}: {}",
                        msg.ontology_id,
                        e
                    );
                }
            }

            result
        })
    }
}

///
#[derive(Message)]
#[rtype(result = "ReasoningResult<Option<Vec<InferredAxiom>>>")]
pub struct GetInferredAxioms {
    pub ontology_id: i64,
}

impl Handler<GetInferredAxioms> for ReasoningActor {
    type Result = ResponseFuture<ReasoningResult<Option<Vec<InferredAxiom>>>>;

    fn handle(&mut self, msg: GetInferredAxioms, _ctx: &mut Self::Context) -> Self::Result {
        let cache = Arc::clone(&self.cache);

        Box::pin(async move {
            
            let cached = cache.load_from_cache(msg.ontology_id)?;
            Ok(cached.map(|c| c.inferred_axioms))
        })
    }
}

///
#[derive(Message)]
#[rtype(result = "ReasoningResult<()>")]
pub struct InvalidateCache {
    pub ontology_id: i64,
}

impl Handler<InvalidateCache> for ReasoningActor {
    type Result = ReasoningResult<()>;

    fn handle(&mut self, msg: InvalidateCache, _ctx: &mut Self::Context) -> Self::Result {
        log::info!("Invalidating cache for ontology {}", msg.ontology_id);
        self.cache.invalidate(msg.ontology_id)
    }
}

///
#[derive(Message)]
#[rtype(result = "ReasoningResult<crate::reasoning::inference_cache::CacheStats>")]
pub struct GetCacheStats;

impl Handler<GetCacheStats> for ReasoningActor {
    type Result = ReasoningResult<crate::reasoning::inference_cache::CacheStats>;

    fn handle(&mut self, _msg: GetCacheStats, _ctx: &mut Self::Context) -> Self::Result {
        self.cache.get_stats()
    }
}

// Note: load_from_cache is implemented in inference_cache.rs module

#[cfg(test)]
mod tests {
    use super::*;
    use actix::Actor;
    use tempfile::TempDir;
    use crate::reasoning::custom_reasoner::{Ontology, OWLClass};

    fn create_test_ontology() -> Ontology {
        let mut ontology = Ontology::default();

        ontology.classes.insert("A".to_string(), OWLClass {
            iri: "A".to_string(),
            label: Some("Class A".to_string()),
            parent_class_iri: None,
        });

        ontology.classes.insert("B".to_string(), OWLClass {
            iri: "B".to_string(),
            label: Some("Class B".to_string()),
            parent_class_iri: Some("A".to_string()),
        });

        ontology.subclass_of.insert("B".to_string(),
            vec!["A".to_string()].into_iter().collect());

        ontology
    }

    #[actix_rt::test]
    async fn test_reasoning_actor() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let actor = ReasoningActor::new(cache_path.to_str().unwrap())
            .unwrap()
            .start();

        let ontology = create_test_ontology();

        
        let result = actor.send(TriggerReasoning {
            ontology_id: 1,
            ontology: ontology.clone(),
        }).await;

        assert!(result.is_ok());
        let axioms = result.unwrap().unwrap();
        assert!(!axioms.is_empty());
    }

    #[actix_rt::test]
    async fn test_cache_invalidation() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let actor = ReasoningActor::new(cache_path.to_str().unwrap())
            .unwrap()
            .start();

        let ontology = create_test_ontology();

        
        actor.send(TriggerReasoning {
            ontology_id: 1,
            ontology: ontology.clone(),
        }).await.unwrap().unwrap();

        
        let result = actor.send(InvalidateCache {
            ontology_id: 1,
        }).await;

        assert!(result.is_ok());
    }

    #[actix_rt::test]
    async fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let actor = ReasoningActor::new(cache_path.to_str().unwrap())
            .unwrap()
            .start();

        let ontology = create_test_ontology();

        
        actor.send(TriggerReasoning {
            ontology_id: 1,
            ontology,
        }).await.unwrap().unwrap();

        
        let result = actor.send(GetCacheStats).await;
        assert!(result.is_ok());

        let stats = result.unwrap().unwrap();
        assert!(stats.total_entries > 0);
    }
}

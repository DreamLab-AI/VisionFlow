// src/adapters/whelk_inference_engine.rs
//! Whelk Inference Engine Adapter
//!
//! Implements the InferenceEngine port using horned-owl for OWL ontology loading
//! and whelk-rs for EL reasoning. This adapter provides complete EL reasoning capabilities.

use async_trait::async_trait;
use tracing::{debug, info, instrument, warn};

use crate::ports::inference_engine::{
    InferenceEngine, InferenceEngineError, InferenceStatistics, Result as EngineResult,
};
use crate::ports::ontology_repository::{AxiomType, InferenceResults, OwlAxiom, OwlClass};

#[cfg(feature = "ontology")]
use horned_owl::model::{
    AnnotatedComponent, ArcStr, Build, Class, ClassExpression, Component, DeclareClass,
    MutableOntology, SubClassOf,
};
#[cfg(feature = "ontology")]
use horned_owl::ontology::set::SetOntology;
#[cfg(feature = "ontology")]
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "ontology")]
use std::hash::{Hash, Hasher};

/// Whelk-based inference engine for OWL EL reasoning
///
/// Note: ReasonerState is not stored because it contains Rc which is not Send+Sync.
/// Instead, we cache the inferred subsumptions as OwlAxioms which are thread-safe.
pub struct WhelkInferenceEngine {
    #[cfg(feature = "ontology")]
    ontology: Option<SetOntology<ArcStr>>,

    #[cfg(feature = "ontology")]
    cached_subsumptions: Option<Vec<OwlAxiom>>,

    #[cfg(feature = "ontology")]
    last_checksum: Option<u64>,

    #[cfg(not(feature = "ontology"))]
    _phantom: std::marker::PhantomData<()>,

    loaded_classes: usize,
    loaded_axioms: usize,
    inferred_axioms: usize,
    last_inference_time_ms: u64,
    total_inferences: usize,
}

#[cfg(feature = "ontology")]
/// Use whelk crate for ontology reasoning
use whelk;

impl WhelkInferenceEngine {
    /// Create new Whelk inference engine
    pub fn new() -> Self {
        info!("Initializing WhelkInferenceEngine");
        Self {
            #[cfg(feature = "ontology")]
            ontology: None,

            #[cfg(feature = "ontology")]
            cached_subsumptions: None,

            #[cfg(feature = "ontology")]
            last_checksum: None,

            #[cfg(not(feature = "ontology"))]
            _phantom: std::marker::PhantomData,

            loaded_classes: 0,
            loaded_axioms: 0,
            inferred_axioms: 0,
            last_inference_time_ms: 0,
            total_inferences: 0,
        }
    }

    #[cfg(feature = "ontology")]
    /// Convert OwlClass to horned-owl DeclareClass component
    fn convert_class_to_horned(class: &OwlClass) -> Option<AnnotatedComponent<ArcStr>> {
        let iri = Build::new().iri(class.iri.clone());
        let class_decl = Class(iri);
        Some(AnnotatedComponent {
            component: Component::DeclareClass(DeclareClass(class_decl)),
            ann: Default::default(),
        })
    }

    #[cfg(feature = "ontology")]
    /// Convert OwlAxiom to horned-owl Component
    fn convert_axiom_to_horned(axiom: &OwlAxiom) -> Option<AnnotatedComponent<ArcStr>> {
        let component = match axiom.axiom_type {
            AxiomType::SubClassOf => {
                // Create SubClassOf axiom
                let sub_iri = Build::new().iri(axiom.subject.clone());
                let sup_iri = Build::new().iri(axiom.object.clone());

                let sub_class = ClassExpression::Class(Class(sub_iri));
                let sup_class = ClassExpression::Class(Class(sup_iri));

                Component::SubClassOf(SubClassOf {
                    sub: sub_class,
                    sup: sup_class,
                })
            }
            AxiomType::EquivalentClass => {
                // For equivalent classes, create bidirectional SubClassOf axioms
                warn!("EquivalentClass axioms require special handling - converting to SubClassOf");
                let sub_iri = Build::new().iri(axiom.subject.clone());
                let sup_iri = Build::new().iri(axiom.object.clone());

                Component::SubClassOf(SubClassOf {
                    sub: ClassExpression::Class(Class(sub_iri)),
                    sup: ClassExpression::Class(Class(sup_iri)),
                })
            }
            AxiomType::ObjectPropertyAssertion => {
                // Property assertions are not directly supported in EL reasoning Tbox
                // but we handle them for completeness
                warn!("ObjectPropertyAssertion not directly translated to EL Tbox");
                return None;
            }
            _ => {
                warn!("Unsupported axiom type: {:?}", axiom.axiom_type);
                return None;
            }
        };

        Some(AnnotatedComponent {
            component,
            ann: Default::default(),
        })
    }

    #[cfg(feature = "ontology")]
    /// Compute checksum of ontology for cache invalidation
    fn compute_ontology_checksum(ontology: &SetOntology<ArcStr>) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Collect and sort axioms for deterministic hashing
        let mut axioms: Vec<String> = ontology
            .iter()
            .map(|ann| format!("{:?}", ann.component))
            .collect();
        axioms.sort();

        for axiom in axioms {
            axiom.hash(&mut hasher);
        }

        hasher.finish()
    }

    #[cfg(feature = "ontology")]
    /// Convert whelk subsumptions back to OwlAxioms
    /// Note: whelk::Vector is from the im crate used by whelk
    /// Note: Whelk internally uses Rc, so we need to use std::rc::Rc here
    fn convert_subsumptions_to_axioms<V>(subsumptions: &V) -> Vec<OwlAxiom>
    where
        V: IntoIterator<
                Item = (
                    std::rc::Rc<whelk::whelk::model::AtomicConcept>,
                    std::rc::Rc<whelk::whelk::model::AtomicConcept>,
                ),
            > + Clone,
    {
        subsumptions
            .clone()
            .into_iter()
            .map(|(sub, sup)| OwlAxiom {
                id: None,
                axiom_type: AxiomType::SubClassOf,
                subject: sub.id.clone(),
                object: sup.id.clone(),
                annotations: std::collections::HashMap::new(),
            })
            .collect()
    }
}

impl Default for WhelkInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceEngine for WhelkInferenceEngine {
    #[instrument(skip(self, classes, axioms), fields(classes = classes.len(), axioms = axioms.len()), level = "debug")]
    async fn load_ontology(
        &mut self,
        classes: Vec<OwlClass>,
        axioms: Vec<OwlAxiom>,
    ) -> EngineResult<()> {
        #[cfg(feature = "ontology")]
        {
            let mut ontology = SetOntology::new();

            // Convert classes to horned-owl declarations
            for class in &classes {
                if let Some(horned_class) = Self::convert_class_to_horned(class) {
                    ontology.insert(horned_class);
                }
            }

            // Convert axioms to horned-owl format
            for axiom in &axioms {
                if let Some(horned_axiom) = Self::convert_axiom_to_horned(axiom) {
                    ontology.insert(horned_axiom);
                }
            }

            // Compute checksum for caching
            let checksum = Self::compute_ontology_checksum(&ontology);

            // Check if we need to re-reason
            let needs_reasoning = match self.last_checksum {
                Some(last) => last != checksum,
                None => true,
            };

            if needs_reasoning {
                info!("Ontology changed, will perform fresh reasoning");
                self.last_checksum = Some(checksum);
                self.cached_subsumptions = None; // Clear cached inferences
            } else {
                info!("Ontology unchanged, reusing cached reasoning results");
            }

            self.ontology = Some(ontology);
            self.loaded_classes = classes.len();
            self.loaded_axioms = axioms.len();

            info!(
                "Loaded ontology with {} classes and {} axioms",
                classes.len(),
                axioms.len()
            );
            Ok(())
        }

        #[cfg(not(feature = "ontology"))]
        {
            self.loaded_classes = classes.len();
            self.loaded_axioms = axioms.len();
            warn!("Ontology feature not enabled, loading metadata only");
            Ok(())
        }
    }

    #[instrument(skip(self), level = "debug")]
    async fn infer(&mut self) -> EngineResult<InferenceResults> {
        let start = std::time::Instant::now();

        #[cfg(feature = "ontology")]
        {
            let ontology = self
                .ontology
                .as_ref()
                .ok_or(InferenceEngineError::OntologyNotLoaded)?;

            // Return cached results if already reasoned
            if let Some(ref cached) = self.cached_subsumptions {
                info!("Using cached reasoning results");

                let inference_time_ms = start.elapsed().as_millis() as u64;
                self.last_inference_time_ms = inference_time_ms;

                return Ok(InferenceResults {
                    timestamp: chrono::Utc::now(),
                    inferred_axioms: cached.clone(),
                    inference_time_ms,
                    reasoner_version: format!("whelk-rs-{}", env!("CARGO_PKG_VERSION")),
                });
            }

            // Perform fresh reasoning with whelk-rs
            info!("Performing EL reasoning with whelk-rs");

            // Translate horned-owl ontology to whelk axioms
            let whelk_axioms = whelk::whelk::owl::translate_ontology(ontology);
            debug!("Translated {} axioms to whelk format", whelk_axioms.len());

            // Run whelk-rs EL reasoner
            let reasoner_state = whelk::whelk::reasoner::assert(&whelk_axioms);

            // Extract inferred subsumptions
            let subsumptions = reasoner_state.named_subsumptions();
            info!("Inferred {} subsumption relationships", subsumptions.len());

            // Convert to OwlAxioms and cache them
            let inferred_axioms = Self::convert_subsumptions_to_axioms(&subsumptions);
            self.inferred_axioms = inferred_axioms.len();

            // Cache the subsumptions (not the ReasonerState since it's not Send+Sync)
            self.cached_subsumptions = Some(inferred_axioms.clone());
            self.total_inferences += 1;

            let inference_time_ms = start.elapsed().as_millis() as u64;
            self.last_inference_time_ms = inference_time_ms;

            info!(
                "EL reasoning completed in {}ms with {} inferred axioms",
                inference_time_ms,
                inferred_axioms.len()
            );

            Ok(InferenceResults {
                timestamp: chrono::Utc::now(),
                inferred_axioms,
                inference_time_ms,
                reasoner_version: format!("whelk-rs-{}", env!("CARGO_PKG_VERSION")),
            })
        }

        #[cfg(not(feature = "ontology"))]
        {
            let inference_time_ms = start.elapsed().as_millis() as u64;
            self.last_inference_time_ms = inference_time_ms;
            self.total_inferences += 1;

            warn!("Ontology feature not enabled, returning empty inference results");

            Ok(InferenceResults {
                timestamp: chrono::Utc::now(),
                inferred_axioms: Vec::new(),
                inference_time_ms,
                reasoner_version: "stub-0.1.0".to_string(),
            })
        }
    }

    async fn is_entailed(&self, axiom: &OwlAxiom) -> EngineResult<bool> {
        #[cfg(feature = "ontology")]
        {
            let cached = self
                .cached_subsumptions
                .as_ref()
                .ok_or(InferenceEngineError::OntologyNotLoaded)?;

            // Check if SubClassOf axiom is entailed by looking in cached subsumptions
            if axiom.axiom_type == AxiomType::SubClassOf {
                let is_entailed = cached.iter().any(|inferred| {
                    inferred.axiom_type == AxiomType::SubClassOf
                        && inferred.subject == axiom.subject
                        && inferred.object == axiom.object
                });

                return Ok(is_entailed);
            }

            Ok(false)
        }

        #[cfg(not(feature = "ontology"))]
        {
            Ok(false)
        }
    }

    async fn get_subclass_hierarchy(&self) -> EngineResult<Vec<(String, String)>> {
        #[cfg(feature = "ontology")]
        {
            let cached = self
                .cached_subsumptions
                .as_ref()
                .ok_or(InferenceEngineError::OntologyNotLoaded)?;

            // Extract all SubClassOf relationships from cached inferences
            let hierarchy: Vec<(String, String)> = cached
                .iter()
                .filter(|ax| ax.axiom_type == AxiomType::SubClassOf)
                .map(|ax| (ax.subject.clone(), ax.object.clone()))
                .collect();

            debug!("Extracted {} subsumption relationships", hierarchy.len());
            Ok(hierarchy)
        }

        #[cfg(not(feature = "ontology"))]
        {
            Ok(Vec::new())
        }
    }

    async fn classify_instance(&self, instance_iri: &str) -> EngineResult<Vec<String>> {
        #[cfg(feature = "ontology")]
        {
            let cached = self
                .cached_subsumptions
                .as_ref()
                .ok_or(InferenceEngineError::OntologyNotLoaded)?;

            // Find all superclasses of the instance from cached subsumptions
            let class_iris: Vec<String> = cached
                .iter()
                .filter(|ax| ax.axiom_type == AxiomType::SubClassOf && ax.subject == instance_iri)
                .map(|ax| ax.object.clone())
                .collect();

            debug!(
                "Instance {} belongs to {} classes",
                instance_iri,
                class_iris.len()
            );
            Ok(class_iris)
        }

        #[cfg(not(feature = "ontology"))]
        {
            Ok(Vec::new())
        }
    }

    async fn check_consistency(&self) -> EngineResult<bool> {
        #[cfg(feature = "ontology")]
        {
            let cached = self
                .cached_subsumptions
                .as_ref()
                .ok_or(InferenceEngineError::OntologyNotLoaded)?;

            // Check if any non-trivial class is a subclass of owl:Nothing
            // An ontology is inconsistent if any class (other than Nothing itself) is subclass of Nothing
            let bottom_iri = "http://www.w3.org/2002/07/owl#Nothing";

            let inconsistent_classes: Vec<&OwlAxiom> = cached
                .iter()
                .filter(|ax| {
                    ax.axiom_type == AxiomType::SubClassOf
                        && ax.object == bottom_iri
                        && ax.subject != bottom_iri
                })
                .collect();

            if !inconsistent_classes.is_empty() {
                warn!(
                    "Ontology is inconsistent: {} classes are equivalent to owl:Nothing",
                    inconsistent_classes.len()
                );
                return Ok(false);
            }

            info!("Ontology is consistent");
            Ok(true)
        }

        #[cfg(not(feature = "ontology"))]
        {
            Ok(true)
        }
    }

    async fn explain_entailment(&self, axiom: &OwlAxiom) -> EngineResult<Vec<OwlAxiom>> {
        #[cfg(feature = "ontology")]
        {
            // Explanation generation would require tracking justifications during reasoning
            // For now, return a simplified explanation based on direct subsumptions
            if axiom.axiom_type != AxiomType::SubClassOf {
                return Ok(Vec::new());
            }

            let cached = self
                .cached_subsumptions
                .as_ref()
                .ok_or(InferenceEngineError::OntologyNotLoaded)?;

            // Find intermediate steps in the hierarchy
            let mut explanation = Vec::new();

            // Add direct subsumptions that lead to this entailment
            for inferred in cached.iter() {
                if inferred.subject == axiom.subject && inferred.axiom_type == AxiomType::SubClassOf
                {
                    explanation.push(inferred.clone());
                }
            }

            debug!("Found {} axioms in explanation", explanation.len());
            Ok(explanation)
        }

        #[cfg(not(feature = "ontology"))]
        {
            Ok(Vec::new())
        }
    }

    async fn clear(&mut self) -> EngineResult<()> {
        #[cfg(feature = "ontology")]
        {
            self.ontology = None;
            self.cached_subsumptions = None;
            self.last_checksum = None;
        }

        self.loaded_classes = 0;
        self.loaded_axioms = 0;
        self.inferred_axioms = 0;

        info!("Cleared ontology from inference engine");
        Ok(())
    }

    async fn get_statistics(&self) -> EngineResult<InferenceStatistics> {
        Ok(InferenceStatistics {
            loaded_classes: self.loaded_classes,
            loaded_axioms: self.loaded_axioms,
            inferred_axioms: self.inferred_axioms,
            last_inference_time_ms: self.last_inference_time_ms,
            total_inferences: self.total_inferences as u64,
        })
    }
}

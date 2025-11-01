// Constraint Translation System - Module Root
// Week 3 Deliverable: OWL Axiom → Physics Constraint Translation

pub mod physics_constraint;
pub mod axiom_mapper;
pub mod priority_resolver;
pub mod constraint_blender;
pub mod gpu_converter;
pub mod constraint_lod;

// Re-export main types
pub use physics_constraint::{
    PhysicsConstraint,
    PhysicsConstraintType,
    NodeId,
    PRIORITY_USER_DEFINED,
    PRIORITY_INFERRED,
    PRIORITY_ASSERTED,
    PRIORITY_DEFAULT,
};

pub use axiom_mapper::{
    AxiomMapper,
    AxiomType,
    OWLAxiom,
    TranslationConfig,
};

pub use priority_resolver::{
    PriorityResolver,
    NodePair,
    ConstraintGroup,
};

pub use constraint_blender::{
    ConstraintBlender,
    BlendingStrategy,
    BlenderConfig,
};

pub use gpu_converter::{
    ConstraintData,
    GPUConstraintBuffer,
    ConstraintStats,
    to_gpu_constraint_data,
    to_gpu_constraint_batch,
    gpu_constraint_kind,
};

pub use constraint_lod::{
    ConstraintLOD,
    LODLevel,
    LODConfig,
    LODStats,
};

/
/
/
/
/
/
/
/
/
/
pub struct ConstraintPipeline {
    mapper: AxiomMapper,
    resolver: PriorityResolver,
    blender: ConstraintBlender,
    lod: ConstraintLOD,
}

impl ConstraintPipeline {
    
    pub fn new() -> Self {
        Self {
            mapper: AxiomMapper::new(),
            resolver: PriorityResolver::new(),
            blender: ConstraintBlender::new(),
            lod: ConstraintLOD::new(),
        }
    }

    
    pub fn with_configs(
        translation_config: TranslationConfig,
        blender_config: BlenderConfig,
        lod_config: LODConfig,
    ) -> Self {
        Self {
            mapper: AxiomMapper::with_config(translation_config),
            resolver: PriorityResolver::new(),
            blender: ConstraintBlender::with_config(blender_config),
            lod: ConstraintLOD::with_config(lod_config),
        }
    }

    
    
    
    
    
    
    
    
    pub fn process(
        &mut self,
        axioms: &[OWLAxiom],
        zoom_level: f32,
    ) -> GPUConstraintBuffer {
        
        let constraints = self.mapper.translate_axioms(axioms);

        
        self.resolver.clear();
        self.resolver.add_constraints(constraints);
        let resolved = self.resolver.resolve();

        
        let blended: Vec<PhysicsConstraint> = self.resolver
            .get_groups()
            .iter()
            .filter_map(|group| {
                self.blender.blend_constraints(&group.constraints)
            })
            .collect();

        
        self.lod.set_constraints(blended);
        self.lod.update_zoom(zoom_level);
        let active = self.lod.get_active_constraints();

        
        let mut buffer = GPUConstraintBuffer::new(active.len());
        buffer.add_constraints(active).unwrap();

        buffer
    }

    
    pub fn update_frame_time(&mut self, frame_time_ms: f32) {
        self.lod.update_frame_time(frame_time_ms);
    }

    
    pub fn get_lod_stats(&self) -> LODStats {
        self.lod.get_stats()
    }

    
    pub fn get_constraint_stats(&self, buffer: &GPUConstraintBuffer) -> ConstraintStats {
        ConstraintStats::from_buffer(buffer)
    }

    
    pub fn get_lod_level(&self) -> LODLevel {
        self.lod.get_current_level()
    }
}

impl Default for ConstraintPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_pipeline() {
        let mut pipeline = ConstraintPipeline::new();

        let axioms = vec![
            OWLAxiom::asserted(AxiomType::SubClassOf {
                subclass: 1,
                superclass: 2,
            }),
            OWLAxiom::asserted(AxiomType::DisjointClasses {
                classes: vec![3, 4],
            }),
            OWLAxiom::inferred(AxiomType::SubClassOf {
                subclass: 5,
                superclass: 2,
            }),
        ];

        
        let buffer = pipeline.process(&axioms, 5.0);

        assert!(buffer.len() > 0);
        assert_eq!(pipeline.get_lod_level(), LODLevel::Close);
    }

    #[test]
    fn test_lod_reduction() {
        let mut pipeline = ConstraintPipeline::new();

        let axioms = vec![
            OWLAxiom::asserted(AxiomType::SubClassOf {
                subclass: 1,
                superclass: 2,
            }),
            OWLAxiom::asserted(AxiomType::DisjointClasses {
                classes: vec![3, 4],
            }),
        ];

        
        let buffer_far = pipeline.process(&axioms, 2000.0);
        assert_eq!(pipeline.get_lod_level(), LODLevel::Far);

        
        let buffer_close = pipeline.process(&axioms, 5.0);
        assert_eq!(pipeline.get_lod_level(), LODLevel::Close);

        
        assert!(buffer_far.len() <= buffer_close.len());
    }

    #[test]
    fn test_adaptive_lod() {
        let mut pipeline = ConstraintPipeline::new();

        let axioms = vec![
            OWLAxiom::asserted(AxiomType::SubClassOf {
                subclass: 1,
                superclass: 2,
            }),
        ];

        pipeline.process(&axioms, 5.0);

        
        pipeline.update_frame_time(30.0);

        let stats = pipeline.get_lod_stats();
        assert!(stats.frame_time_ms > stats.target_frame_time_ms);
    }
}

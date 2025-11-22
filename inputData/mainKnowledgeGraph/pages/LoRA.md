- ### OntologyBlock
  id:: lora-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: AI-0254
	- preferred-term:: LoRA
	- source-domain:: metaverse
	- status:: draft
	- definition:: A parameter-efficient fine-tuning method that freezes pre-trained weights and injects trainable low-rank decomposition matrices into each layer of the transformer, dramatically reducing trainable parameters whilst maintaining performance. LoRA represents weight updates as the product of two low-rank matrices.

## Academic Context

LoRA has become the most widely used and effective PEFT method for adapting large language models, offering superior efficiency and performance compared to earlier techniques like adapters.

**Primary Sources**:
- Hu et al., foundational LoRA paper
- Widely discussed in arXiv:2305.14314 (2023) - QLoRA paper

## Key Characteristics

- Freezes all pre-trained weights
- Adds low-rank decomposition matrices (A, B)
- Trainable parameters typically 0.1-1% of model
- Can merge with base weights for zero inference overhead
- Enables efficient multi-task deployment

## Technical Details

**Mathematical Formulation**:
```
W' = W₀ + ΔW = W₀ + BA

Where:
- W₀: Frozen pre-trained weights (d × d)
- B: Trainable matrix (d × r)
- A: Trainable matrix (r × d)
- r: Rank (typically 4-64)
- r << d
```

**Training**:
```
h = W₀x + BAx = W₀x + Δx
```

**Inference** (after merging):
```
W_merged = W₀ + BA
h = W_merged x  (no overhead)
```

## Usage in AI/ML

"LoRA is the most widely used and effective PEFT method for adapting large language models."

Applications:
- Fine-tuning large language models
- Multi-task model deployment
- Personalized model adaptation
- Domain-specific specialization
- Instruction tuning with limited resources

## Related Concepts

- **Parameter-Efficient Fine-Tuning (PEFT)**: Broader category
- **QLoRA**: Quantized variant for extreme efficiency
- **Adapter Modules**: Earlier PEFT approach
- **Low-Rank Decomposition**: Mathematical foundation
- **Matrix Factorization**: Core technique

## Key Advantages

**Efficiency**:
- 10,000× fewer parameters than full fine-tuning
- Reduced memory requirements
- Faster training

**Performance**:
- Matches or exceeds full fine-tuning
- No degradation on most tasks

**Deployment**:
- Zero inference overhead (after merging)
- Easy multi-task switching (before merging)
- Small storage per task

## Typical Hyperparameters

**Rank (r)**:
- Low complexity tasks: 4-8
- Medium complexity: 16-32
- High complexity: 64+

**Alpha (scaling factor)**: Often r or 2r

**Target Modules**:
- Attention weights (Q, K, V, O)
- Feed-forward layers
- All linear layers (maximum adaptation)

## Training Process

1. Freeze all pre-trained weights W₀
2. Initialise A (Gaussian), B (zeros)
3. Forward pass: h = W₀x + BAx
4. Compute loss and gradients
5. Update only A and B
6. Optionally merge: W' = W₀ + BA

## Comparison to Other PEFT Methods

**vs. Full Fine-Tuning**:
- 0.01% of parameters
- Comparable performance
- Much faster training

**vs. Adapters**:
- Lower parameters
- No inference overhead (when merged)
- Better performance

**vs. Prefix Tuning**:
- Different modification strategy
- Can merge weights
- Generally more efficient

## Implementation Considerations

**Memory Savings**:
- Only store gradients for A, B
- Can use smaller batch sizes
- Enables larger models on same hardware

**Multi-Task Deployment**:
- Store separate (A, B) per task
- Typically <10MB per task (vs. GB for full model)
- Fast task switching

## Historical Development

- 2021: LoRA introduced
- 2022: Rapid adoption in community
- 2023: QLoRA extends to extreme efficiency
- 2024+: Standard method for LLM fine-tuning
- 2025: Hybrid approaches combining LoRA variants

## Significance

LoRA revolutionised efficient fine-tuning by demonstrating that low-rank adaptations could match full fine-tuning performance whilst requiring minimal resources, democratising access to large model customisation.

## OWL Functional Syntax

```clojure
(Declaration (Class :LoRA))
(SubClassOf :LoRA :ParameterEfficientFineTuning)
(SubClassOf :LoRA
  (ObjectSomeValuesFrom :freezes :PreTrainedWeights))
(SubClassOf :LoRA
  (ObjectSomeValuesFrom :injects :LowRankDecompositionMatrices))
(SubClassOf :LoRA
  (ObjectSomeValuesFrom :represents :WeightUpdates))
(SubClassOf :LoRA
  (ObjectSomeValuesFrom :canMergeWith :BaseWeights))
(SubClassOf :LoRA
  (ObjectSomeValuesFrom :enables :ZeroInferenceOverhead))
(SubClassOf :LoRA
  (DataPropertyAssertion :hasParameterFraction "0.1-1%"))

(AnnotationAssertion rdfs:comment :LoRA
  "Parameter-efficient fine-tuning method injecting trainable low-rank decomposition matrices into transformer layers whilst freezing pre-trained weights"@en)
(AnnotationAssertion :hasAcademicSource :LoRA
  "Hu et al., foundational LoRA paper; discussed in QLoRA arXiv:2305.14314 (2023)")
```

## UK English Notes

- "Whilst maintaining" (British usage)
- "Optimisation" in related contexts
- "Parameterisation" (not "parameterization")

**Last Updated**: 2025-10-27
**Verification Status**: Verified against QLoRA paper (arXiv:2305.14314)
	- maturity:: draft
	- owl:class:: mv:LoRA
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- is-subclass-of:: [[ArtificialIntelligence]]
	- belongsToDomain:: [[MetaverseDomain]]

- ### OntologyBlock
  id:: 0377-fairness-metrics-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[PerformanceMetric]]
    - term-id:: AI-0377
    - preferred-term:: Fairness Metrics
    - source-domain:: ai
    - status:: approved
    - version:: 1.0
    - last-updated:: 2025-11-16

  - **Definition**
    - definition:: Fairness Metrics are quantitative measures and mathematical frameworks used to evaluate and ensure equitable treatment across different demographic groups in AI systems. These metrics provide objective, measurable criteria to assess whether an algorithmic system produces disparate impacts, maintains statistical parity, or achieves equalized odds across protected attributes such as race, gender, age, or disability status. Key fairness metrics include demographic parity (equal positive prediction rates across groups), equalized odds (equal true positive and false positive rates), equal opportunity (equal true positive rates), and predictive parity (equal precision across groups). The selection and application of fairness metrics depends on the specific context, stakeholder values, and regulatory requirements, as different metrics can conflict and no single metric satisfies all fairness criteria simultaneously. Implementation requires confusion matrix analysis, statistical testing, and careful consideration of base rate differences between groups, as formalized in IEEE P7003-2021 and NIST SP 1270 guidelines for algorithmic fairness assessment.
    - maturity:: mature
    - source:: [[IEEE P7003-2021]], [[ISO/IEC TR 24027]], [[NIST SP 1270]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: aigo:FairnessMetrics
    - owl:physicality:: VirtualEntity
    - owl:role:: Process
    - owl:inferred-class:: aigo:VirtualProcess
    - belongsToDomain:: [[AIEthicsDomain]]
    - implementedInLayer:: [[ConceptualLayer]]

  - #### Relationships
    id:: 0377-fairness-metrics-relationships
    - is-part-of:: [[Algorithmic Fairness]], [[AI Ethics]], [[Bias Detection]]
    - requires:: [[Confusion Matrix]], [[Statistical Testing]], [[Protected Attributes]]
    - enables:: [[Bias Mitigation]], [[Fairness Auditing]], [[Regulatory Compliance]]
    - related-to:: [[AI Safety Research]], [[Value Alignment]], [[AI Trustworthiness]], [[Algorithmic Accountability]]
    - measured-by:: [[Fairness Auditing Tools]]
    - depends-on:: [[IEEE P7003-2021]], [[ISO/IEC TR 24027]], [[NIST SP 1270]]

  - #### OWL Axioms
    id:: 0377-fairness-metrics-owl-axioms
    collapsed:: true
    - ```clojure
      (Declaration (Class :FairnessMetric))
(SubClassOf :FairnessMetric :EvaluationMetric)
(SubClassOf :FairnessMetric :EthicalAIComponent)

(AnnotationAssertion rdfs:label :FairnessMetric
  "Fairness Metric"@en)
(AnnotationAssertion rdfs:comment :FairnessMetric
  "Quantitative measures for assessing algorithmic fairness across protected groups, including demographic parity, equalized odds, and equality of opportunity."@en)
(AnnotationAssertion :dcterms:source :FairnessMetric
  "IEEE P7003-2021, ISO/IEC TR 24027:2021, NIST SP 1270")

;; Object Properties
(Declaration (ObjectProperty :measures))
(ObjectPropertyDomain :measures :FairnessMetric)
(ObjectPropertyRange :measures :AlgorithmicFairness)

(Declaration (ObjectProperty :detectsBias))
(ObjectPropertyDomain :detectsBias :FairnessMetric)
(ObjectPropertyRange :detectsBias :ProtectedAttribute)

(Declaration (ObjectProperty :appliesTo))
(ObjectPropertyDomain :appliesTo :FairnessMetric)
(ObjectPropertyRange :appliesTo :AIModel)

(Declaration (ObjectProperty :requiresConfusionMatrix))
(SubObjectPropertyOf :requiresConfusionMatrix :dependsOn)

;; Data Properties
(Declaration (DataProperty :hasValueRange))
(DataPropertyAssertion :hasValueRange :FairnessMetric
  "[0,1] for most metrics"^^xsd:string)

(Declaration (DataProperty :hasThreshold))
(DataPropertyDomain :hasThreshold :FairnessMetric)
(DataPropertyRange :hasThreshold xsd:decimal)

(Declaration (DataProperty :requiresGroundTruth))
(DataPropertyAssertion :requiresGroundTruth :FairnessMetric
  "true"^^xsd:boolean)

;; Subclass Definitions
(Declaration (Class :DemographicParity))
(SubClassOf :DemographicParity :FairnessMetric)
(AnnotationAssertion rdfs:comment :DemographicParity
  "P(Ŷ=1|A=0) = P(Ŷ=1|A=1) where A is protected attribute and Ŷ is prediction"@en)

(Declaration (Class :EqualizedOdds))
(SubClassOf :EqualizedOdds :FairnessMetric)
(AnnotationAssertion rdfs:comment :EqualizedOdds
  "P(Ŷ=1|A=0,Y=y) = P(Ŷ=1|A=1,Y=y) for y ∈ {0,1}"@en)

(Declaration (Class :EqualOpportunity))
(SubClassOf :EqualOpportunity :FairnessMetric)
(AnnotationAssertion rdfs:comment :EqualOpportunity
  "P(Ŷ=1|A=0,Y=1) = P(Ŷ=1|A=1,Y=1) - equal true positive rates"@en)

(Declaration (Class :PredictiveParity))
(SubClassOf :PredictiveParity :FairnessMetric)
(AnnotationAssertion rdfs:comment :PredictiveParity
  "P(Y=1|Ŷ=1,A=0) = P(Y=1|Ŷ=1,A=1) - equal precision across groups"@en)

;; Disjoint Classes
(DisjointClasses :DemographicParity :EqualizedOdds :EqualOpportunity)

;; Domain Constraints
(SubClassOf :FairnessMetric
  (ObjectSomeValuesFrom :measures :AlgorithmicFairness))
(SubClassOf :FairnessMetric
  (ObjectSomeValuesFrom :detectsBias :ProtectedAttribute))
(SubClassOf :FairnessMetric
  (DataSomeValuesFrom :hasThreshold xsd:decimal))
      ```

- ## About Fairness Metrics
  id:: 0377-fairness-metrics-about

  Fairness Metrics provide quantitative frameworks for evaluating algorithmic equity across demographic groups. These mathematical measures are essential for detecting and mitigating bias in AI systems, ensuring compliance with regulatory frameworks such as the [[EU AI Act]], [[IEEE P7003-2021]], and [[NIST AI Risk Management Framework]].

  ### Core Fairness Metrics

  - **[[Demographic Parity]]**: Equal positive prediction rates across groups: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
  - **[[Equalized Odds]]**: Equal true positive and false positive rates: P(Ŷ=1|A=0,Y=y) = P(Ŷ=1|A=1,Y=y)
  - **[[Equal Opportunity]]**: Equal true positive rates across groups
  - **[[Predictive Parity]]**: Equal precision across demographic groups
  - **[[Calibration]]**: Predicted probabilities match actual outcomes across groups

  ### Application Domains

  - **Criminal Justice**: Risk assessment tools, recidivism prediction
  - **Financial Services**: Credit scoring, loan approval systems
  - **Healthcare**: Diagnosis algorithms, treatment recommendations
  - **Employment**: Hiring algorithms, performance evaluation
  - **Education**: Admissions systems, grading automation

  ### Implementation Challenges

  Fairness metrics often conflict with one another - achieving one form of fairness may preclude others. The **impossibility theorem** (Kleinberg et al., 2017) demonstrates that demographic parity, equalized odds, and predictive parity cannot be simultaneously satisfied except in trivial cases. This requires careful stakeholder engagement to determine which fairness criteria align with societal values and regulatory requirements.

## Current Landscape (2024-2025)

- ### Industry Adoption and Implementations
  - Many organisations now embed fairness metrics into their AI governance strategies, using them to comply with regulations, build trust, and protect brand reputation
  - Notable platforms include Iterate.ai, Shelf.io, and IEEE's machine learning fairness standards
  - **Technical Capabilities**:
    - Fairness metrics can identify and quantify bias across groups, but they cannot eliminate all forms of unfairness due to inherent trade-offs between different fairness definitions
    - Metrics are most effective when combined with transparency, explainability, and continuous monitoring
  - **Standards and Frameworks**:
    - [[IEEE 3198-2025]]: Comprehensive standard for evaluating machine learning fairness, specifying methods, metrics, and test cases
    - [[EU AI Act]]: European regulatory framework for AI systems
    - [[UK AI Regulation]]: Regulatory guidance on automated decision-making

- ### UK Context
  - The [[UK AI Regulation]] emphasises fairness assessment through the [[ICO AI Auditing Framework]] and [[BSI ADS standards]]. UK organisations implementing fairness metrics include:
    - **[[NHS AI Lab]]**: Fairness testing for medical diagnosis algorithms
    - **[[Financial Conduct Authority]]**: Credit decisioning fairness requirements
    - **[[University of Oxford]]**: Research on fairness metric selection and tradeoffs
    - **Manchester AI Ethics Hub**: Regional fairness auditing initiatives

  - **North England Regional Context**:
    - UK-based companies and public sector bodies increasingly adopt fairness metrics, particularly in sectors such as finance, healthcare, and public services
    - In North England, cities like Manchester, Leeds, Newcastle, and Sheffield host innovation hubs and research centres focused on ethical AI, including fairness and bias mitigation

- ### 2024-2025 Developments
  - Integration with [[Large Language Models]] fairness testing
  - [[Intersectional Fairness]] metrics accounting for multiple protected attributes
  - [[Causal Fairness]] frameworks using causal inference
  - [[Dynamic Fairness]] metrics for evolving populations
  - [[Differential Privacy]] integration for fairness with privacy guarantees

## Research & Literature

- Key academic papers and sources
  - Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning: Limitations and Opportunities. fairmlbook.org. https://fairmlbook.org/
  - Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A Survey on Bias and Fairness in Machine Learning. ACM Computing Surveys, 54(6), 1–37. https://doi.org/10.1145/3457607
  - Mitchell, S., Potash, E., Barocas, S., D’Amour, A., & Lum, K. (2021). Algorithmic Fairness: Choices, Assumptions, and Definitions. Annual Review of Statistics and Its Application, 8, 141–163. https://doi.org/10.1146/annurev-statistics-042720-020326
- Ongoing research directions
  - Contextual fairness standards tailored to specific domains (e.g., healthcare, criminal justice)
  - Global and cultural variations in fairness perceptions and requirements
  - Integration of fairness metrics with explainable AI and human-in-the-loop systems

## UK Policy and Research

- ### British Contributions and Implementations
  - The UK has been active in developing regulatory frameworks and best practices for AI fairness, with contributions from academic institutions, industry, and government bodies
  - The [[Centre for Data Ethics and Innovation]] (CDEI) and the [[Alan Turing Institute]] play key roles in shaping national policy and research

- ### North England Innovation Hubs
  - Manchester, Leeds, Newcastle, and Sheffield are home to several universities and research centres engaged in AI ethics and fairness
  - **[[University of Manchester]]**: AI for Social Good initiative
  - **[[Newcastle University]]**: Centre for Social Justice and Community Action
  - **[[University of Leeds]]**: Institute for Data Analytics with fairness focus
  - **[[University of Sheffield]]**: Machine Learning Research Group

- ### Regional Case Studies
  - Local authorities in North England have piloted AI systems for social services, using fairness metrics to ensure equitable outcomes for diverse communities
  - **Leeds Housing Allocation**: Recent project used fairness metrics to evaluate an AI-driven housing allocation system, highlighting the importance of context-specific fairness standards
  - **Manchester NHS Collaboration**: Fairness auditing for diagnostic algorithms
  - **Newcastle Social Services**: Algorithmic accountability initiatives

## Future Directions

- Emerging trends and developments
  - Increasing focus on domain-specific fairness standards and global harmonisation of regulatory approaches
  - Growing interest in the role of cultural and societal factors in shaping fairness perceptions
- Anticipated challenges
  - Balancing competing fairness criteria and managing trade-offs in real-world applications
  - Ensuring that fairness metrics are accessible and usable for non-expert stakeholders
- Research priorities
  - Developing more robust and context-aware fairness metrics
  - Exploring the intersection of fairness, transparency, and accountability in AI systems
  - Investigating the long-term societal impacts of fairness-aware AI

## References

1. Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning: Limitations and Opportunities. fairmlbook.org. https://fairmlbook.org/
2. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A Survey on Bias and Fairness in Machine Learning. ACM Computing Surveys, 54(6), 1–37. https://doi.org/10.1145/3457607
3. Mitchell, S., Potash, E., Barocas, S., D’Amour, A., & Lum, K. (2021). Algorithmic Fairness: Choices, Assumptions, and Definitions. Annual Review of Statistics and Its Application, 8, 141–163. https://doi.org/10.1146/annurev-statistics-042720-020326
4. IEEE 3198-2025. IEEE Standard for Machine Learning Fairness. IEEE. https://standards.ieee.org/ieee/3198/11068/
5. Centre for Data Ethics and Innovation (CDEI). (2023). AI Barometer Report. https://www.gov.uk/government/organisations/centre-for-data-ethics-and-innovation
6. Alan Turing Institute. (2023). Fairness in AI. https://www.turing.ac.uk/research/research-programmes/fairness-ai
7. University of Manchester. (2023). AI for Social Good. https://www.manchester.ac.uk/research/themes/ai-for-social-good/
8. Newcastle University. (2023). Centre for Social Justice and Community Action. https://www.ncl.ac.uk/csjca/

## Metadata

- **Document Type**: Knowledge Graph Entry - [[AI Ethics]] Domain
- **Primary Category**: [[Algorithmic Fairness]], [[AI Ethics]]
- **Secondary Categories**: [[Bias Detection]], [[AI Governance]]
- **Term ID**: AI-0377
- **Status**: Approved
- **Version**: 1.0
- **Last Updated**: 2025-11-16
- **Review Status**: Comprehensive editorial review completed
- **Verification**: Academic sources verified, citations cross-referenced
- **Regional Context**: UK/Northern England where applicable
- **Quality Score**: 0.95 (post-processing)
- **Authority Score**: 0.95 (IEEE P7003-2021, NIST SP 1270)
- **Completeness**: High - Comprehensive coverage with academic references and OWL axioms
- **Link Density**: High - Extensive [[wiki-links]] to related concepts

---

**Processing Notes**:
- Merged content from AI-0377-fairness-metrics.md (duplicate)
- Enhanced OntologyBlock with complete definition and relationships
- Added detailed OWL axioms for semantic web integration
- Expanded UK regional context and case studies
- Integrated 2024-2025 developments
- Fixed Logseq formatting inconsistencies
- Cross-referenced with regulatory frameworks

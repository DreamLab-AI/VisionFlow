- ### OntologyBlock
  id:: 0386-fairness-auditing-tools-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - term-id:: AI-0386
    - preferred-term:: Fairness Auditing Tools
    - source-domain:: ai
    - status:: approved
    - version:: 1.0
    - last-updated:: 2025-11-16

  - **Definition**
    - definition:: Fairness Auditing Tools are software libraries, platforms, and frameworks designed to detect, measure, and mitigate algorithmic bias in AI systems through automated analysis, visualization, and intervention capabilities. Leading open-source tools include Fairlearn (Microsoft, MIT license) providing fairness metrics and mitigation algorithms for Python with scikit-learn integration, AIF360 (IBM, Apache-2.0 license) offering comprehensive bias detection and mitigation across the ML pipeline with 70+ fairness metrics, What-If Tool (Google, Apache-2.0) providing interactive visual interfaces for TensorFlow model exploration and counterfactual analysis, Aequitas (University of Chicago, MIT license) focusing on fairness auditing for criminal justice and policy applications, and FairTest (Columbia University, MIT license) enabling statistical fairness testing with association discovery. These tools implement fairness metrics including demographic parity, equalized odds, and predictive parity, provide visualizations such as fairness dashboards, confusion matrices disaggregated by group, and disparity charts, and support mitigation techniques including reweighting, threshold optimization, and adversarial debiasing. Adoption best practices include multi-tool validation to cross-verify findings, integration into CI/CD pipelines for continuous fairness monitoring, documentation of fairness decisions and tradeoffs, and stakeholder engagement in selecting appropriate fairness metrics. These tools operationalize fairness requirements from standards including IEEE P7003-2021, ISO/IEC TR 24027:2021, and the EU AI Act Article 10 on data governance and bias mitigation.
    - maturity:: mature
    - source:: [[Fairlearn]], [[AIF360]], [[IEEE P7003-2021]], [[ISO/IEC TR 24027]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: ai:FairnessAuditingTools
    - owl:physicality:: VirtualEntity
    - owl:role:: Process
    - owl:inferred-class:: ai:AuditingInfrastructure
    - belongsToDomain:: [[AIDomain]], [[AIEthicsDomain]]
    - implementedInLayer:: [[ConceptualLayer]]

  - #### Relationships
    id:: 0386-fairness-auditing-tools-relationships
    - is-part-of:: [[AI Governance]], [[Algorithmic Accountability]], [[Bias Mitigation]]
    - implements:: [[Fairness Metrics]], [[Bias Detection Methods]]
    - enables:: [[Fairness Auditing]], [[Regulatory Compliance]], [[Algorithmic Transparency]]
    - requires:: [[Statistical Testing]], [[Data Quality]], [[Protected Attributes]]
    - integrates-with:: [[scikit-learn]], [[TensorFlow]], [[PyTorch]], [[CI/CD Pipelines]]
    - related-to:: [[AI Safety Research]], [[Value Alignment]], [[Constitutional AI]], [[AI Trustworthiness]]
    - depends-on:: [[IEEE P7003-2021]], [[ISO/IEC TR 24027]], [[EU AI Act]]

  - #### OWL Axioms
    id:: 0386-fairness-auditing-tools-owl-axioms
    collapsed:: true
    - ```clojure
      (Declaration (Class :FairnessAuditingTool))
(SubClassOf :FairnessAuditingTool :SoftwareTool)
(SubClassOf :FairnessAuditingTool :EthicalAIInfrastructure)
(SubClassOf :FairnessAuditingTool :AuditingFramework)

(AnnotationAssertion rdfs:label :FairnessAuditingTool
  "Fairness Auditing Tool"@en)
(AnnotationAssertion rdfs:comment :FairnessAuditingTool
  "Software libraries and platforms for detecting, measuring, and mitigating algorithmic bias, including Fairlearn, AIF360, What-If Tool, Aequitas, and FairTest."@en)

;; Object Properties
(Declaration (ObjectProperty :implements))
(ObjectPropertyDomain :implements :FairnessAuditingTool)
(ObjectPropertyRange :implements :FairnessMetric)

(Declaration (ObjectProperty :providesVisualization))
(ObjectPropertyDomain :providesVisualization :FairnessAuditingTool)
(ObjectPropertyRange :providesVisualization :VisualizationType)

(Declaration (ObjectProperty :supportsMitigation))
(ObjectPropertyDomain :supportsMitigation :FairnessAuditingTool)
(ObjectPropertyRange :supportsMitigation :BiasMitigationTechnique)

(Declaration (ObjectProperty :integratesWith))
(ObjectPropertyDomain :integratesWith :FairnessAuditingTool)
(ObjectPropertyRange :integratesWith :MLFramework)

;; Data Properties
(Declaration (DataProperty :hasLicense))
(DataPropertyDomain :hasLicense :FairnessAuditingTool)
(DataPropertyRange :hasLicense xsd:string)

(Declaration (DataProperty :supportsProgrammingLanguage))
(DataPropertyDomain :supportsProgrammingLanguage :FairnessAuditingTool)
(DataPropertyRange :supportsProgrammingLanguage xsd:string)

(Declaration (DataProperty :hasRepositoryURL))
(DataPropertyDomain :hasRepositoryURL :FairnessAuditingTool)
(DataPropertyRange :hasRepositoryURL xsd:anyURI)

(Declaration (DataProperty :supportsMetricCount))
(DataPropertyDomain :supportsMetricCount :FairnessAuditingTool)
(DataPropertyRange :supportsMetricCount xsd:integer)

;; Tool Subclasses
(Declaration (Class :Fairlearn))
(SubClassOf :Fairlearn :FairnessAuditingTool)
(DataPropertyAssertion :hasLicense :Fairlearn "MIT"^^xsd:string)
(DataPropertyAssertion :supportsProgrammingLanguage :Fairlearn "Python"^^xsd:string)
(DataPropertyAssertion :hasRepositoryURL :Fairlearn
  "https://github.com/fairlearn/fairlearn"^^xsd:anyURI)
(AnnotationAssertion rdfs:comment :Fairlearn
  "Microsoft-developed fairness toolkit with scikit-learn integration"@en)

(Declaration (Class :AIF360))
(SubClassOf :AIF360 :FairnessAuditingTool)
(DataPropertyAssertion :hasLicense :AIF360 "Apache-2.0"^^xsd:string)
(DataPropertyAssertion :supportsProgrammingLanguage :AIF360 "Python"^^xsd:string)
(DataPropertyAssertion :supportsMetricCount :AIF360 "70"^^xsd:integer)
(AnnotationAssertion rdfs:comment :AIF360
  "IBM comprehensive bias detection framework with 70+ fairness metrics"@en)

(Declaration (Class :WhatIfTool))
(SubClassOf :WhatIfTool :FairnessAuditingTool)
(DataPropertyAssertion :hasLicense :WhatIfTool "Apache-2.0"^^xsd:string)
(AnnotationAssertion rdfs:comment :WhatIfTool
  "Google interactive visual interface for TensorFlow model exploration and counterfactual analysis"@en)

(Declaration (Class :Aequitas))
(SubClassOf :Aequitas :FairnessAuditingTool)
(DataPropertyAssertion :hasLicense :Aequitas "MIT"^^xsd:string)
(AnnotationAssertion rdfs:comment :Aequitas
  "University of Chicago tool focused on criminal justice fairness auditing"@en)

(Declaration (Class :FairTest))
(SubClassOf :FairTest :FairnessAuditingTool)
(DataPropertyAssertion :hasLicense :FairTest "MIT"^^xsd:string)
(AnnotationAssertion rdfs:comment :FairTest
  "Columbia University statistical fairness testing framework with association discovery"@en)

;; Domain Constraints
(SubClassOf :FairnessAuditingTool
  (ObjectSomeValuesFrom :implements :FairnessMetric))
(SubClassOf :FairnessAuditingTool
  (ObjectSomeValuesFrom :supportsMitigation :BiasMitigationTechnique))
(SubClassOf :FairnessAuditingTool
  (DataSomeValuesFrom :hasLicense xsd:string))
      ```

- ## About Fairness Auditing Tools
  id:: 0386-fairness-auditing-tools-about

  Fairness Auditing Tools provide essential infrastructure for implementing [[Algorithmic Accountability]] and [[Bias Mitigation]] in production AI systems. These tools operationalize fairness requirements from regulatory frameworks including the [[EU AI Act]], [[IEEE P7003-2021]], and [[ISO/IEC TR 24027]].

  ### Core Tools and Capabilities

  #### [[Fairlearn]] (Microsoft, MIT Licence)

  - **Focus**: Integration with scikit-learn ecosystem
  - **Capabilities**: Fairness metrics computation, threshold optimization, reweighting algorithms
  - **Use Cases**: Classification fairness, regression fairness, model selection
  - **Visualization**: Interactive fairness dashboard
  - **Repository**: https://github.com/fairlearn/fairlearn

  #### [[AIF360]] (IBM, Apache-2.0 Licence)

  - **Focus**: Comprehensive bias detection across ML pipeline
  - **Metrics**: 70+ fairness metrics covering pre-processing, in-processing, post-processing
  - **Algorithms**: Reweighting, disparate impact remover, adversarial debiasing, calibrated equalized odds
  - **Datasets**: 10+ benchmark datasets for fairness research
  - **Use Cases**: Credit scoring, hiring, criminal justice risk assessment

  #### [[What-If Tool]] (Google, Apache-2.0 Licence)

  - **Focus**: Interactive visual exploration and counterfactual analysis
  - **Integration**: TensorFlow, TensorFlow Serving
  - **Features**: Feature importance analysis, partial dependence plots, individual fairness testing
  - **Use Cases**: Model debugging, fairness exploration, stakeholder communication

  #### [[Aequitas]] (University of Chicago, MIT Licence)

  - **Focus**: Policy and criminal justice fairness auditing
  - **Features**: Bias report generation, disparity visualization, group-based fairness analysis
  - **Standards**: COMPAS compliance, ProPublica methodology
  - **Use Cases**: Criminal justice, policy evaluation, public sector AI

  #### [[FairTest]] (Columbia University, MIT Licence)

  - **Focus**: Statistical association discovery and fairness testing
  - **Method**: Bug-finding approach to fairness violations
  - **Features**: Association rule mining, context-aware testing
  - **Use Cases**: Exploratory fairness analysis, hypothesis generation

  ### Best Practices for Tool Adoption

  #### Multi-Tool Validation

  Cross-verify fairness findings using multiple tools to ensure robustness:
  - **[[Fairlearn]]** for quick initial assessment
  - **[[AIF360]]** for comprehensive multi-stage analysis
  - **[[Aequitas]]** for regulatory compliance verification
  - **[[What-If Tool]]** for stakeholder communication

  #### CI/CD Integration

  Embed fairness testing in continuous integration pipelines:
  ```python
  # Example GitHub Actions workflow
  - name: Fairness Audit
    run: |
      python -m pytest tests/fairness_tests.py
      fairlearn-audit --model model.pkl --threshold 0.8
      aif360-report --output fairness_report.html
  ```

  #### Documentation and Governance

  - Document fairness metric selection rationale
  - Track fairness-accuracy tradeoff decisions
  - Maintain audit trails for regulatory compliance
  - Engage stakeholders in metric selection

  ### Implementation Patterns

  #### Pre-Processing Fairness

  Modify training data to reduce bias:
  - **Reweighting**: Adjust sample weights to balance group representation
  - **Disparate Impact Remover**: Transform features to remove group associations
  - **Fair Sampling**: Stratified sampling to ensure group parity

  #### In-Processing Fairness

  Modify learning algorithms to enforce fairness constraints:
  - **Adversarial Debiasing**: Train model to prevent group prediction
  - **Fairness-Constrained Optimization**: Add fairness penalties to loss function
  - **Fair Representation Learning**: Learn group-invariant features

  #### Post-Processing Fairness

  Adjust model outputs to improve fairness:
  - **Threshold Optimization**: Select group-specific decision thresholds
  - **Calibrated Equalized Odds**: Post-hoc adjustment for fairness criteria
  - **Reject Option Classification**: Defer uncertain predictions near threshold

  ### UK Context and Implementation

  #### UK Regulatory Framework

  - **[[ICO AI Auditing Framework]]**: Fairness testing requirements for high-risk AI
  - **[[Financial Conduct Authority]]**: Credit decisioning fairness standards
  - **[[NHS AI Lab]]**: Healthcare algorithm fairness guidelines
  - **[[Centre for Data Ethics and Innovation]]**: Fairness best practices

  #### UK Organisations Using Fairness Tools

  - **[[University of Oxford]]**: Research on fairness metric selection
  - **[[Imperial College London]]**: Healthcare AI fairness assessment
  - **[[University of Cambridge]]**: Causal fairness frameworks
  - **[[Alan Turing Institute]]**: Fairness tool development and validation
  - **Manchester AI Ethics Hub**: Regional fairness auditing initiatives
  - **Leeds Digital Trust Centre**: Financial services fairness testing

  ### 2024-2025 Developments

  - **[[Large Language Model]] Fairness**: Tools for LLM bias detection (e.g., [[HELM]], [[BIG-bench]])
  - **[[Causal Fairness]]**: Integration of causal inference into fairness assessment
  - **[[Intersectional Fairness]]**: Multi-attribute fairness analysis
  - **[[Differential Privacy]]** Integration: Privacy-preserving fairness testing
  - **[[Explainable AI]]** Integration: Linking fairness violations to feature importance
  - **Regulatory Automation**: Direct compliance reporting for [[EU AI Act]] Article 10

  ### Research Directions

  - **Fairness-Efficiency Tradeoffs**: Optimizing fairness under computational constraints
  - **Dynamic Fairness**: Monitoring fairness in production systems
  - **Multi-Objective Fairness**: Balancing competing fairness criteria
  - **Causal Fairness Metrics**: Fairness definitions based on causal graphs
  - **Federated Fairness Auditing**: Privacy-preserving distributed fairness testing


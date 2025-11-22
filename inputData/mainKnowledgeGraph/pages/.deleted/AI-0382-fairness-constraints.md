- ### OntologyBlock
    - term-id:: AI-0382
    - preferred-term:: Fairness Constraints
    - ontology:: true
    - version:: 1.0

## Fairness Constraints

Fairness Constraints refers to fairness constraints are mathematical formalizations of equitable treatment in ai systems, expressed as conditions that predictions must satisfy relative to protected attributes. these constraints are categorized into three fundamental types based on independence criteria: independence (demographic parity) requires predictions to be independent of protected attributes (ŷ ⊥ a), meaning p(ŷ|a=0) = p(ŷ|a=1); separation (equalized odds) requires predictions to be independent of protected attributes conditional on true labels (ŷ ⊥ a | y), ensuring equal true positive and false positive rates across groups; and sufficiency (predictive parity) requires true labels to be independent of protected attributes conditional on predictions (y ⊥ a | ŷ), ensuring equal precision and calibration across groups. these constraints formalize fairness concepts like demographic parity, equalized odds, equal opportunity (separation for positive class only), and calibration into optimization problems during model training. however, impossibility theorems (chouldechova 2017, kleinberg et al. 2017) prove that when base rates differ between groups, certain combinations of fairness constraints cannot be simultaneously satisfied, necessitating context-dependent tradeoffs. implementation typically involves constrained optimization with lagrange multipliers, where accuracy loss is balanced against fairness violations through tunable regularization parameters, as formalized in foundational research by hardt et al. (2016) and barocas et al. (2019).

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: 0382-fairness-constraints-about
- **Collapsed**: true
- **Public Access**: true
- **Source Domain**: ai
- **Status**: in-progress
- **Last Updated**: 2025-10-29
- **Maturity**: mature
- **Source**: [[Hardt et al. (2016)]], [[Barocas et al. (2019)]], [[Chouldechova (2017)]]
- **Authority Score**: 0.95
- **Owl:Class**: aigo:FairnessConstraints
- **Owl:Physicality**: VirtualEntity
- **Owl:Role**: Process
- **Owl:Inferred Class**: aigo:VirtualProcess
- **Belongstodomain**: [[AIEthicsDomain]]
- **Implementedinlayer**: [[ConceptualLayer]]

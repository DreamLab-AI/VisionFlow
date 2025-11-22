- ### OntologyBlock
    - term-id:: AI-0385
    - preferred-term:: Fairness Accuracy Tradeoffs
    - ontology:: true
    - version:: 1.0

## Fairness Accuracy Tradeoffs

Fairness Accuracy Tradeoffs refers to fairness accuracy tradeoffs represent the fundamental tension in machine learning between maximizing predictive accuracy and satisfying fairness constraints, characterized by the pareto frontier of achievable (accuracy, fairness) pairs where improving one objective typically requires sacrificing the other. this tradeoff arises because fairness constraints restrict the hypothesis space of permissible models, excluding solutions that achieve maximum accuracy through reliance on correlations between protected attributes and outcomes, even when those correlations reflect genuine statistical relationships in the data. the magnitude of accuracy cost depends on several factors: the strength of correlation between protected attributes and outcomes, which fairness constraint is enforced (with independence constraints typically more costly than separation constraints), the flexibility of the model class, and base rate differences between groups. implementation typically involves multi-objective optimization with a tradeoff parameter λ balancing accuracy loss l_accuracy and fairness violation l_fairness in the combined objective l = l_accuracy + λ·l_fairness, where varying λ traces out the pareto frontier. while some contexts permit minimal accuracy costs for fairness improvements, others involve substantial tradeoffs requiring normative judgment about acceptable accuracy sacrifices for fairness gains. research by corbett-davies et al. (2017) demonstrates that fairness constraints can sometimes improve accuracy for disadvantaged groups while reducing overall accuracy, and that the tradeoff is context-dependent based on deployment objectives and stakeholder priorities.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: 0385-fairness-accuracy-tradeoffs-about
- **Collapsed**: true
- **Public Access**: true
- **Source Domain**: ai
- **Status**: in-progress
- **Last Updated**: 2025-10-29
- **Maturity**: mature
- **Source**: [[Corbett-Davies et al. (2017)]], [[Kleinberg et al. (2017)]], [[Chouldechova (2017)]]
- **Authority Score**: 0.95
- **Owl:Class**: aigo:FairnessAccuracyTradeoffs
- **Owl:Physicality**: VirtualEntity
- **Owl:Role**: Process
- **Owl:Inferred Class**: aigo:VirtualProcess
- **Belongstodomain**: [[AIEthicsDomain]]
- **Implementedinlayer**: [[ConceptualLayer]]

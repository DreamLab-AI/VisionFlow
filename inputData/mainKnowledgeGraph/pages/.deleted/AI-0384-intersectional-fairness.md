- ### OntologyBlock
    - term-id:: AI-0384
    - preferred-term:: Intersectional Fairness
    - ontology:: true
    - version:: 1.0

## Intersectional Fairness

Intersectional Fairness refers to intersectional fairness is an approach to algorithmic fairness that accounts for overlapping and interacting protected attributes, recognizing that individuals with multiple marginalized identities may experience unique forms of discrimination not captured by analyzing single attributes in isolation. rooted in intersectionality theory from critical race and feminist scholarship (crenshaw 1989), this framework acknowledges that the experiences of, for example, black women cannot be understood simply as the combination of being black and being a woman, but involve distinct discriminatory patterns at the intersection of race and gender. in ai systems, intersectional fairness requires evaluating bias and fairness metrics across intersectional subgroups defined by specific combinations of protected attribute values, where the number of subgroups equals the product of attribute cardinalities (e.g., 2 genders × 4 race categories × 3 age brackets = 24 subgroups). this analysis often reveals intersectional disparities where subgroups experience worse outcomes than predicted by single-attribute analysis, particularly affecting individuals with multiple marginalized identities. implementation challenges include exponential growth of subgroups with additional attributes, sample size limitations for rare intersectional groups, and computational complexity of enforcing fairness across all subgroups simultaneously. intersectional fairness auditing is increasingly required by comprehensive ai governance frameworks and documented in research by buolamwini and gebru (2018) on gender-race bias in facial recognition.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: 0384-intersectional-fairness-about
- **Collapsed**: true
- **Public Access**: true
- **Source Domain**: ai
- **Status**: in-progress
- **Last Updated**: 2025-10-29
- **Maturity**: mature
- **Source**: [[Crenshaw (1989)]], [[Buolamwini and Gebru (2018)]], [[IEEE P7003-2021]]
- **Authority Score**: 0.95
- **Owl:Class**: aigo:IntersectionalFairness
- **Owl:Physicality**: VirtualEntity
- **Owl:Role**: Process
- **Owl:Inferred Class**: aigo:VirtualProcess
- **Belongstodomain**: [[AIEthicsDomain]]
- **Implementedinlayer**: [[ConceptualLayer]]

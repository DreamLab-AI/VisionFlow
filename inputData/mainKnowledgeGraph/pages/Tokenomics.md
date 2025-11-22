- ### OntologyBlock
  id:: tokenomics-ontology
  collapsed:: true

  - **Identification**
    - domain-prefix:: BC
    - sequence-number:: 0576
    - filename-history:: ["BC-0576-tokenomics.md"]
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[BlockchainTechnology]]
    - term-id:: BC-0576
    - preferred-term:: Tokenomics
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-11-13

  - **Definition**
    - definition:: Economic model of token systems.
    - maturity:: established
    - source:: Chimera Prime Research
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:Tokenomics
    - belongsToDomain:: [[Blockchain]]

  - #### CrossDomainBridges
    - bridges-from:: [[PlayToEarnP2e]] via has-part

  - #### Relationships

  - #### OWL Axioms
    - ```clojure
      ; Class Declaration
      (Declaration (Class :Tokenomics))
      
      ; Annotations
      (AnnotationAssertion rdfs:label :Tokenomics "Tokenomics"@en)
      (AnnotationAssertion rdfs:comment :Tokenomics
        "Economic model of token systems."@en)
      ```

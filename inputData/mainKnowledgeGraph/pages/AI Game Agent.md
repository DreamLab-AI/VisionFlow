- ### OntologyBlock
  id:: ai-game-agent-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: AI-0800
    - preferred-term:: AI Game Agent
    - source-domain:: ai
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-11-05

  - **Definition**
    - definition:: An intelligent autonomous entity within a video game or virtual environment that exhibits goal-directed behavior, adapts to player actions, and creates engaging interactive experiences through AI techniques including behavior trees, reinforcement learning, and procedural generation.
    - maturity:: mature
    - source:: [[Game AI Pro]], [[Unity ML-Agents]], [[IEEE CIG]]
    - authority-score:: 0.92

  - **Semantic Classification**
    - owl:class:: ai:AIGameAgent
    - owl:physicality:: VirtualEntity
    - owl:role:: Agent
    - owl:inferred-class:: ai:VirtualAgent
    - belongsToDomain:: [[AI-GroundedDomain]], [[InteractionDomain]], [[CreativeMediaDomain]]

  - #### OWL Restrictions
    - requires some GameState
    - requires some GameEngine
    - has-part some DecisionEngine
    - requires some NavigationMesh
    - has-part some BehaviorTree
    - enables some AdaptiveChallenge
    - implements some ProceduralBehavior
    - enables some DynamicGameplay
    - enables some PlayerEngagement
    - implements some ReinforcementLearning
    - implements some AdaptiveDifficulty
    - has-part some PathfindingSystem
    - has-part some StateMachine
    
    
    - enables some EmergentBehavior

  - #### CrossDomainBridges
    - bridges-to:: [[IntelligentVirtualEntity]] via is-subclass-of
    - bridges-to:: [[NavigationMesh]] via requires
    - bridges-to:: [[PathfindingSystem]] via has-part
    - bridges-to:: [[PlayerEngagement]] via enables
    - bridges-to:: [[ProceduralBehavior]] via implements
    - bridges-to:: [[EmergentBehavior]] via enables
    - bridges-to:: [[DynamicGameplay]] via enables
    - bridges-to:: [[GameState]] via requires
    - bridges-to:: [[DecisionEngine]] via has-part
    - bridges-to:: [[BehaviorTree]] via has-part
    - bridges-to:: [[GameEngine]] via requires
    - bridges-to:: [[AdaptiveDifficulty]] via implements
    - bridges-to:: [[AdaptiveChallenge]] via enables
    - bridges-to:: [[ReinforcementLearning]] via implements
    - bridges-from:: [[GameEngine]] via is-required-by

  - 
### Relationships
- is-subclass-of:: [[AIAgentSystem]]


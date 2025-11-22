- ### OntologyBlock
  id:: manipulator-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[RoboticsTechnology]]
    - term-id:: RB-0003
    - preferred-term:: Manipulator
    - source-domain:: robotics
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: A manipulator is a robot consisting of a series of segments, typically moving in a serial or parallel kinematic chain, with an end-effector for performing tasks.
    - maturity:: mature
    - source:: [[ISO 8373:2021]]
    - authority-score:: 0.99

  - **Semantic Classification**
    - owl:class:: rb:Manipulator
    - owl:physicality:: PhysicalEntity
    - owl:role:: Object
    - belongsToDomain:: [[Robotics]]

  - #### OWL Restrictions
    - hasKinematicChain some KinematicStructure
    - is-part-of some RobotRb0001
    - definesWorkspace some WorkspaceVolume
    - hasEndEffector some EndEffector

  - #### CrossDomainBridges
    - bridges-to:: [[WorkspaceVolume]] via definesWorkspace
    - bridges-to:: [[KinematicStructure]] via hasKinematicChain

  - 
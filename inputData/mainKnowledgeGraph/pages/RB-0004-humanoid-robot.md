- ### OntologyBlock
  id:: humanoid-robot-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[RoboticsTechnology]]
    - term-id:: RB-0004
    - preferred-term:: Humanoid Robot
    - source-domain:: robotics
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: A humanoid robot is a robot whose overall appearance and kinematic structure are based on the human body, typically including a head, torso, two arms, and two legs.
    - maturity:: mature
    - source:: [[ISO 8373:2021]]
    - authority-score:: 0.96

  - **Semantic Classification**
    - owl:class:: rb:HumanoidRobot
    - owl:physicality:: PhysicalEntity
    - owl:role:: Object
    - belongsToDomain:: [[Robotics]]

  - #### OWL Restrictions
    - is-part-of some MobilerobotRb0002
    - performsBipedalLocomotion some WalkingGait
    - is-part-of some RobotRb0001

  - #### CrossDomainBridges
    - bridges-to:: [[WalkingGait]] via performsBipedalLocomotion

  - 
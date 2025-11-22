- ### OntologyBlock
  id:: industrial-robot-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[RoboticsTechnology]]
    - term-id:: RB-0005
    - preferred-term:: Industrial Robot
    - source-domain:: robotics
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: An industrial robot is an automatically controlled, reprogrammable, multipurpose manipulator programmable in three or more axes, which may be either fixed in place or mobile for use in industrial automation applications.
    - maturity:: mature
    - source:: [[ISO 8373:2021]]
    - authority-score:: 1.0

  - **Semantic Classification**
    - owl:class:: rb:IndustrialRobot
    - owl:physicality:: PhysicalEntity
    - owl:role:: Object
    - belongsToDomain:: [[Robotics]]

  - #### OWL Restrictions
    - is-part-of some ManipulatorRb0003
    - operatesIn some IndustrialEnvironment

  - #### CrossDomainBridges
    - bridges-to:: [[ManipulatorRb0003]] via is-part-of
    - dt:requires:: [[PathPlanning]]
    - dt:uses:: [[Machine Learning]]
    - dt:uses:: [[Computer Vision]]

  - 
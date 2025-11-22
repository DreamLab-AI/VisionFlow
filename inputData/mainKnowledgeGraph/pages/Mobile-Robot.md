- ### OntologyBlock
  id:: mobile-robot-ontology
  collapsed:: true

  - **Identification**

    - domain-prefix:: RB

    - sequence-number:: 0002

    - filename-history:: ["RB-0002-mobile-robot.md"]
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[RoboticsTechnology]]
    - term-id:: RB-0002
    - preferred-term:: Mobile Robot
    - source-domain:: robotics
    - status:: complete
    - version:: 2.0.0
    - last-updated:: 2025-11-14

  - **Definition**
    - definition:: A mobile robot is a robot with locomotion capabilities that enable it to move within its working environment.
    - maturity:: mature
    - source:: [[ISO 8373:2021]]
    - authority-score:: 0.98

  - **Semantic Classification**
    - owl:class:: rb:MobileRobot
    - owl:physicality:: PhysicalEntity
    - owl:role:: Object
    - belongsToDomain:: [[Robotics]]

  - #### OWL Restrictions
    - hasNavigationCapability some NavigationSystem
    - performsLocalisation some LocalisationMethod
    - is-part-of some RobotRb0001
    - hasLocomotionSystem some LocomotionMechanism
    - operatesInEnvironment some OperationalEnvironment

  - #### CrossDomainBridges
    - bridges-to:: [[LocalisationMethod]] via performsLocalisation
    - bridges-to:: [[LocomotionMechanism]] via hasLocomotionSystem
    - bridges-to:: [[NavigationSystem]] via hasNavigationCapability
    - bridges-to:: [[OperationalEnvironment]] via operatesInEnvironment
    - dt:requires:: [[PathPlanning]]
    - dt:uses:: [[Computer Vision]]

  - 
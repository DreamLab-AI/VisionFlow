# RB-0001-autonomous-robot

- ontology:: Robotics
  term-id:: RB-0001
  preferred-term:: Autonomous Robot
  definition:: A robot capable of performing tasks without continuous human guidance using sensors and decision-making algorithms
  alt-terms:: autonomous system, self-directed robot, intelligent robot
  source-domain:: rb:autonomous-systems:ground
  status:: complete
  maturity:: established
  public-access:: true
  version:: 3.1.0
  last-updated:: 2025-11-20
  quality-score:: 93
  authority-score:: 95
  source:: Robotics: Modelling, Planning and Control (Siciliano et al., 2010)
  owl:class:: AutonomousRobot
  owl:physicality:: PhysicalEntity
  owl:role:: Agent
  belongsToDomain:: [[rb:]]
  implementedInLayer:: [[hardware-layer]], [[software-layer]]
  is-subclass-of:: [[Robot]], [[Autonomous-System]]
  physicality:: physical
  autonomy-level:: level-4
  sensing-modality:: LIDAR, camera, IMU, GPS
  actuation-type:: electric-motors
  control-architecture:: hierarchical
  mobility-type:: wheeled
  task-domain:: navigation, object-manipulation
  human-robot-interaction:: supervisory-control
  cross-domain-links:: [[ai-rb:autonomous-navigation]], [[rb-tc:teleoperation]]

Autonomous robots represent the cutting edge of robotics research and deployment.

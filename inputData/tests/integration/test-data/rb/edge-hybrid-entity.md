# RB-0003-hybrid-robot

- ontology:: Robotics
  term-id:: RB-0003
  preferred-term:: Hybrid Physical-Virtual Robot
  definition:: A robotic system that exists simultaneously in physical and virtual spaces
  alt-terms:: cyber-physical robot, digital-twin robot
  source-domain:: rb:autonomous-systems:hybrid
  status:: draft
  maturity:: emerging
  public-access:: true
  version:: 1.0.0
  last-updated:: 2025-11-21
  owl:class:: HybridRobot
  owl:physicality:: HybridEntity
  owl:role:: Agent
  belongsToDomain:: [[rb:]], [[mv:]]
  implementedInLayer:: [[hardware-layer]], [[software-layer]], [[platform-layer]]
  is-subclass-of:: [[Robot]], [[Digital-Twin]]
  physicality:: hybrid
  autonomy-level:: level-5
  sensing-modality:: physical-sensors, virtual-sensors, simulated-sensors
  actuation-type:: physical-motors, virtual-actuators
  control-architecture:: distributed
  mobility-type:: omnidirectional
  task-domain:: simulation-to-reality-transfer
  human-robot-interaction:: VR-interface
  cross-domain-links:: [[rb-mv:digital-twin-robot]], [[rb-ai:sim-to-real]]

Edge case testing hybrid entities that span physical and virtual domains.

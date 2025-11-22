# RB-0002-namespace-error

- ontology:: Robotics
  term-id:: RB-0002
  preferred-term:: Namespace Mismatch Robot
  definition:: This has incorrect namespace prefix (common migration error)
  source-domain:: mv:autonomous-systems
  status:: complete
  public-access:: true
  version:: 1.0.0
  last-updated:: 2025-11-21
  owl:class:: NamespaceMismatchRobot
  owl:physicality:: PhysicalEntity
  owl:role:: Agent
  physicality:: physical
  autonomy-level:: level-2

This is a common error where robotics files incorrectly use mv: namespace.
The migration pipeline should detect and fix this to rb:

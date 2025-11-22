- ### OntologyBlock
    - term-id:: RB-0183
    - preferred-term:: Differential Kinematics
    - ontology:: true
    - is-subclass-of:: [[Robot Kinematics]]
    - version:: 1.0.0

## Differential Kinematics

Differential Kinematics establishes the mathematical relationship between joint velocities and end-effector Cartesian velocities through the Jacobian matrix, enabling velocity-level motion control, real-time trajectory tracking, and singularity analysis. Unlike forward/inverse kinematics operating on positions, differential kinematics operates on instantaneous velocities, providing the foundation for resolved-rate control, teleoperation, and impedance control in modern robotics.

The fundamental relationship is ẋ = J(q)q̇, where ẋ represents end-effector velocity (linear and angular), q̇ denotes joint velocities, and J(q) is the manipulator Jacobian matrix. For an n-joint robot moving in m-dimensional task space, J is an m×n matrix whose elements are partial derivatives ∂xi/∂qj. The Jacobian maps joint space velocities to task space velocities, with its pseudoinverse J† enabling inverse velocity mapping: q̇ = J†ẋ for redundant manipulators.

Singularities occur when J loses rank (det(J) = 0 for square matrices), representing configurations where certain task-space motions become impossible or require infinite joint velocities. Singularities include wrist singularities (reduced orientation control), elbow singularities, and boundary singularities at workspace limits. Manipulability μ = √det(JJᵀ) quantifies dexterity, with μ → 0 near singularities. Redundancy resolution exploits null-space motion for secondary objectives like obstacle avoidance or joint limit avoidance.

As of 2024-2025, differential kinematics enables advanced control modes: Cartesian velocity control for teleoperated surgery, admittance control for collaborative robots per ISO/TS 15066, and visual servoing for vision-guided manipulation. Damped least-squares inverse J† = Jᵀ(JJᵀ + λI)⁻¹ provides numerical robustness near singularities. Real-time implementations leverage analytic Jacobians for common robot architectures (e.g., 6-DOF anthropomorphic arms) or automatic differentiation for complex kinematics. Computational efficiency enables 1 kHz update rates critical for force control and dynamic obstacle avoidance in industrial and service robotics applications.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: differentialkinematics-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0183
- **Filename History**: ["RB-0183-differentialkinematics.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:DifferentialKinematics
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Robot Kinematics]]

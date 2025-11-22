- ### OntologyBlock
    - term-id:: RB-0134
    - preferred-term:: Cylindrical Robot
    - ontology:: true
    - is-subclass-of:: [[Industrial Robot]]
    - version:: 1.0.0

## Cylindrical Robot

Cylindrical Robot features kinematic architecture comprising one revolute (rotary) base joint and two prismatic (linear) joints, generating a cylindrical workspace envelope. This configuration provides vertical reach and radial extension while maintaining rotational positioning around a central axis, combining the simplicity of Cartesian robots with the space efficiency of polar coordinates. The typical structure includes a rotating base, a vertical sliding column, and a radially extending arm.

The cylindrical coordinate system (r, θ, z) maps naturally to the robot's joint variables: theta for base rotation, z for vertical height, and r for radial reach. Forward kinematics remain straightforward, enabling rapid real-time position computation. Workspace volume forms a hollow cylinder, with the central column creating a dead zone directly on the rotation axis. Typical specifications include 1-3 meters radial reach, 1-2 meters vertical travel, and 270-360 degree rotation, with repeatability of ±0.1-0.5mm.

Cylindrical robots dominated assembly and material handling applications during the 1970s-1990s, particularly for machine tending, spot welding, and packaging. Major manufacturers included PUMA (Programmable Universal Machine for Assembly) and Cincinnati Milacron. However, articulated and SCARA robots have largely supplanted cylindrical designs in modern manufacturing due to superior dexterity and workspace optimization. Cylindrical robots persist in niche applications leveraging their strengths: clean room material handling where minimal horizontal footprint matters, vertical assembly tasks requiring precise height control, and educational settings demonstrating fundamental robotics concepts.

As of 2024-2025, cylindrical robots represent less than 5% of industrial robot installations. Legacy systems remain in automotive plants and semiconductor fabs. Modern implementations utilize AC servo motors with absolute encoders, eliminating homing routines. When specified, they comply with ISO 10218 safety standards, incorporating light curtains and speed/separation monitoring per ISO/TS 15066. Programming typically employs teach pendant methods or offline simulation in RobotStudio or similar packages.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: cylindricalrobot-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0134
- **Filename History**: ["RB-0134-cylindricalrobot.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:CylindricalRobot
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Industrial Robot]]

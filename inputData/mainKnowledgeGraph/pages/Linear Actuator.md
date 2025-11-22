- ### OntologyBlock
    - term-id:: RB-0174
    - preferred-term:: Electric Linear Actuator
    - ontology:: true
    - is-subclass-of:: [[DisruptiveTechnology]]
    - version:: 1.0.0

## Electric Linear Actuator

Electric Linear Actuator transforms rotational motor output into controlled linear motion through mechanical transmission mechanisms including ball screws, lead screws, belt drives, or rack-and-pinion systems, combining electric motor controllability with linear force generation essential for numerous robotic tasks. These integrated mechatronic devices package motor, transmission, guide rails, position feedback, and limit switches into single assemblies providing direct linear motion without external conversion mechanisms, simplifying robot design and improving reliability through reduced component count.

Common conversion mechanisms each offer distinct characteristics. Ball screw actuators provide high efficiency (90%+), precision positioning (±0.01mm), and high load capacity but cost more and require clean environments. Lead screw variants trade efficiency (30-70%) for lower cost and self-locking behavior preventing backdrive. Belt drive actuators achieve high speeds (to 3 m/s) and long strokes (to 10 meters) efficiently but with reduced stiffness. Rack-and-pinion designs offer unlimited stroke through modular racks, moderate efficiency, and simple construction. Rodless cylinders using magnetic or mechanical coupling through sealed tubes provide compact packaging for long strokes.

Performance specifications span stroke lengths from 25mm to 6 meters, speeds from 1 mm/s to 3 m/s, forces from 100 N to 100 kN, and positioning accuracy from ±1mm to ±0.01mm depending on mechanism and feedback system. Integrated features typically include DC or stepper motors with controllers, absolute or incremental position encoders, electromagnetic brakes, adjustable limit switches, and IP-rated housings. Supply voltages commonly are 12-48 VDC for mobile robotics or 115-230 VAC for industrial installations.

As of 2024-2025, electric linear actuators dominate collaborative robot base positioning, mobile robot lifting mechanisms, service robot extendable arms, and modular automation systems. Progressive Automation, Thomson Industries, and Actuonix supply standardized actuators with CANopen, EtherCAT, or serial control interfaces. Compact designs integrate planetary gearboxes and BLDC motors achieving 24 VDC operation for battery-powered mobile platforms. Force limiting and zero-velocity detection enable safe human contact per ISO/TS 15066. Smart actuators incorporate current monitoring for collision detection and predictive maintenance. Applications span warehouse logistics (vertically mobile robots), medical robotics (ISO 13485 certified), and outdoor mobile platforms meeting IP65/IP67 requirements for UK agricultural and construction environments.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: linearactuator-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0174
- **Filename History**: ["RB-0174-linearactuator.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:LinearActuator
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Electric Actuator]]

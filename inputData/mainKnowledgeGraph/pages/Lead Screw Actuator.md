- ### OntologyBlock
    - term-id:: RB-0176
    - preferred-term:: Lead Screw Actuator
    - ontology:: true
    - is-subclass-of:: [[Linear Actuator]]
    - version:: 1.0.0

## Lead Screw Actuator

Lead Screw Actuator converts rotary motion into linear displacement through threaded rod (screw) engaging a nut, offering cost-effective linear positioning with high load capacity and mechanical self-locking properties. This simple mechanism, dating to Archimedes, persists in robotics where reliability, low cost, and inherent backdrive resistance outweigh limitations in speed and efficiency compared to ball screw alternatives. Lead screws excel in holding vertical loads without power consumption and providing high force multiplication.

Configuration comprises a threaded screw shaft (typically trapezoidal or ACME threads) rotating within a nut that translates along the screw axis when prevented from rotating. Motor coupling drives screw rotation while nut attachment provides linear output. Key parameters include lead (linear travel per revolution, typically 1-25mm), thread angle (29° for ACME, 30° for metric trapezoidal), and efficiency determined by friction between nut and screw surfaces. Standard thread profiles include ACME (strong, easy to manufacture), trapezoidal (ISO standard metric), and buttress (high efficiency in one direction).

Mechanical advantage follows l = 2πrη/p, where p is screw pitch, r is input moment arm, and η is efficiency. Typical efficiency ranges 20-80% depending on lead angle, lubrication, and nut material. Plastic nuts (acetal, PTFE-filled) offer quiet operation and corrosion resistance but limited temperature range. Bronze nuts provide higher load capacity and better wear resistance. Self-locking occurs when friction prevents backdriving, critical for vertical positioning without continuous motor power. Critical speed limiting maximum RPM prevents whirling depends on screw length and diameter per Fc = 4.76×10^6 × d² / L².

As of 2024-2025, lead screw actuators serve cost-sensitive robotic applications: educational robots, prototype systems, door opening mechanisms in service robots, and low-speed positioning stages. Typical specifications include travel to 1 meter, speeds to 100 mm/s, and loads to 50 kN. Anti-backlash nuts using spring-loaded split designs reduce positioning error. Stainless steel construction suits food processing robots meeting IP65/IP69K standards. While ball screws dominate precision industrial robotics due to 90%+ efficiency and minimal friction, lead screws remain relevant where simplicity, cost, and self-locking behavior provide system-level advantages, particularly in collaborative robots requiring inherent backdrive resistance for safety per ISO/TS 15066 requirements.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: leadscrewactuator-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0176
- **Filename History**: ["RB-0176-leadscrewactuator.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:LeadScrewActuator
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Linear Actuator]]

- ### OntologyBlock
    - term-id:: RB-0180
    - preferred-term:: Hydraulic Motor
    - ontology:: true
    - is-subclass-of:: [[Hydraulic Actuator]]
    - version:: 1.0.0

## Hydraulic Motor

Hydraulic Motor converts pressurized fluid flow into continuous rotational motion, generating high torque from compact dimensions through positive displacement principles. These actuators enable mobile robots and heavy-duty manipulators to achieve power outputs and torque densities unattainable with electric motors of comparable size, particularly at low speeds where electric motors require bulky gear reducers. Hydraulic motors excel in harsh environments withstanding contamination, temperature extremes, and mechanical abuse that would destroy electric alternatives.

Operating types include gear motors (external, internal, gerotor), vane motors, piston motors (axial, radial), and gerotor motors. Gear motors offer simple construction, low cost, and tolerances to contamination, suitable for mobile robot wheel drives. Vane motors provide smooth operation and moderate efficiency (75-85%). Piston motors achieve highest efficiency (90-95%) and power density, with variable displacement variants enabling speed control at constant pressure. Performance metrics include displacement (cm³/revolution), rated pressure (100-350 bar), maximum speed (500-10,000 RPM depending on type), and torque capacity.

Unlike electric motors requiring magnetic materials and windings, hydraulic motors feature purely mechanical operation: fluid pressure acting on gears, vanes, or pistons creates imbalance forces producing rotation. This enables operation in explosive atmospheres without ignition risk, immersion in liquids, and extreme temperature ranges (-40°C to +120°C with appropriate fluids). Bi-directional operation results from reversing flow direction. Speed control via flow regulation provides infinite variability, while torque limiting through pressure relief valves prevents overload damage.

As of 2024-2025, hydraulic motors power track drives in heavy mobile robots (demolition, military EOD), swing drives in excavator robots, and joint actuators in heavy-payload manipulators. Variable displacement motors enable energy-efficient speed control matching load requirements. Integrated brake valves prevent runaway under gravity loads. Low-speed high-torque (LSHT) motors eliminate gearboxes in wheel drives for agricultural and construction robots. ISO 3019 and ISO 4391 standardize mounting flanges and shaft dimensions. Modern designs incorporate load sensing and flow sharing for multi-actuator coordination, critical in complex mobile platforms operating in UK construction and agriculture sectors.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: hydraulicmotor-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0180
- **Filename History**: ["RB-0180-hydraulicmotor.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:HydraulicMotor
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Hydraulic Actuator]]

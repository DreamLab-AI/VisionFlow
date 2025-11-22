- ### OntologyBlock
    - term-id:: RB-0179
    - preferred-term:: Hydraulic Cylinder
    - ontology:: true
    - is-subclass-of:: [[Hydraulic Actuator]]
    - version:: 1.0.0

## Hydraulic Cylinder

Hydraulic Cylinder converts fluid pressure into linear force and motion through a piston moving within a sealed cylindrical chamber, representing the most fundamental hydraulic actuator for heavy-duty robotic applications. These devices generate tremendous forces (tens to hundreds of kilonewtons) from compact packages, making them essential for applications exceeding electric actuator capabilities such as heavy material handling, earthmoving robotics, and hydraulic rescue robots. The power density advantage of hydraulics (10-20 kW/kg vs. 1-5 kW/kg for electric) enables robot designs impossible with alternative actuation.

Construction comprises a cylindrical barrel, piston with seals, piston rod extending through one end cap, and hydraulic ports for fluid inlet/outlet. Double-acting cylinders accept pressurized fluid on both sides of the piston, enabling bidirectional force application and precise position control. Single-acting cylinders use pressure on one side with spring or gravity return. Telescoping cylinders provide extended stroke from compact retracted length. Cushioning mechanisms (tapered piston or adjustable orifices) decelerate the piston near stroke ends, preventing shock loads. Typical specifications span bore diameters from 25mm to 500mm, strokes to 3+ meters, and operating pressures of 100-350 bar (1450-5000 psi).

Force output follows F = P × A, where P is fluid pressure and A is piston area, providing linear force-pressure relationship ideal for constant force applications. Position control requires servo valves or proportional directional valves modulating flow, with linear encoders or magnetostrictive sensors providing feedback. Response bandwidth remains limited (5-50 Hz) compared to electric actuators due to fluid compressibility and valve dynamics, but force capacity far exceeds electric alternatives.

As of 2024-2025, hydraulic cylinders persist in heavy mobile robots (construction, agriculture, forestry), large industrial manipulators (forging, foundry), and rescue robotics per ISO 13849 safety requirements. Modern designs incorporate wear rings reducing seal friction, chrome-plated rods resisting corrosion, and integrated sensors (LVDT, temposonic) for closed-loop control. ISO 6020 and ISO 6022 standardize mounting dimensions and dimensions. Fire-resistant fluids (HFC, HFD) replace petroleum oils in underground mining robots per UK mining regulations. Electrohydraulic actuation combining electric control with hydraulic power serves applications demanding both precision and high force.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: hydrauliccylinder-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0179
- **Filename History**: ["RB-0179-hydrauliccylinder.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:HydraulicCylinder
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Hydraulic Actuator]]

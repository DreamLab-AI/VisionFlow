- ### OntologyBlock
    - term-id:: RB-0158
    - preferred-term:: Linear Encoder
    - ontology:: true
    - is-subclass-of:: [[Encoder]]
    - version:: 1.0.0

## Linear Encoder

Linear Encoder measures linear position or displacement through non-contact sensing of periodic graduations on a scale, providing essential feedback for precise position control in linear axes of robots including gantry systems, linear stages, and prismatic joints. These sensors achieve resolution to sub-micrometer levels with measurement ranges spanning millimeters to tens of meters, enabling accuracy and repeatability impossible through alternative methods like lead screw rotation counting that suffer cumulative transmission errors.

Operating principles include optical, magnetic, capacitive, and inductive technologies. Optical encoders use LED or laser illumination reading precision glass or metal scales etched with periodic gratings (typically 20-40 µm pitch), with photodiode arrays detecting diffraction patterns through phase or amplitude modulation. Magnetic encoders sense magnetized pole pairs on steel or plastic strips using Hall effect or magnetoresistive sensors, offering superior contamination resistance. Capacitive encoders measure changing capacitance between scale and sensor. Incremental encoders output quadrature pulse trains (A, B channels with 90° phase shift plus reference index) enabling direction detection and subdivision to resolution λ/n where λ is grating period and n is interpolation factor (typically 100-4096). Absolute encoders output unique position codes across the entire measurement range, eliminating homing requirements.

Performance characteristics include resolution (0.1 µm to 5 µm typical), accuracy (±1 µm to ±10 µm per meter), maximum velocity (to 10 m/s), and operating conditions. Open-loop systems mount exposed scales requiring clean environments, while enclosed systems protect scales within sealed housings suitable for industrial conditions per IP67 or higher ratings. Interface standards include differential TTL, EIA-422, 1 Vpp analog (interpolated by controller), SSI, BiSS, Fanuc serial, and EnDat protocols.

As of 2024-2025, linear encoders provide critical feedback for precision robotics: gantry robots achieving ±0.05mm absolute accuracy across multi-meter workspaces, medical surgical robots meeting ISO 13485 precision requirements, semiconductor handling robots with sub-micrometer positioning, and machine tool loading systems. Renishaw, Heidenhain, and Fagor manufacture scale lengths to 30 meters with TouchDSD contactless interfaces eliminating wear. Time-of-flight laser interferometry offers nanometer resolution for ultra-precision applications. Safety-rated encoders with SIL 2/PLd certification enable speed and position monitoring for collaborative robot safety functions per ISO/TS 15066, critical for UK manufacturing automation deployments.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: linearencoder-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0158
- **Filename History**: ["RB-0158-linearencoder.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:LinearEncoder
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Encoder]]

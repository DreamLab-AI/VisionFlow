- ### OntologyBlock
    - term-id:: RB-0140
    - preferred-term:: Lower Limb Exoskeleton
    - ontology:: true
    - is-subclass-of:: [[Exoskeleton Robot]]
    - version:: 1.0.0

## Lower Limb Exoskeleton

Lower Limb Exoskeleton constitutes a wearable robotic system mechanically coupled to the user's legs, hips, and pelvis to assist, augment, or restore walking and standing functions. These powered orthotic devices apply torques and forces to hip, knee, and ankle joints through actuators controlled by sensors detecting user intent or gait patterns, enabling mobility restoration for individuals with paralysis, gait training for stroke patients, or endurance augmentation for industrial workers. Applications span medical rehabilitation per ISO 13482, military load bearing, and industrial ergonomics.

Architectural components include rigid or soft exoskeletal structures aligned with biological joint axes, actuators (electric motors, series elastic actuators, pneumatic artificial muscles) providing joint torques, sensors (IMUs, joint encoders, force sensors, EMG) detecting motion intent and ground contact, embedded control systems executing gait algorithms, and power supplies (lithium batteries typically 24-48V, 200-400Wh capacity). Attachment interfaces using adjustable straps, cuffs, or custom-fitted shells transfer forces between robot and user. Passive variants utilize springs and clutches without powered actuation, while active systems employ closed-loop control adjusting assistance in real-time.

Control strategies vary by application. Rehabilitation exoskeletons implement predefined gait trajectories with impedance control allowing patient contribution, using iterative learning to reduce assistance as capability improves. Assistance amplification systems detect biological intent through surface EMG or force sensors, proportionally augmenting natural muscle output. Autonomous walking controllers execute full gait cycles for individuals with complete paralysis, using finite state machines transitioning between stance and swing phases based on ground reaction forces and tilt sensing.

As of 2024-2025, commercial medical exoskeletons include ReWalk, Ekso GT, and Cyberdyne HAL (Hybrid Assistive Limb) providing FDA/CE-marked rehabilitation and personal mobility for spinal cord injury. Industrial exoskeletons from Lockheed Martin (FORTIS) and Ekso Bionics (EksoVest) reduce metabolic cost 15-40% during repetitive bending and lifting in UK manufacturing and logistics. Research advances include soft fabric-based designs reducing weight to 3-7 kg, AI-based intent recognition improving responsiveness, and variable stiffness actuators enhancing energy efficiency. UK's National Institute for Health Research funds exoskeleton trials evaluating clinical outcomes per NICE guidelines. Safety standards ISO 13482 and EN 1907-1 govern personal care robots and PPE requirements.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: lowerlimbexoskeleton-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0140
- **Filename History**: ["RB-0140-lowerlimbexoskeleton.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:LowerLimbExoskeleton
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Exoskeleton Robot]]

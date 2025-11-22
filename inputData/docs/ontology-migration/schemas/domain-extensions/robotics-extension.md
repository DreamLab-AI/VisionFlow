# Robotics Domain Extension Schema

**Version:** 2.0.0
**Date:** 2025-11-21
**Domain:** Robotics (rb:)
**Base URI:** `http://narrativegoldmine.com/robotics#`
**Term Prefix:** RB-XXXX

---

## Domain Overview

The Robotics domain covers autonomous systems, sensors, actuators, control systems, human-robot interaction, and various robot types. This domain extension defines robotics-specific properties and patterns that extend the core ontology schema.

**CRITICAL MIGRATION NOTE**: All existing `mv:rb*` namespaces must migrate to `rb:*` as part of the multi-ontology standardization.

---

## Sub-Domains

| Sub-Domain | Namespace | Description | Example Concepts |
|------------|-----------|-------------|------------------|
| Autonomous Systems | `rb:auto:` | Self-governing robots, autonomous vehicles | Autonomous Navigation, Self-Driving Car, Drone |
| Sensors & Perception | `rb:sense:` | Environmental sensing, perception systems | LiDAR, Computer Vision, Tactile Sensor, IMU |
| Actuators & Control | `rb:control:` | Movement, manipulation, control systems | Servo Motor, Gripper, PID Controller, Motion Planning |
| Human-Robot Interaction | `rb:hri:` | Human collaboration, social robotics | Collaborative Robot, Social Robot, Teleoperation |
| Robot Types | `rb:types:` | Classifications of robots | Industrial Robot, Service Robot, Medical Robot, Humanoid |
| Kinematics | `rb:kine:` | Motion and mechanical analysis | Forward Kinematics, Inverse Kinematics, Dynamics |

---

## Robotics-Specific Properties

### Physical Properties

**rb:robot-type** (enum)
- **Purpose**: Primary classification of robot form
- **Values**: mobile, manipulator, aerial, underwater, humanoid, hybrid, legged, wheeled, tracked
- **Example**: `rb:robot-type:: mobile`

**rb:degrees-of-freedom** (integer)
- **Purpose**: Number of independent motion axes
- **Format**: Integer (DOF count)
- **Example**: `rb:degrees-of-freedom:: 6`

**rb:payload-capacity** (decimal)
- **Purpose**: Maximum load the robot can carry/manipulate
- **Format**: Decimal (kilograms)
- **Example**: `rb:payload-capacity:: 100.5`

**rb:operating-environment** (enum)
- **Purpose**: Primary operational environment
- **Values**: indoor, outdoor, underwater, aerial, space, hazardous, cleanroom
- **Example**: `rb:operating-environment:: indoor`

**rb:dimensions** (string)
- **Purpose**: Physical size of robot
- **Format**: String (LxWxH in meters)
- **Example**: `rb:dimensions:: 0.8x0.6x0.4`

**rb:weight** (decimal)
- **Purpose**: Total weight of robot
- **Format**: Decimal (kilograms)
- **Example**: `rb:weight:: 45.2`

**rb:battery-life** (decimal)
- **Purpose**: Operating duration on battery
- **Format**: Decimal (hours)
- **Example**: `rb:battery-life:: 8.0`

**rb:power-source** (enum)
- **Purpose**: Primary power source
- **Values**: battery, electric, hydraulic, pneumatic, hybrid, solar
- **Example**: `rb:power-source:: battery`

### Capability Properties

**rb:autonomy-level** (enum)
- **Purpose**: Degree of autonomous operation
- **Values**: teleoperated, semi-autonomous, fully-autonomous, supervised-autonomous
- **Example**: `rb:autonomy-level:: fully-autonomous`

**rb:navigation-method** (page link list)
- **Purpose**: Navigation algorithms and approaches used
- **Example**: `rb:navigation-method:: [[SLAM]], [[GPS]], [[Visual Odometry]], [[Occupancy Grid Mapping]]`

**rb:manipulation-capability** (boolean)
- **Purpose**: Can manipulate objects in environment
- **Values**: true, false
- **Example**: `rb:manipulation-capability:: true`

**rb:perception-modalities** (page link list)
- **Purpose**: Sensing technologies employed
- **Example**: `rb:perception-modalities:: [[LiDAR]], [[RGB-D Camera]], [[Ultrasonic]], [[IMU]], [[Force Sensor]]`

**rb:communication-protocols** (page link list)
- **Purpose**: Communication standards supported
- **Example**: `rb:communication-protocols:: [[ROS]], [[ROS2]], [[MQTT]], [[CAN Bus]], [[Ethernet]]`

**rb:mobility-type** (enum)
- **Purpose**: How the robot moves
- **Values**: wheeled, tracked, legged, aerial, aquatic, hybrid
- **Example**: `rb:mobility-type:: wheeled`

**rb:speed** (string)
- **Purpose**: Maximum operational speed
- **Format**: String with units (m/s, km/h)
- **Example**: `rb:speed:: 1.5 m/s`

**rb:reach** (decimal)
- **Purpose**: Maximum reach distance (for manipulators)
- **Format**: Decimal (meters)
- **Example**: `rb:reach:: 0.85`

### Control Properties

**rb:control-architecture** (page link list)
- **Purpose**: Control system architecture type
- **Example**: `rb:control-architecture:: [[Hierarchical Control]], [[Reactive Control]], [[Hybrid Deliberative-Reactive]]`

**rb:control-frequency** (integer)
- **Purpose**: Control loop update rate
- **Format**: Integer (Hertz)
- **Example**: `rb:control-frequency:: 100`

**rb:feedback-control** (boolean)
- **Purpose**: Uses closed-loop feedback control
- **Values**: true, false
- **Example**: `rb:feedback-control:: true`

**rb:planning-algorithm** (page link list)
- **Purpose**: Path/motion planning algorithms
- **Example**: `rb:planning-algorithm:: [[RRT]], [[A*]], [[DWA]], [[MPC]]`

### Safety Properties

**rb:safety-features** (page link list)
- **Purpose**: Safety mechanisms implemented
- **Example**: `rb:safety-features:: [[Emergency Stop]], [[Collision Avoidance]], [[Force Limiting]], [[Safety Scanner]], [[Audible Warnings]]`

**rb:certifications** (list)
- **Purpose**: Safety and quality certifications
- **Example**: `rb:certifications:: ISO 10218, ISO 13849, CE Mark, UL Listed`

**rb:safety-rating** (enum)
- **Purpose**: Safety integrity level
- **Values**: SIL1, SIL2, SIL3, PLa, PLb, PLc, PLd, PLe
- **Example**: `rb:safety-rating:: PLd`

**rb:human-safe** (boolean)
- **Purpose**: Safe for direct human interaction
- **Values**: true, false
- **Example**: `rb:human-safe:: true`

### Application Properties

**rb:application-domain** (page link list)
- **Purpose**: Primary application areas
- **Example**: `rb:application-domain:: [[Manufacturing]], [[Warehouse Logistics]], [[Healthcare]], [[Agriculture]]`

**rb:tasks-performed** (page link list)
- **Purpose**: Specific tasks robot can perform
- **Example**: `rb:tasks-performed:: [[Material Transport]], [[Pick and Place]], [[Inspection]], [[Assembly]]`

**rb:industry-sector** (list)
- **Purpose**: Industry sectors deployed in
- **Example**: `rb:industry-sector:: Automotive, Electronics, Pharmaceutical, Food & Beverage`

---

## Robotics-Specific Relationships

### Sensing Relationships

**rb:senses-with** (page link list)
- **Purpose**: Sensor modalities used for perception
- **Example**: `rb:senses-with:: [[2D LiDAR]], [[3D Camera]], [[Ultrasonic Sensors]], [[Force-Torque Sensor]]`

**rb:detects** (page link list)
- **Purpose**: What the robot can detect/perceive
- **Example**: `rb:detects:: [[Obstacles]], [[Humans]], [[Objects]], [[Landmarks]]`

### Actuation Relationships

**rb:actuates-with** (page link list)
- **Purpose**: Actuator types employed
- **Example**: `rb:actuates-with:: [[DC Motors]], [[Servo Motors]], [[Hydraulic Actuators]], [[Pneumatic Grippers]]`

**rb:manipulates** (page link list)
- **Purpose**: What the robot can manipulate
- **Example**: `rb:manipulates:: [[Boxes]], [[Pallets]], [[Tools]], [[Surgical Instruments]]`

### Control Relationships

**rb:controlled-by** (page link list)
- **Purpose**: Control system or software
- **Example**: `rb:controlled-by:: [[ROS2 Navigation Stack]], [[MoveIt]], [[Custom Controller]]`

**rb:navigates-using** (page link list)
- **Purpose**: Navigation algorithms employed
- **Example**: `rb:navigates-using:: [[AMCL]], [[Cartographer]], [[RTABMap]], [[Visual SLAM]]`

### Interaction Relationships

**rb:collaborates-with** (page link list)
- **Purpose**: Humans or other robots it works with
- **Example**: `rb:collaborates-with:: [[Human Workers]], [[Other Robots in Fleet]]`

**rb:interfaces-with** (page link list)
- **Purpose**: External systems interfaced with
- **Example**: `rb:interfaces-with:: [[Warehouse Management System]], [[ERP]], [[Cloud Platform]]`

---

## Extended Template for Robotics Domain

```markdown
- ### [Robotics Concept Name]
  id:: rb-[concept-slug]-ontology
  collapsed:: true

  - **Identification** [CORE - Tier 1]
    - ontology:: true
    - term-id:: RB-XXXX
    - preferred-term:: [Human Readable Name]
    - alt-terms:: [[Alternative 1]], [[Alternative 2]]
    - source-domain:: robotics
    - status:: [draft | in-progress | complete | deprecated]
    - public-access:: [true | false]
    - version:: [M.m.p]
    - last-updated:: [YYYY-MM-DD]
    - quality-score:: [0.0-1.0]
    - cross-domain-links:: [number]

  - **Definition** [CORE - Tier 1]
    - definition:: [2-5 sentence comprehensive definition with [[concept links]]]
    - maturity:: [draft | emerging | mature | established]
    - source:: [[IEEE Robotics]], [[ISO 8373]], [[Academic Source]]
    - authority-score:: [0.0-1.0]

  - **Semantic Classification** [CORE - Tier 1]
    - owl:class:: rb:[ClassName]
    - owl:physicality:: [PhysicalEntity | HybridEntity most common]
    - owl:role:: [Object | Agent | Process]
    - owl:inferred-class:: rb:[PhysicalityRole]
    - belongsToDomain:: [[RoboticsDomain]], [[AutonomousSystemsDomain]]
    - belongsToSubDomain:: [[Autonomous Systems]], [[Sensors]], etc.

  - **Physical Properties** [RB EXTENSION]
    - rb:robot-type:: [mobile | manipulator | aerial | etc.]
    - rb:degrees-of-freedom:: [integer]
    - rb:payload-capacity:: [kg]
    - rb:operating-environment:: [indoor | outdoor | etc.]
    - rb:dimensions:: [LxWxH]
    - rb:weight:: [kg]
    - rb:battery-life:: [hours]

  - **Capability Properties** [RB EXTENSION]
    - rb:autonomy-level:: [teleoperated | semi-autonomous | fully-autonomous]
    - rb:navigation-method:: [[Method1]], [[Method2]]
    - rb:manipulation-capability:: [true | false]
    - rb:perception-modalities:: [[Sensor1]], [[Sensor2]]
    - rb:communication-protocols:: [[Protocol1]], [[Protocol2]]

  - **Control Properties** [RB EXTENSION]
    - rb:control-architecture:: [[Architecture Type]]
    - rb:control-frequency:: [Hz]
    - rb:planning-algorithm:: [[Algorithm1]], [[Algorithm2]]

  - **Safety Properties** [RB EXTENSION]
    - rb:safety-features:: [[Feature1]], [[Feature2]]
    - rb:certifications:: [ISO standards, etc.]
    - rb:human-safe:: [true | false]

  - #### Relationships [CORE - Tier 1]
    id:: rb-[concept-slug]-relationships

    - is-subclass-of:: [[ParentClass1]], [[ParentClass2]]
    - has-part:: [[Component1]], [[Component2]]
    - requires:: [[Requirement1]]
    - enables:: [[Capability1]]

  - #### Robotics-Specific Relationships [RB EXTENSION]
    - rb:senses-with:: [[Sensor1]], [[Sensor2]]
    - rb:actuates-with:: [[Actuator1]], [[Actuator2]]
    - rb:controlled-by:: [[Control System]]
    - rb:navigates-using:: [[Navigation Algorithm]]
    - rb:collaborates-with:: [[Collaborator]]

  - #### CrossDomainBridges [CORE - Tier 3]
    - bridges-to:: [[AI Algorithm]] via uses (RB → AI)
    - bridges-to:: [[Digital Twin]] via simulated-in (RB → MV)
    - bridges-to:: [[Blockchain Tracking]] via integrates (RB → BC)
```

---

## Common Robotics Patterns

### Pattern 1: Mobile Robot

```markdown
- ### [Robot Name]
  - **Semantic Classification**
    - owl:class:: rb:[RobotName]
    - owl:physicality:: PhysicalEntity
    - owl:role:: Agent
    - belongsToSubDomain:: [[Autonomous Systems]], [[Mobile Robots]]

  - **Physical Properties**
    - rb:robot-type:: mobile
    - rb:mobility-type:: wheeled
    - rb:dimensions:: [LxWxH]

  - **Capability Properties**
    - rb:autonomy-level:: fully-autonomous
    - rb:navigation-method:: [[SLAM]], [[Visual Odometry]]

  - #### Relationships
    - is-subclass-of:: [[Mobile Robot]], [[Autonomous System]]
```

### Pattern 2: Sensor System

```markdown
- ### [Sensor Name]
  - **Semantic Classification**
    - owl:class:: rb:[SensorName]
    - owl:physicality:: PhysicalEntity
    - owl:role:: Object
    - belongsToSubDomain:: [[Sensors & Perception]]

  - **Capability Properties**
    - rb:perception-modalities:: [[Modality Type]]

  - #### Relationships
    - is-subclass-of:: [[Sensor]], [[Perception System]]
    - is-part-of:: [[Robot System]]
```

### Pattern 3: Control Algorithm

```markdown
- ### [Algorithm Name]
  - **Semantic Classification**
    - owl:class:: rb:[AlgorithmName]
    - owl:physicality:: AbstractEntity
    - owl:role:: Process
    - belongsToSubDomain:: [[Actuators & Control]]

  - **Control Properties**
    - rb:control-architecture:: [[Architecture]]
    - rb:control-frequency:: [Hz]

  - #### Relationships
    - is-subclass-of:: [[Control Algorithm]]
    - used-by:: [[Robot Type]]
```

---

## Cross-Domain Bridge Patterns

### RB → AI

```markdown
- bridges-to:: [[Machine Learning Algorithm]] via uses (RB → AI)
- bridges-to:: [[Computer Vision]] via uses (RB → AI)
- bridges-to:: [[Reinforcement Learning]] via trained-with (RB → AI)
- bridges-from:: [[AI Perception Model]] via implements (AI → RB)
```

### RB → Metaverse

```markdown
- bridges-to:: [[Digital Twin]] via simulated-in (RB → MV)
- bridges-to:: [[Robot Simulation]] via modeled-in (RB → MV)
- bridges-to:: [[Teleoperation Interface]] via controlled-via (RB → MV)
```

### RB → Blockchain

```markdown
- bridges-to:: [[Supply Chain Tracking]] via integrates (RB → BC)
- bridges-to:: [[Robot-to-Robot Transactions]] via uses (RB → BC)
- bridges-to:: [[Provenance Logging]] via secured-by (RB → BC)
```

### RB → Disruptive Technologies

```markdown
- bridges-to:: [[Automation Disruption]] via exemplifies (RB → DT)
- bridges-to:: [[Autonomous Systems Innovation]] via enables (RB → DT)
- bridges-from:: [[Technology Assessment]] via evaluated-by (DT → RB)
```

---

## Validation Rules for Robotics Domain

### RB-Specific Validations

1. **Physical Consistency**
   - Payload capacity reasonable for robot size/type
   - Degrees of freedom align with robot type
   - Weight and dimensions consistent

2. **Autonomy Alignment**
   - Fully-autonomous robots should specify navigation methods
   - Teleoperated robots should specify control interface

3. **Safety Requirements**
   - Human-safe robots must have appropriate safety features
   - Safety certifications required for commercial deployment

4. **Capability Validation**
   - Manipulation capability requires actuators specified
   - Navigation requires perception modalities

---

## Migration Notes

### Migrating Existing Robotics Blocks

**CRITICAL: Namespace Migration**
1. **Change all `mv:rb*` to `rb:*`** in existing robotics blocks
2. **Update term-id** to use RB-XXXX prefix
3. **Change source-domain** to `robotics`

**Additional Migrations:**
4. **Add RB Physical Properties** to all robot systems
5. **Add RB Capability Properties** to autonomous systems
6. **Add RB Safety Properties** to deployed robots
7. **Specify Sub-Domain** for all robotics concepts
8. **Add rb:senses-with** relationships where applicable

### Priority Robotics Concepts for Migration

- Autonomous Mobile Robots (AMRs)
- Industrial Robot Arms
- Sensor Systems (LiDAR, cameras, etc.)
- Control Algorithms
- Navigation Systems
- Human-Robot Interaction Systems

---

**Document Control:**
- **Version**: 2.0.0
- **Status**: Authoritative
- **Domain Coordinator**: TBD
- **Last Updated**: 2025-11-21
- **Next Review**: 2026-01-21
- **Critical Migration**: All mv:rb* → rb:* namespace changes required

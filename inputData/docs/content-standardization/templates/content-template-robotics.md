# Robotics Domain Content Template

**Domain:** Robotics & Autonomous Systems
**Version:** 1.0.0
**Date:** 2025-11-21
**Purpose:** Template for robotics and autonomous systems concept pages

---

## Template Structure

```markdown
- ### OntologyBlock
  id:: [concept-slug]-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: RB-NNNN
    - preferred-term:: [Concept Name]
    - alt-terms:: [[Alternative 1]], [[Alternative 2]]
    - source-domain:: robotics
    - status:: [draft | in-progress | complete]
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: YYYY-MM-DD

  - **Definition**
    - definition:: [2-3 sentence technical definition with [[links]]]
    - maturity:: [emerging | mature | established]
    - source:: [[Authoritative Source 1]], [[Source 2]]

  - **Semantic Classification**
    - owl:class:: rb:ConceptName
    - owl:physicality:: [PhysicalEntity | HybridEntity | AbstractEntity]
    - owl:role:: [Agent | Object | Process | Quality]
    - belongsToDomain:: [[RoboticsDomain]]

  - #### Relationships
    id:: [concept-slug]-relationships

    - is-subclass-of:: [[Parent Robotics Concept]]
    - has-part:: [[Component1]], [[Component2]]
    - requires:: [[Sensor]], [[Actuator]], [[Control System]]
    - enables:: [[Capability1]], [[Capability2]]
    - relates-to:: [[Related Concept1]], [[Related Concept2]]

# {Concept Name}

## Technical Overview
- **Definition**: [2-3 sentence precise technical definition. For robotics concepts, focus on physical embodiment, sensing and actuation, control systems, or autonomous behaviour. Include [[Robot]], [[Autonomous Systems]], [[Sensor]], [[Actuator]], or other foundational concepts.]

- **Key Characteristics**:
  - [Physical design or morphology]
  - [Sensing capabilities and sensor suite]
  - [Actuation mechanisms and mobility]
  - [Control architecture and autonomy level]
  - [Environmental interaction and manipulation capabilities]

- **Primary Applications**: [Specific robotics applications this concept enables, such as [[Manufacturing Automation]], [[Autonomous Navigation]], [[Human-Robot Interaction]], [[Inspection]], etc.]

- **Related Concepts**: [[Broader Robotics Category]], [[Related System]], [[Alternative Approach]], [[Enabled Application]]

## Detailed Explanation
- Comprehensive overview
  - [Opening paragraph: What this robotics concept is, its role in autonomous systems, and why it matters. Connect to established paradigms like [[Mobile Robotics]], [[Industrial Robotics]], [[Autonomous Vehicles]], or [[Humanoid Robots]].]
  - [Second paragraph: How it works technically—mechanical design, sensor integration, control algorithms, or autonomous decision-making. Explain the mechatronics, kinematics, dynamics, or AI integration.]
  - [Third paragraph: Evolution and development—historical context (e.g., "Shakey robot 1960s", "DARPA Grand Challenge 2005"), breakthrough innovations, key milestones in robotics history.]

- Physical architecture and design
  - [Mechanical structure: Robot morphology, degrees of freedom, materials, weight, size.]
  - [Locomotion or manipulation: Wheels, legs, arms, grippers, end-effectors.]
  - [Actuation systems: Motors, hydraulics, pneumatics, artificial muscles.]
  - [Power systems: Batteries, power consumption, energy efficiency, charging/refuelling.]

- Sensing and perception
  - [Sensor suite: Cameras, LiDAR, radar, ultrasonic, IMU, GPS, tactile sensors.]
  - [Perception algorithms: Computer vision ([[Object Detection]], [[SLAM]]), sensor fusion, localisation.]
  - [Environmental awareness: Obstacle detection, mapping, scene understanding.]
  - [Human sensing: Face recognition, gesture recognition, speech recognition for HRI.]

- Control systems and autonomy
  - [Control architecture: Hierarchical, reactive, hybrid architectures; ROS (Robot Operating System).]
  - [Motion planning: Path planning algorithms ([[A*]], [[RRT]], [[Dijkstra]]), trajectory generation, collision avoidance.]
  - [Decision-making: Behaviour trees, finite state machines, reinforcement learning policies.]
  - [Autonomy levels: Teleoperated, semi-autonomous, fully autonomous; SAE levels for vehicles.]

- Artificial intelligence integration
  - [Machine learning: Supervised learning for perception, reinforcement learning for control.]
  - [Deep learning: CNNs for vision, RNNs/LSTMs for temporal sequences, transformers for language.]
  - [Planning and reasoning: Task planning, symbolic reasoning, knowledge representation.]
  - [Learning from demonstration: Imitation learning, inverse reinforcement learning.]

- Capabilities and features
  - [Primary capabilities: Navigation, manipulation, inspection, delivery, assembly, etc.]
  - [Advanced features: Adaptive behaviour, learning, multi-robot coordination, human collaboration.]
  - [Distinguishing characteristics: Autonomy level, robustness, versatility, safety features.]

- Human-robot interaction (HRI)
  - [Interaction modalities: Natural language, gestures, GUI, augmented reality interfaces.]
  - [Collaboration: Shared workspaces, safety systems, intent recognition, adaptive assistance.]
  - [Social aspects: Social robots, emotional expression, conversational agents.]

- Safety and reliability
  - [Safety systems: Emergency stops, collision avoidance, fail-safe mechanisms, redundancy.]
  - [Standards compliance: ISO 10218 (industrial robots), ISO 13482 (personal care robots), functional safety.]
  - [Reliability: Mean time between failures (MTBF), fault detection and diagnosis, maintenance.]

- Implementation considerations
  - [Deployment environments: Indoors vs. outdoors, structured vs. unstructured, GPS-denied.]
  - [Integration requirements: Existing systems, factory floors, warehouse infrastructure.]
  - [Operational aspects: Maintenance schedules, operator training, remote monitoring.]
  - [Cost factors: Hardware costs, software development, deployment, ongoing operation.]

## Academic Context
- Theoretical foundations
  - [Mechanical engineering: Kinematics, dynamics, control theory, mechanism design.]
  - [Electrical engineering: Sensors, actuators, embedded systems, power electronics.]
  - [Computer science: AI, machine learning, computer vision, path planning, real-time systems.]
  - [Interdisciplinary: Cognitive science, biomechanics, neuroscience for bio-inspired robotics.]

- Key researchers and institutions
  - [Pioneering researchers: E.g., "Rodney Brooks (behaviour-based robotics)", "Sebastian Thrun (autonomous vehicles)", "Hiroaki Kitano (humanoid robots)"]
  - **UK Institutions**:
    - **University of Oxford**: Robotics and autonomous systems, mobile robotics research
    - **Imperial College London**: Robotics, Hamlyn Centre for Robotic Surgery, aerial robotics
    - **University of Edinburgh**: Robotics, autonomous systems, AI for robotics
    - **University of Bristol**: Bristol Robotics Laboratory (BRL), largest robotics research centre in UK
    - **University of Cambridge**: Robotics, intelligent systems, bio-inspired robotics
    - **King's College London**: Soft robotics, medical robotics
    - **University of Leeds**: Autonomous vehicles, mobile robotics
  - [International institutions: MIT CSAIL, CMU Robotics Institute, Stanford AI Lab, ETH Zurich, etc.]

- Seminal papers and publications
  - [Foundational paper: E.g., Brooks, R. (1986). "A Robust Layered Control System for a Mobile Robot". IEEE Journal of Robotics and Automation.]
  - [SLAM: Thrun, S. et al. "Probabilistic Robotics"—comprehensive textbook.]
  - [Learning: Kober, J. & Peters, J. "Reinforcement Learning in Robotics"—survey of RL for robots.]
  - [Manipulation: Mason, M. "Mechanics of Robotic Manipulation"—grasping and manipulation theory.]
  - [Recent advance: Papers from 2023-2025 showing current state of the art in learning-based robotics, sim-to-real transfer, etc.]

- Current research directions (2025)
  - [Learning-based control: End-to-end learning, model-based RL, sim-to-real transfer, meta-learning.]
  - [Manipulation: Dexterous manipulation, non-prehensile manipulation, soft grippers, tactile sensing.]
  - [Multi-robot systems: Swarm robotics, distributed coordination, multi-agent reinforcement learning.]
  - [Human-robot collaboration: Safe coexistence, intent prediction, adaptive assistance, explainable behaviour.]
  - [Robustness and generalisation: Domain adaptation, robustness to sensor failures, long-term autonomy.]
  - [Bio-inspired robotics: Soft robotics, biomimetic design, neuromorphic control.]

## Current Landscape (2025)
- Industry adoption and implementations
  - [Current state: Industrial automation maturity, service robot growth, autonomous vehicle testing. Quantify if possible.]
  - **Major robotics companies**: [[Boston Dynamics]], [[ABB Robotics]], [[KUKA]], [[Fanuc]], [[Universal Robots]]
  - **Autonomous vehicles**: [[Waymo]], [[Cruise]], [[Tesla Autopilot]], [[Wayve]] (UK), [[Oxbotica]] (UK)
  - **UK robotics sector**: [[Ocado Technology]] (warehouse automation), [[Wayve]] (autonomous driving), [[CMR Surgical]] (medical robotics), [[Shadow Robot Company]]
  - [Industry verticals: Manufacturing, logistics, agriculture, healthcare, construction, defence, etc.]

- Technical capabilities and limitations
  - **Capabilities**:
    - [What robots can do well—repetitive tasks, precision, hazardous environments, inspection]
    - [State-of-the-art performance levels—navigation accuracy, manipulation success rates]
    - [Practical deployment success stories]
  - **Limitations**:
    - [Dexterity gap—human-level manipulation remains challenging]
    - [Unstructured environments—adaptation to novel situations]
    - [Cost and complexity—high upfront costs, maintenance requirements]
    - [Safety and reliability—ensuring safe human interaction, long-term reliability]
    - [Energy and autonomy—battery life, charging infrastructure for mobile robots]

- Standards and frameworks
  - **Robotics frameworks**: [[ROS]] (Robot Operating System), [[ROS 2]], [[YARP]], [[Webots]] (simulation)
  - **Simulation platforms**: [[Gazebo]], [[Isaac Sim]] (NVIDIA), [[PyBullet]], [[MuJoCo]]
  - **Safety standards**: ISO 10218 (industrial robots), ISO 13482 (personal care robots), ISO 13849 (functional safety)
  - **Autonomous vehicle standards**: SAE J3016 (levels of driving automation), ISO 26262 (automotive functional safety)
  - **Industry standards**: [RIA standards, ANSI/RIA R15.06, CE marking for EU]

- Ecosystem and tools
  - **Development tools**: ROS/ROS2, robot simulators, MATLAB Robotics Toolbox, V-REP
  - **Hardware platforms**: [[TurtleBot]], [[Fetch]], [[Spot]] (Boston Dynamics), [[Pepper]] (SoftBank), [[UR10]] (Universal Robots)
  - **Sensors**: [[Velodyne LiDAR]], [[Realsense]] (Intel), [[ZED]] (Stereolabs), [[RPLidar]]
  - **Cloud robotics**: AWS RoboMaker, Google Cloud Robotics, Azure Robotics
  - **Open source**: Open-source robot designs, sensor drivers, navigation stacks, manipulation packages

## UK Context
- British contributions and implementations
  - [UK innovations: E.g., "Bristol Robotics Laboratory—largest UK academic robotics centre", "Oxford mobile robotics research", "Cambridge bio-inspired robotics"]
  - [British robotics pioneers: Alan Turing (computational foundations), Grey Walter (early autonomous robots—tortoises)]
  - [Current UK leadership: Autonomous vehicles (Wayve, Oxbotica), warehouse automation (Ocado), medical robotics (CMR Surgical)]

- Major UK institutions and organisations
  - **Universities**:
    - **University of Bristol**: Bristol Robotics Laboratory (BRL)—joint venture with UWE Bristol, largest in UK
    - **University of Oxford**: Oxford Robotics Institute, mobile robotics, autonomous vehicles
    - **Imperial College London**: Hamlyn Centre for Robotic Surgery, aerial robotics, soft robotics
    - **University of Edinburgh**: Autonomous systems, AI for robotics
    - **University of Cambridge**: Bio-inspired robotics, machine intelligence
    - **University of Leeds**: Institute for Robotics, Autonomous Systems and Sensing (IRASS)
    - **King's College London**: Medical robotics, soft robotics
  - **Research Labs & Centres**:
    - **Bristol Robotics Laboratory (BRL)**: Largest UK academic robotics research centre
    - **Oxford Robotics Institute**: Spin-outs include Oxbotica
    - **Imperial College Hamlyn Centre**: Medical and surgical robotics
    - **RACE (Remote Applications in Challenging Environments)**: Offshore and hazardous environments robotics
  - **Companies**:
    - **Wayve** (London): Autonomous driving with end-to-end learning
    - **Ocado Technology**: Warehouse automation, robotic fulfilment centres
    - **Oxbotica** (Oxford): Autonomous vehicle software for various platforms
    - **CMR Surgical** (Cambridge): Versius surgical robot system
    - **Shadow Robot Company**: Dexterous robotic hands
    - **Tharsus Group**: Robotics and automation engineering

- Regional innovation hubs
  - **London**:
    - [Wayve: Autonomous driving startup with deep learning approach]
    - [King's College: Medical robotics research]
    - [Growing robotics startup ecosystem]
  - **Cambridge**:
    - [CMR Surgical: Medical robotics commercialisation]
    - [University robotics research: Bio-inspired, intelligent systems]
    - [Tech ecosystem: ARM (computing for robotics), multiple startups]
  - **Oxford**:
    - [Oxford Robotics Institute: Leading mobile robotics research]
    - [Oxbotica: Autonomous vehicle spin-out, universal autonomy software]
    - [Strong university-industry links]
  - **Bristol**:
    - [Bristol Robotics Laboratory: UK's largest academic robotics centre]
    - [Research strengths: Swarm robotics, soft robotics, assistive robotics]
    - [Aerospace robotics connections (Airbus, Rolls-Royce)]
  - **Edinburgh**:
    - [University robotics: Autonomous systems, social robotics]
    - [Scotland's leading robotics research hub]
  - **Leeds**:
    - [IRASS: Institute for Robotics, Autonomous Systems and Sensing]
    - [Autonomous vehicles research, surgical robotics]

- Regional case studies
  - [London case study: E.g., "Wayve's end-to-end learning approach to autonomous driving in complex urban environments"]
  - [Cambridge case study: E.g., "CMR Surgical's Versius robot—modular, portable surgical robot"]
  - [Oxford case study: E.g., "Oxbotica's universal autonomy deployed in mining, airports, and urban environments"]
  - [Bristol case study: E.g., "Bristol Robotics Laboratory's assistive robotics for elderly care"]

## Practical Implementation
- Technology stack and tools
  - **Robotics frameworks**: [[ROS]] / [[ROS 2]] (de facto standard), [[YARP]], custom frameworks
  - **Programming languages**: C++ (performance), Python (prototyping), MATLAB (simulation/control)
  - **Simulation**: [[Gazebo]], [[Isaac Sim]], [[Webots]], [[PyBullet]], [[MuJoCo]]
  - **Computer vision**: [[OpenCV]], [[PCL]] (Point Cloud Library), [[Open3D]]
  - **Machine learning**: [[PyTorch]], [[TensorFlow]], [[scikit-learn]] for perception and control
  - **Control systems**: MATLAB/Simulink, real-time operating systems (RTOS)

- Development workflow
  - **Mechanical design**: CAD (SolidWorks, Fusion 360), simulation (ANSYS for FEA)
  - **Electrical design**: Circuit design, PCB layout, embedded systems programming
  - **Software development**: Algorithm development, ROS package creation, testing in simulation
  - **Integration**: Hardware-software integration, sensor calibration, actuator tuning
  - **Testing**: Simulation testing, hardware-in-the-loop (HIL), field testing
  - **Deployment**: System installation, operator training, maintenance planning

- Best practices and patterns
  - **Modularity**: Separate perception, planning, control; use ROS nodes for modularity
  - **Simulation-first**: Develop and test in simulation before hardware deployment (sim-to-real)
  - **Safety by design**: Redundancy, fail-safe mechanisms, emergency stops, compliance with safety standards
  - **Robustness**: Sensor fusion for reliability, fault detection and diagnosis, graceful degradation
  - **Testing**: Rigorous testing in simulation, controlled environments, then real-world gradually
  - **Documentation**: Detailed system documentation, calibration procedures, maintenance guides

- Common challenges and solutions
  - **Challenge**: Sim-to-real gap (simulation doesn't match reality)
    - **Solution**: Domain randomisation, system identification, real-world data collection, sim-to-real transfer techniques
  - **Challenge**: Sensor noise and failures
    - **Solution**: Sensor fusion (Kalman filters, particle filters), redundancy, anomaly detection
  - **Challenge**: Real-time constraints
    - **Solution**: Optimised algorithms, real-time operating systems, hardware acceleration (GPUs, FPGAs)
  - **Challenge**: Safe human-robot interaction
    - **Solution**: Compliant actuators, force/torque sensing, predictive collision avoidance, safety-rated systems
  - **Challenge**: Long-term autonomy
    - **Solution**: Reliable perception, fault-tolerant planning, self-diagnosis, remote monitoring

- Case studies and examples
  - [Example 1: Industrial robot deployment—use case, integration, productivity gains]
  - [Example 2: Autonomous mobile robot in warehouse—navigation, fleet management, ROI]
  - [Example 3: Collaborative robot (cobot) in manufacturing—safety, flexibility, outcomes]
  - [Quantified outcomes: Productivity improvements, error reductions, cost savings, operational uptime]

## Research & Literature
- Key academic papers and sources
  1. [Foundational Paper] Brooks, R. A. (1986). "A Robust Layered Control System for a Mobile Robot". IEEE Journal of Robotics and Automation, 2(1), 14-23. [Annotation: Behaviour-based robotics, subsumption architecture.]
  2. [SLAM] Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press. [Annotation: Comprehensive textbook on probabilistic methods in robotics.]
  3. [Learning] Kober, J., Bagnell, J. A., & Peters, J. (2013). "Reinforcement Learning in Robotics: A Survey". IJRR, 32(11), 1238-1274. [Annotation: Survey of RL applications in robotics.]
  4. [Manipulation] Mason, M. T. (2001). *Mechanics of Robotic Manipulation*. MIT Press. [Annotation: Theory of grasping and manipulation.]
  5. [Autonomous Vehicles] Levinson, J. et al. (2011). "Towards Fully Autonomous Driving". IEEE IV. [Annotation: Stanford's autonomous vehicle approach.]
  6. [UK Contribution] Author, X. et al. (Year). "Title". Conference/Journal. DOI. [Annotation about UK robotics research—e.g., Oxford, Bristol, or Imperial work.]
  7. [Recent Advance] Author, Y. et al. (2024). "Title on learning-based control/sim-to-real/etc". Conference. DOI. [Annotation about current state of the art.]
  8. [Human-Robot Interaction] Goodrich, M. A., & Schultz, A. C. (2007). "Human-Robot Interaction: A Survey". Foundations and Trends in HRI, 1(3), 203-275. [Annotation: Comprehensive HRI survey.]

- Ongoing research directions
  - **Learning-based approaches**: End-to-end learning, model-based RL, imitation learning, sim-to-real transfer, meta-learning
  - **Dexterous manipulation**: Multi-fingered hands, tactile sensing, non-prehensile manipulation, contact-rich tasks
  - **Perception**: Robust object recognition, 3D scene understanding, semantic SLAM, active perception
  - **Multi-robot systems**: Distributed algorithms, swarm intelligence, multi-agent RL, task allocation
  - **Human-robot collaboration**: Intent prediction, shared autonomy, natural interfaces, trust and transparency
  - **Soft robotics**: Soft actuators, compliant structures, biomimetic design, safe physical interaction
  - **Long-term autonomy**: Lifelong learning, self-maintenance, energy management, adaptability

- Academic conferences and venues
  - **Premier robotics conferences**: ICRA (IEEE International Conference on Robotics and Automation), IROS (Intelligent Robots and Systems), RSS (Robotics: Science and Systems)
  - **AI and learning**: CoRL (Conference on Robot Learning), NeurIPS, ICML (for learning-based robotics)
  - **Domain-specific**: IV (Intelligent Vehicles), Humanoids, ICARCV (Control, Automation, Robotics and Vision)
  - **UK venues**: UKRAS Conference (UK Robotics and Autonomous Systems), TAROS (Towards Autonomous Robotic Systems)
  - **Key journals**: International Journal of Robotics Research (IJRR), IEEE Transactions on Robotics, Autonomous Robots

## Future Directions
- Emerging trends and developments
  - **Embodied AI**: Integration of large language models with robotic control for general-purpose robots
  - **Soft robotics**: Compliant, adaptable robots for unstructured environments and safe human interaction
  - **Swarm and distributed robotics**: Coordinated multi-robot systems for large-scale tasks
  - **Humanoid resurgence**: Advanced humanoid robots with dexterous manipulation (Tesla Optimus, Figure, etc.)
  - **Autonomous everywhere**: Expansion beyond vehicles—drones, delivery robots, agricultural robots, inspection robots
  - **Cloud robotics**: Offloading computation, shared learning, fleet coordination
  - **Brain-computer interfaces**: Direct neural control of robotic systems

- Anticipated challenges
  - **Technical challenges**:
    - Dexterity gap: Achieving human-level manipulation remains elusive
    - Robustness: Reliable performance in unstructured, dynamic environments
    - Energy and autonomy: Battery life, recharging infrastructure for mobile robots
    - Real-time perception and decision-making: Latency, computational limits
  - **Safety and reliability**: Ensuring safe operation around humans, fail-safe mechanisms, regulatory compliance
  - **Ethical and social**:
    - Job displacement: Automation impact on employment in manufacturing, logistics, services
    - Privacy: Surveillance concerns with camera-equipped robots
    - Autonomy and accountability: Who is responsible when autonomous systems fail?
  - **Regulatory**: Evolving regulations for autonomous vehicles, drones, medical robots, public spaces

- Research priorities
  - General-purpose manipulation and dexterity
  - Robust and adaptive perception
  - Safe and intuitive human-robot interaction
  - Long-term autonomy and self-sufficiency
  - Scalable multi-robot coordination
  - Trustworthy and explainable autonomous systems

- Predicted impact (2025-2030)
  - **Manufacturing**: Further automation, flexible and reconfigurable production, cobots widespread
  - **Logistics**: Autonomous warehouses, delivery robots, drone delivery, autonomous trucks
  - **Healthcare**: Surgical robots, rehabilitation robots, eldercare assistance, telepresence
  - **Agriculture**: Autonomous tractors, harvesting robots, precision agriculture, reduced labour demands
  - **Transportation**: Autonomous vehicles (shuttles, buses, trucks), urban air mobility (drones, flying taxis)
  - **Service**: Hospitality robots, cleaning robots, security and inspection, customer service

## References
1. [Citation 1 - Foundational work (e.g., Brooks subsumption architecture)]
2. [Citation 2 - Probabilistic robotics textbook]
3. [Citation 3 - Reinforcement learning in robotics survey]
4. [Citation 4 - Manipulation theory]
5. [Citation 5 - Autonomous vehicles]
6. [Citation 6 - UK robotics research]
7. [Citation 7 - Recent learning-based robotics]
8. [Citation 8 - HRI survey]
9. [Citation 9 - ROS documentation or standard]
10. [Citation 10 - Additional relevant source]

## Metadata
- **Last Updated**: YYYY-MM-DD
- **Review Status**: [Initial Draft | Comprehensive Editorial Review | Expert Reviewed]
- **Content Quality**: [High | Medium | Requires Enhancement]
- **Completeness**: [100% | 80% | 60% | Stub]
- **Verification**: Academic sources and technical details verified
- **Regional Context**: UK robotics hubs (Bristol, Oxford, Cambridge, London) where applicable
- **Curator**: Robotics Research Team
- **Version**: 1.0.0
- **Domain**: Robotics & Autonomous Systems
```

---

## Robotics-Specific Guidelines

### Technical Depth
- Explain physical design, sensors, and actuators in detail
- Describe control architectures and autonomy levels
- Discuss safety systems and standards compliance
- Include performance metrics (accuracy, speed, payload, autonomy duration)
- Address human-robot interaction considerations

### Linking Strategy
- Link to foundational robotics concepts ([[Robot]], [[Autonomous Systems]], [[Mobile Robotics]])
- Link to specific components ([[Sensor]], [[Actuator]], [[LiDAR]], [[IMU]])
- Link to control methods ([[SLAM]], [[Path Planning]], [[Motion Planning]])
- Link to frameworks and tools ([[ROS]], [[Gazebo]], [[OpenCV]])
- Link to application domains ([[Manufacturing Automation]], [[Autonomous Vehicles]])

### UK Robotics Context
- Emphasise UK research centres (Bristol Robotics Laboratory, Oxford Robotics Institute)
- Highlight UK companies (Wayve, Ocado, Oxbotica, CMR Surgical, Shadow Robot)
- Note regional strengths (Bristol for largest lab, Oxford for mobile robotics, Cambridge for medical)
- Include UK autonomous vehicle initiatives and regulations

### Common Robotics Sections
- Physical Architecture and Design (for robot systems)
- Sensing and Perception (for autonomous systems)
- Control Systems and Autonomy (for intelligent robots)
- Safety and Reliability (for all robots, especially those interacting with humans)
- Human-Robot Interaction (for service and collaborative robots)

---

**Template Version:** 1.0.0
**Last Updated:** 2025-11-21
**Status:** Ready for Use

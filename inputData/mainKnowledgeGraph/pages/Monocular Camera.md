- ### OntologyBlock
    - term-id:: RB-0160
    - preferred-term:: Monocular Camera
    - ontology:: true
    - is-subclass-of:: [[Camera]]
    - version:: 1.0.0

## Monocular Camera

Monocular Camera employs a single lens and image sensor to capture two-dimensional intensity and color information, representing the simplest and most computationally efficient vision system for robotics. While lacking direct depth perception available from stereo or structured light systems, monocular cameras enable numerous critical capabilities including object detection, classification, tracking, visual servoing, and visual odometry through sophisticated image processing and computer vision algorithms. Their low cost, compact size, minimal power consumption, and simple calibration make them ubiquitous in mobile, service, and collaborative robots.

Technical specifications include resolution (VGA 640×480 to 4K 3840×2160), frame rate (30-120 FPS typical), sensor type (CMOS or CCD), interface (USB, GigE, CSI), field of view (60-180° depending on lens), and dynamic range (8-12 bits per channel). Global shutter sensors capture entire frames simultaneously, eliminating motion blur critical for fast-moving robots, while rolling shutter variants offer lower cost for stationary or slow applications. Autofocus mechanisms enable operation across varying distances, while fixed-focus designs simplify integration for known working ranges.

Depth estimation from monocular images employs structure from motion (SfM) algorithms triangulating features across multiple viewpoints as camera moves, or learning-based approaches using deep neural networks trained on RGB-D datasets to infer depth from single images. Accuracy remains inferior to stereo systems but sufficient for many applications. Visual odometry tracks camera motion by matching features across consecutive frames, enabling robot localization. Visual SLAM (Simultaneous Localization and Mapping) builds environment maps while tracking position. Marker-based pose estimation uses ArUco, AprilTag, or QR codes for precise 6-DOF localization.

As of 2024-2025, monocular cameras integrate AI accelerators enabling 30+ FPS object detection (YOLO, EfficientDet) on embedded platforms like NVIDIA Jetson or Raspberry Pi. Applications include warehouse robots using overhead cameras for localization, collaborative robots with wrist-mounted cameras for part recognition, service robots performing gesture recognition, and mobile robots executing semantic navigation. UK's Oxford Robotics Institute advances monocular vision-based navigation. Compliance with machine vision standards including EMVA 1288 (sensor characterization) and GigE Vision protocol ensures interoperability. Privacy concerns drive edge processing per GDPR requirements rather than cloud-based vision processing.

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable

## Technical Details

- **Id**: monocularcamera-ontology
- **Collapsed**: true
- **Domain Prefix**: RB
- **Sequence Number**: 0160
- **Filename History**: ["RB-0160-monocularcamera.md"]
- **Public Access**: true
- **Source Domain**: metaverse
- **Status**: complete
- **Last Updated**: 2025-11-13
- **Maturity**: established
- **Source**: Chimera Prime Research
- **Authority Score**: 0.95
- **Owl:Class**: rb:MonocularCamera
- **Belongstodomain**: [[Robotics]]
- **Is Subclass Of**: [[Camera]]

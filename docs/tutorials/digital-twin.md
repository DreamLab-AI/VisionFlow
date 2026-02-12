---
title: "Tutorial: Building a Digital Twin with VisionFlow"
description: Connect live sensor data to a VisionFlow entity graph and visualise your physical system as a real-time 3D digital twin.
category: tutorial
tags:
  - digital-twin
  - sensors
  - tutorial
  - neo4j
  - real-time
  - websocket
updated-date: 2026-02-12
difficulty-level: intermediate
---

# Tutorial: Building a Digital Twin with VisionFlow

This tutorial guides you through connecting sensor data sources to VisionFlow,
modelling physical assets as a knowledge graph in Neo4j, and viewing the result
as a live 3D digital twin that updates in real time.

## Prerequisites

- VisionFlow stack running (`docker compose --profile dev up -d`).
- Familiarity with the [First Graph tutorial](first-graph.md).
- A sensor data source -- this tutorial uses a bundled CSV simulator, but the
  same steps apply to OPC UA, Modbus TCP, or MQTT feeds.

## What You Will Build

By the end of this tutorial you will have:

1. An entity graph in Neo4j representing a small production line (5 stations,
   12 sensors).
2. A bridge service that streams simulated telemetry into VisionFlow over the
   binary WebSocket protocol.
3. A 3D view where sensor readings animate node colour and size in real time.

## Step 1 -- Model the Physical Assets

Create the asset graph through the VisionFlow API. Open a terminal and run:

```bash
# Create station nodes
for i in 1 2 3 4 5; do
  curl -s -X POST http://localhost:3030/api/nodes \
    -H "Content-Type: application/json" \
    -d "{\"label\": \"Station_$i\", \"type\": \"Station\"}"
done

# Link stations in sequence
for i in 1 2 3 4; do
  next=$((i + 1))
  curl -s -X POST http://localhost:3030/api/edges \
    -H "Content-Type: application/json" \
    -d "{\"source\": \"Station_$i\", \"target\": \"Station_$next\", \"type\": \"FEEDS_INTO\"}"
done
```

Open the browser at **http://localhost:3030**. You should see five nodes
arranged in a line by the physics engine, connected by `FEEDS_INTO` edges.

## Step 2 -- Attach Sensor Nodes

Each station has sensors for temperature, vibration, and (on some stations)
torque. Add them as child nodes:

```bash
for i in 1 2 3 4 5; do
  curl -s -X POST http://localhost:3030/api/nodes \
    -H "Content-Type: application/json" \
    -d "{\"label\": \"Temp_$i\", \"type\": \"Sensor\", \"unit\": \"C\"}"
  curl -s -X POST http://localhost:3030/api/edges \
    -H "Content-Type: application/json" \
    -d "{\"source\": \"Station_$i\", \"target\": \"Temp_$i\", \"type\": \"HAS_SENSOR\"}"
done
```

Repeat for vibration sensors. In the 3D view, sensor nodes cluster tightly
around their parent station thanks to the semantic spring forces.

## Step 3 -- Stream Simulated Telemetry

VisionFlow ships with a Python-based sensor simulator for development. Start it
with:

```bash
python scripts/sensor-simulator.py \
  --ws ws://localhost:3030/ws/binary \
  --nodes Temp_1,Temp_2,Temp_3,Temp_4,Temp_5 \
  --interval-ms 100
```

The simulator generates sinusoidal temperature curves with random noise and
pushes values using the 34-byte binary WebSocket frame format. In the browser
you will see sensor node colours shift from blue (cool) to red (hot) as values
change.

## Step 4 -- Configure Visual Mappings

Open the **Visual Settings** panel on the right side of the UI:

1. Under **Node Colour Mapping**, select the `value` property and choose a
   gradient from blue to red.
2. Under **Node Size Mapping**, bind size to the `value` property so that
   higher readings produce larger nodes.
3. Under **Edge Glow**, enable glow on `FEEDS_INTO` edges to trace the
   production flow visually.

These mappings are stored per-session and can be saved as a preset for your
team.

## Step 5 -- Set Alarm Thresholds

VisionFlow's physics engine can make anomalies self-evident. In the **Physics**
panel:

1. Enable **Alarm Magnification**. When a sensor value exceeds the threshold
   you configure (e.g., 80 degrees C), the node's repulsion radius increases,
   pushing it away from its cluster.
2. The displaced node immediately draws the operator's eye -- no separate
   alerting system required.

You can verify by editing the simulator to inject a spike:

```bash
python scripts/sensor-simulator.py --spike-node Temp_3 --spike-value 95
```

Watch `Temp_3` push outward from `Station_3` in the 3D view.

## Step 6 -- Query the Twin in Neo4j

Because all state is persisted, you can run Cypher queries for post-shift
analysis:

```cypher
MATCH (s:Sensor)-[:HAS_SENSOR]-(st:Station)
WHERE s.value > 80
RETURN st.label AS station, s.label AS sensor, s.value AS reading
ORDER BY s.value DESC
```

## Connecting Real Sensors

To move beyond the simulator, replace the Python script with a gateway that
reads from your sensor bus (OPC UA, Modbus TCP, MQTT) and writes to the binary
WebSocket endpoint. The protocol specification is documented in
[Binary WebSocket Protocol](../diagrams/infrastructure/websocket/binary-protocol-complete.md).

## Next Steps

- [Case Study: Digital Twin Manufacturing](../use-cases/case-studies/manufacturing-digital-twin.md)
  -- A production deployment of this pattern.
- [GPU Acceleration Concepts](../explanation/concepts/gpu-acceleration.md) --
  How VisionFlow keeps the twin responsive at scale.
- [Architecture Overview](../explanation/architecture/README.md) -- Understand
  the full system stack.

---

**Document Version**: 1.0
**Last Updated**: 2026-02-12
**Maintained By**: VisionFlow Documentation Team

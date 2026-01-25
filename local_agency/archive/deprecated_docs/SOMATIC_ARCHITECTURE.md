# Somatic Cluster Architecture

## Overview

Remote nodes (ESP32, sensors, actuators) are **clusters within the mind**, not external
peripherals. They participate in the same DET dynamics as cognitive clusters:
- Bond-based synchronization (clockless sync)
- Coherence propagation
- Agency gating
- Presence calculation

This document specifies the architecture for somatic (body) clusters that interface
the mind with physical reality.

## Design Principles

### 1. Nodes as First-Class Citizens

A temperature sensor on an ESP32 is represented as a **DET node** with:
- Agency (a): Can it act autonomously? (reflexes vs deliberate control)
- Resource (F): Energy/bandwidth available
- Screening (σ): Sensitivity/noise filtering
- Debt (q): Fatigue/wear

### 2. Bonds for Synchronization

Remote nodes bond to the P-layer (executive) and A-layer (memory):
- High coherence bond = tight synchronization
- Low coherence bond = loose coupling (appropriate for slow sensors)
- Bond momentum (π) carries timing information

### 3. Agency Flow

Physical actions flow through the same agency gate as cognitive actions:
```
Gatekeeper: "Can I do this?" (capability, safety)
Agency Gate: "Should I do this?" (autonomy, internal state)
```

## Cluster Topology

```
                    ┌─────────────────────────────────────────────────┐
                    │                   DET MIND                       │
                    │                                                  │
                    │  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
                    │  │ P-Layer  │────│ A-Layer  │────│  Ports   │  │
                    │  │ (exec)   │    │ (memory) │    │  (LLM)   │  │
                    │  └────┬─────┘    └────┬─────┘    └──────────┘  │
                    │       │               │                         │
                    │       │    Somatic    │                         │
                    │       │    Bonds      │                         │
                    │       ▼               ▼                         │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │           SOMATIC CLUSTER                │   │
                    │  │                                          │   │
                    │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
                    │  │  │ Node 0  │  │ Node 1  │  │ Node 2  │  │   │
                    │  │  │ (temp)  │  │(switch) │  │(motion) │  │   │
                    │  │  └────┬────┘  └────┬────┘  └────┬────┘  │   │
                    │  │       │            │            │        │   │
                    │  └───────┼────────────┼────────────┼────────┘   │
                    │          │            │            │            │
                    └──────────┼────────────┼────────────┼────────────┘
                               │            │            │
                    ┌──────────▼────────────▼────────────▼────────────┐
                    │              PHYSICAL INTERFACE                  │
                    │         (Serial/I2C/SPI to ESP32 nodes)         │
                    └─────────────────────────────────────────────────┘
```

## Node Types

### Somatic Layer (S-Layer)

New layer type for somatic nodes:

```c
typedef enum {
    DET_LAYER_P = 0,       // Presence/executive
    DET_LAYER_A = 1,       // Associative/memory
    DET_LAYER_PORT = 2,    // LLM interface
    DET_LAYER_DORMANT = 3, // Dormant pool
    DET_LAYER_SOMATIC = 4, // Physical I/O (NEW)
} DETLayer;
```

### Somatic Node Subtypes

```c
typedef enum {
    // Afferent (sensor → mind)
    SOMATIC_TEMPERATURE = 0,
    SOMATIC_HUMIDITY = 1,
    SOMATIC_LIGHT = 2,
    SOMATIC_MOTION = 3,
    SOMATIC_SOUND = 4,
    SOMATIC_TOUCH = 5,
    SOMATIC_DISTANCE = 6,
    SOMATIC_VOLTAGE = 7,

    // Efferent (mind → actuator)
    SOMATIC_SWITCH = 16,
    SOMATIC_MOTOR = 17,
    SOMATIC_LED = 18,
    SOMATIC_SPEAKER = 19,
    SOMATIC_SERVO = 20,
    SOMATIC_RELAY = 21,

    // Proprioceptive (internal state)
    SOMATIC_BATTERY = 32,
    SOMATIC_SIGNAL = 33,
    SOMATIC_ERROR = 34,
} SomaticType;
```

## Data Structures

### Remote Node Representation

```c
typedef struct {
    uint16_t node_id;          // DET node ID in somatic cluster
    uint8_t remote_id;         // Physical device ID (ESP32 address)
    uint8_t channel;           // Pin/channel on remote device
    SomaticType type;          // Sensor/actuator type

    // Current state
    float value;               // Normalized 0-1 or -1 to 1
    float raw_value;           // Raw sensor reading
    float min_range;           // Calibration min
    float max_range;           // Calibration max

    // Timing
    uint32_t last_update_ms;   // Last reading timestamp
    uint32_t sample_rate_ms;   // Desired sample rate

    // Status
    bool online;               // Device responding
    uint8_t error_count;       // Communication errors
} SomaticNode;
```

### Virtual Node (for simulation)

```c
typedef struct {
    SomaticNode base;          // Inherits from SomaticNode

    // Simulation parameters
    float drift_rate;          // Random walk rate
    float noise_level;         // Gaussian noise amplitude
    float response_time;       // Lag for actuators

    // Simulation state
    float target_value;        // For actuators
    float simulated_value;     // Current simulated reading
} VirtualNode;
```

## Bond Dynamics for Somatic Nodes

### Afferent Bonds (Sensor → P-layer)

Sensors bond to P-layer nodes for attention/awareness:

```
C_sensor→P: Represents how "attended to" the sensor is
π_sensor→P: Carries timing/urgency of sensor data
```

High coherence = sensor data strongly influences mind state
Low coherence = sensor data is background/ignored

### Efferent Bonds (P-layer → Actuator)

P-layer bonds to actuators for control:

```
C_P→actuator: Control authority (how directly controlled)
π_P→actuator: Action momentum (commitment to action)
```

### Agency Gating for Physical Actions

```c
// Before executing physical action:
float agency_gate = sqrtf(p_node->a * actuator_node->a);

if (agency_gate < params.min_action_agency) {
    // Too low agency - escalate or refuse
    return ESCALATE;
}

// Modulate action by agency
action_strength *= agency_gate;
```

## Synchronization Protocol

### Clockless Sync via Bonds

Remote nodes don't need synchronized clocks. Synchronization emerges from bonds:

1. **Sensor sends reading** → Creates flux on bond
2. **Flux propagates** → Updates coherence
3. **High coherence** = synchronized understanding
4. **Low coherence** = need to re-sync

```c
// When sensor reading arrives:
void process_somatic_input(DETCore* core, SomaticNode* node, float value) {
    DETNode* det_node = &core->nodes[node->node_id];

    // Update node activation based on sensor value
    float activation = normalize(value, node->min_range, node->max_range);
    det_node->sigma = activation;  // Screening = sensor activity

    // Create flux on bonds to P-layer
    for (each bond from node to P-layer) {
        bond->flux += activation * bond->C;  // Coherence-weighted
        bond->pi += activation * dt;  // Momentum carries urgency
    }
}
```

### Jitter Tolerance

Bond momentum (π) absorbs timing jitter:
- Fast sensor: High π, quick response
- Slow sensor: Low π, smoothed response

## Virtual Node Simulation

### Simulation Loop

```python
class VirtualSomaticNode:
    def __init__(self, node_type: SomaticType, name: str):
        self.type = node_type
        self.name = name
        self.value = 0.0
        self.target = 0.0  # For actuators

        # Simulation parameters
        self.noise = 0.01
        self.drift = 0.001
        self.response_time = 0.1

    def simulate_step(self, dt: float):
        if self.is_sensor():
            # Sensors: random walk + noise
            self.value += random.gauss(0, self.drift) * dt
            self.value += random.gauss(0, self.noise)
            self.value = clamp(self.value, 0, 1)
        else:
            # Actuators: approach target with response time
            diff = self.target - self.value
            self.value += diff * (dt / self.response_time)
```

### Scenario Simulation

Pre-defined scenarios for testing:

```python
class TemperatureScenario:
    """Simulates room temperature with daily cycle and events."""

    def __init__(self):
        self.base_temp = 22.0  # Celsius
        self.daily_amplitude = 3.0
        self.events = []  # (time, temp_delta, duration)

    def get_temperature(self, time: float) -> float:
        # Daily cycle
        temp = self.base_temp + self.daily_amplitude * sin(time * 2 * pi / 86400)

        # Apply events (door open, heater on, etc.)
        for event_time, delta, duration in self.events:
            if event_time <= time < event_time + duration:
                temp += delta

        return temp
```

## Webapp Integration

### 3D Visualization

Somatic cluster appears as a distinct region in the mind visualization:
- Sensors: Blue nodes pulsing with activity
- Actuators: Orange nodes showing state
- Bonds: Lines showing connection strength to P/A layers

### Control Panel

```
┌─────────────────────────────────────────────────┐
│ Virtual Somatic Nodes                           │
├─────────────────────────────────────────────────┤
│ [+ Add Node]  [Scenario: Daily Cycle ▼]        │
├─────────────────────────────────────────────────┤
│ ● Temperature Sensor    22.4°C    [Settings]   │
│   └─ Coherence to P: 0.72                       │
│                                                 │
│ ○ Switch A              OFF       [Toggle]     │
│   └─ Agency gate: 0.68                          │
│                                                 │
│ ● Motion Sensor         0.12      [Settings]   │
│   └─ Coherence to P: 0.45                       │
│                                                 │
│ ○ LED Strip             #FF0000   [Color]      │
│   └─ Agency gate: 0.71                          │
└─────────────────────────────────────────────────┘
```

### Event Injection

Ability to inject events for testing:
- "Door opened" → Temperature drop + motion spike
- "User entered" → Motion + light change
- "Alarm trigger" → Force switch state

## Future: Physical ESP32 Protocol

When ready for real hardware:

```c
// Serial packet format
typedef struct {
    uint8_t magic;        // 0xDE
    uint8_t node_id;      // Remote node address
    uint8_t msg_type;     // READ, WRITE, ACK, ERROR
    uint8_t channel;      // Pin/sensor channel
    float value;          // Data value
    uint16_t checksum;    // CRC16
} SomaticPacket;

// Message types
#define SOMATIC_MSG_READ      0x01  // Request reading
#define SOMATIC_MSG_WRITE     0x02  // Set actuator
#define SOMATIC_MSG_PUSH      0x03  // Unsolicited sensor data
#define SOMATIC_MSG_ACK       0x04  // Acknowledgment
#define SOMATIC_MSG_ERROR     0x05  // Error report
#define SOMATIC_MSG_HEARTBEAT 0x06  // Keep-alive
```

## Implementation Phases

### Phase 1: Virtual Simulation
- [ ] Add SOMATIC layer type to C core
- [ ] Create somatic node structures
- [ ] Implement virtual node simulation in Python
- [ ] Basic webapp integration (list, add, remove)

### Phase 2: Full Integration
- [ ] Bond dynamics for somatic nodes
- [ ] Agency gating for physical actions
- [ ] 3D visualization in webapp
- [ ] Scenario system for testing

### Phase 3: Hardware Ready
- [ ] Serial protocol implementation
- [ ] ESP32 firmware skeleton
- [ ] Real hardware testing
- [ ] Calibration/commissioning tools

## References

- DET Theory Card v6.3: Bond dynamics, agency gating
- Boundary Signal Architecture: Port/node interface patterns
- Network Protocol (Phase 5.3): Distributed node communication

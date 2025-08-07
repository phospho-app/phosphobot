# SO-100 Robotic Arm Simulation

A robust MuJoCo simulation of the SO-100 robotic arm with interactive GUI and real robot mirroring capabilities.

## Features

- **Real-time 3D visualization** with MuJoCo viewer
- **Robust threading** - no more viewer conflicts during collisions
- **Interactive banana object** for manipulation testing
- **6-DOF robotic arm** simulation with gripper
- **Real robot mirroring** support with automatic calibration
- **Thread-safe operations** with enhanced error handling
- **Interactive configuration** for robot endpoint setup

## Requirements

```bash
pip install mujoco numpy requests
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Quick Start

### Setup Requirements
- Python 3.8+ with MuJoCo dependencies
- SO-100 robot connected and accessible via HTTP API
- Robot endpoint URL configured (default: `http://192.168.178.190:80`)

### Direct Launch
Run the integrated simulation with the MuJoCo viewer:

```bash
python main.py
```

This starts the SO-100 arm simulation, opens the viewer, and prompts for robot endpoint configuration.

### Interactive Configuration

When you run the script, you'll be prompted to configure:

1. **Robot Endpoint**: Enter the robot's IP address and port (or press Enter for default)
   - Default: `http://192.168.178.190:80`
   - Example: `http://192.168.178.100:8080`

The system will then:
- Initialize the MuJoCo simulation
- Test connection to the physical robot
- Start real-time mirroring of robot movements

### Use in Your Code

```python
from so100_simulation import SO100Simulation

# Create simulation
sim = SO100Simulation()

# Set joint angles
sim.set_joint_angles([0.0, -1.0, 1.0, 0.5, -0.5, 0.0])

# Get end effector position
position = sim.get_end_effector_position()
print(f"End effector at: {position}")

# Close simulation
sim.close()
```

## File Structure

```
08-mujoco-simulation/
├── so100_simulation.py      # Main simulation class
├── main.py                  # Interactive simulation + robot mirroring
├── so100_arm.xml            # MuJoCo model definition
├── rgbd_feed.py             # Record3D RGB-D camera helper
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── assets/                  # 3D model files, textures, etc.
```

## Available Commands

Once the simulation is running, you can use these commands:

- **`mirror`** - Start real-time mirroring from physical robot
- **`stop` or `s`** - Stop mirroring
- **`current`** - Show current pose and joint angles
- **`workspace`** - Show workspace limits
- **`commands`** - Show help
- **`q`** - Quit

## Robot Mirroring

The simulation can mirror movements from a physical SO-100 robot:

1. **Automatic Calibration**: Joint offsets are automatically applied
2. **Real-time Updates**: Robot movements are mirrored at 5Hz
3. **Error Handling**: Robust error handling for connection issues
4. **Thread Safety**: Thread-safe operations prevent conflicts

### Joint Calibration

The system uses hardcoded joint calibration:
- Joint 1 (Pitch): -90° offset
- Joint 2 (Elbow): +90° offset  
- Joint 4 (Wrist_Roll): -180° offset

## Workspace Limits

The SO-100 workspace limits:
- **X (left/right)**: -0.338 to +0.338 meters (±34cm)
- **Y (forward)**: -0.383 to -0.045 meters (4.5-38cm forward)
- **Z (up/down)**: 0.096 to 0.315 meters (10-32cm high)

**Note**: Y must be NEGATIVE (negative = forward from base)

## Troubleshooting

### Common Issues

1. **MuJoCo Viewer Issues**
   - Install MuJoCo: `pip install mujoco`
   - On macOS, use `mjpython` instead of `python`

2. **Robot Connection Issues**
   - Verify robot is powered on and accessible
   - Check IP address and port configuration
   - Ensure no firewall is blocking connection

3. **Model Loading Issues**
   - Verify `so100_arm.xml` exists in current directory
   - Check `assets/` folder contains required 3D models

### Error Messages

- **"Cannot connect to physical robot"**: Check robot IP and network connectivity
- **"Viewer initialization failed"**: Try using `mjpython` on macOS
- **"Error loading model"**: Verify XML file and assets are present

## Dependencies

- **mujoco**: Physics engine and viewer
- **numpy**: Numerical computations
- **requests**: HTTP communication with robot
- **threading**: Thread-safe operations
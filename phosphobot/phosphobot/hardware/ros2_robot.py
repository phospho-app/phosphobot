import asyncio
import threading
import time
from typing import List, Optional

import numpy as np
from loguru import logger

from phosphobot.hardware.base import BaseManipulator
from phosphobot.models.robot import BaseRobotConfig, BaseRobotPIDGains
from phosphobot.utils import get_resources_path

# ROS2 imports with graceful fallback for testing
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, Header
    from geometry_msgs.msg import PoseStamped
    ROS2_AVAILABLE = True
except ImportError:
    logger.warning("ROS2 packages not available. ROS2Robot will run in mock mode for testing.")
    ROS2_AVAILABLE = False
    
    # Mock classes for testing without ROS2
    class Node:
        def __init__(self, name): pass
        def create_subscription(self, *args, **kwargs): return None
        def create_publisher(self, *args, **kwargs): return MockPublisher()
        def destroy_node(self): pass
    
    class SingleThreadedExecutor:
        def __init__(self): pass
        def add_node(self, node): pass
        def spin(self): 
            while True:
                time.sleep(0.1)
        def shutdown(self): pass
    
    class JointState:
        def __init__(self):
            self.position = []
            self.velocity = []
            self.effort = []
    
    class Float64MultiArray:
        def __init__(self):
            self.data = []
    
    class Header:
        def __init__(self):
            self.stamp = None
    
    class MockPublisher:
        def publish(self, msg): pass
    
    class MockRclpy:
        def init(self): pass
        def ok(self): return True
        def shutdown(self): pass
    
    rclpy = MockRclpy()


class ROS2Robot(BaseManipulator):
    """
    ROS2 Robot hardware interface for phosphobot.
    Connects to any robot that implements standard ROS2 joint control interface.
    """
    
    name = "ros2-robot"
    
    # Required class attributes for BaseManipulator
    URDF_FILE_PATH = str(get_resources_path() / "urdf" / "ros2_generic" / "robot.urdf")
    AXIS_ORIENTATION = [0, 0, 1, 1]  # Default orientation
    SERIAL_ID = "ros2_connection"
    SERVO_IDS = [1, 2, 3, 4, 5, 6]  # Default 6-DOF arm
    CALIBRATION_POSITION = [0.0, -np.pi/4, np.pi/2, 0.0, np.pi/4, 0.0]  # Safe calibration pose
    SLEEP_POSITION = [0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0]  # Safe sleep pose
    RESOLUTION = 4096  # Default resolution (will be overridden by ROS2 joint limits)
    END_EFFECTOR_LINK_INDEX = 5  # Last joint before gripper
    GRIPPER_JOINT_INDEX = 6  # Gripper joint index
    
    def __init__(
        self, 
        namespace: str = "", 
        joint_states_topic: str = "/joint_states",
        joint_commands_topic: str = "/joint_commands", 
        num_joints: int = 6,
        joint_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize ROS2 robot connection.
        
        Args:
            namespace: ROS2 namespace for the robot (e.g., "/my_robot")
            joint_states_topic: Topic name for joint state feedback
            joint_commands_topic: Topic name for joint commands  
            num_joints: Number of joints in the robot
            joint_names: List of joint names (optional)
        """
        self.namespace = namespace
        self.joint_states_topic = joint_states_topic
        self.joint_commands_topic = joint_commands_topic
        self.num_joints = num_joints
        self.joint_names = joint_names or [f"joint_{i}" for i in range(num_joints)]
        
        # Update SERVO_IDS based on num_joints
        self.SERVO_IDS = list(range(1, num_joints + 1))
        
        # ROS2 state
        self._node = None
        self._executor = None
        self._ros_thread = None
        self._joint_state_sub = None
        self._joint_cmd_pub = None
        self._current_joint_states = JointState()
        self._current_positions = np.zeros(num_joints)
        self._current_velocities = np.zeros(num_joints) 
        self._current_efforts = np.zeros(num_joints)
        self._shutdown_event = threading.Event()
        
        # Robot state
        self.is_connected = False
        self.is_moving = False
        self.device_name = f"ros2://{namespace}"
        
        # Initialize default config
        self.config = BaseRobotConfig(
            name=self.name,
            servos_voltage=12.0,  # Default voltage
            servos_offsets=[2048.0] * num_joints,
            servos_calibration_position=self.CALIBRATION_POSITION[:num_joints],
            servos_offsets_signs=[1.0] * num_joints,
            pid_gains=[BaseRobotPIDGains(p_gain=32, i_gain=0, d_gain=32) for _ in range(num_joints)]
        )
        
        # Call parent constructor
        super().__init__(device_name=self.device_name, **kwargs)
    
    async def connect(self) -> None:
        """Initialize ROS2 connection and start spinning."""
        try:
            if not ROS2_AVAILABLE:
                logger.info("ROS2 not available, running in mock mode")
                await asyncio.sleep(0.1)  # Simulate connection delay
                self.is_connected = True
                return
                
            # Initialize ROS2 if not already done
            if not rclpy.ok():
                rclpy.init()
            
            # Create ROS2 node
            node_name = f'phosphobot_client_{id(self)}'
            self._node = Node(node_name)
            logger.info(f"Created ROS2 node: {node_name}")
            
            # Create subscribers and publishers
            self._joint_state_sub = self._node.create_subscription(
                JointState,
                self.joint_states_topic,
                self._joint_state_callback,
                10
            )
            logger.info(f"Subscribed to joint states: {self.joint_states_topic}")
            
            self._joint_cmd_pub = self._node.create_publisher(
                Float64MultiArray,
                self.joint_commands_topic,
                10
            )
            logger.info(f"Publishing joint commands to: {self.joint_commands_topic}")
            
            # Start ROS2 spinning in separate thread
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            
            self._ros_thread = threading.Thread(
                target=self._spin_thread, 
                daemon=True
            )
            self._ros_thread.start()
            
            # Wait for initial connection
            await asyncio.sleep(2.0)
            
            self.is_connected = True
            logger.success(f"Connected to ROS2 robot on namespace: {self.namespace}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ROS2 robot: {e}")
            raise e
    
    def _spin_thread(self):
        """ROS2 spinning thread."""
        try:
            while not self._shutdown_event.is_set() and rclpy.ok():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            logger.error(f"ROS2 spin thread error: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Handle incoming joint state messages."""
        self._current_joint_states = msg
        if msg.position:
            # Ensure we have the right number of positions
            positions = np.array(msg.position[:self.num_joints])
            if len(positions) == self.num_joints:
                self._current_positions = positions
        if msg.velocity:
            velocities = np.array(msg.velocity[:self.num_joints])
            if len(velocities) == self.num_joints:
                self._current_velocities = velocities
        if msg.effort:
            efforts = np.array(msg.effort[:self.num_joints])
            if len(efforts) == self.num_joints:
                self._current_efforts = efforts
    
    def disconnect(self) -> None:
        """Clean up ROS2 connection."""
        try:
            self._shutdown_event.set()
            
            if self._executor:
                self._executor.shutdown()
            if self._node:
                self._node.destroy_node()
            if self._ros_thread and self._ros_thread.is_alive():
                self._ros_thread.join(timeout=2.0)
                
            self.is_connected = False
            logger.info("Disconnected from ROS2 robot")
        except Exception as e:
            logger.warning(f"Error during ROS2 disconnect: {e}")
    
    # BaseRobot required methods
    def set_motors_positions(self, positions: np.ndarray, enable_gripper: bool = False) -> None:
        """Send joint position commands via ROS2."""
        if not self.is_connected:
            logger.warning("Robot not connected, cannot send commands")
            return
            
        try:
            # Ensure positions array has correct size
            if len(positions) != self.num_joints:
                logger.warning(f"Position array size {len(positions)} doesn't match num_joints {self.num_joints}")
                return
            
            if not ROS2_AVAILABLE:
                # Mock mode - just update internal state
                self._current_positions = positions.copy()
                self.is_moving = True
                return
                
            cmd_msg = Float64MultiArray()
            cmd_msg.data = positions.tolist()
            self._joint_cmd_pub.publish(cmd_msg)
            self.is_moving = True
            
        except Exception as e:
            logger.error(f"Failed to send joint commands: {e}")
    
    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current robot state for observations."""
        # Return pose (x, y, z, qx, qy, qz, qw) and joint positions
        # For manipulators, pose could be computed from forward kinematics
        # For now, return identity pose and current joint positions
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        joints = self._current_positions.copy()
        return pose, joints
    
    def get_info_for_dataset(self):
        """Return robot information for dataset creation."""
        return {
            "robot_type": self.name,
            "namespace": self.namespace,
            "joint_states_topic": self.joint_states_topic,
            "joint_commands_topic": self.joint_commands_topic,
            "num_joints": self.num_joints,
            "joint_names": self.joint_names,
            "resolution": self.RESOLUTION
        }
    
    async def move_robot_absolute(
        self, 
        target_position: np.ndarray,
        target_orientation_rad: np.ndarray | None, 
        **kwargs
    ) -> None:
        """Move robot to absolute Cartesian position."""
        # For now, this is a placeholder - would need inverse kinematics
        logger.warning("Cartesian movement not implemented for ROS2Robot. Use joint space control.")
        
    async def move_to_initial_position(self) -> None:
        """Move robot to initial/calibration position."""
        logger.info("Moving ROS2 robot to initial position")
        positions = np.array(self.CALIBRATION_POSITION[:self.num_joints])
        self.set_motors_positions(positions)
        
        # Update initial position for dataset info
        self.initial_position = np.array([0.0, 0.0, 0.0])  # Default position
        self.initial_orientation_rad = np.array([0.0, 0.0, 0.0])  # Default orientation
        
        # Wait for movement to complete
        await asyncio.sleep(3.0)
        self.is_moving = False
    
    async def move_to_sleep(self) -> None:
        """Move robot to sleep position."""
        logger.info("Moving ROS2 robot to sleep position")
        positions = np.array(self.SLEEP_POSITION[:self.num_joints])
        self.set_motors_positions(positions)
        
        # Wait for movement to complete
        await asyncio.sleep(3.0)
        self.is_moving = False
    
    # BaseManipulator required methods
    def enable_torque(self) -> None:
        """Enable robot torque (for ROS2, this is usually handled by the robot driver)."""
        logger.info("Torque enable requested for ROS2 robot")
        # In ROS2, torque is typically managed by the robot's hardware interface
        pass
    
    def disable_torque(self) -> None:
        """Disable robot torque."""
        logger.info("Torque disable requested for ROS2 robot")
        # In ROS2, torque is typically managed by the robot's hardware interface
        pass
    
    def read_motor_torque(self, servo_id: int) -> float | None:
        """Read torque of a specific motor."""
        if servo_id < 1 or servo_id > self.num_joints:
            return None
        
        # Return effort from joint states (servo_id is 1-indexed)
        if len(self._current_efforts) >= servo_id:
            return float(self._current_efforts[servo_id - 1])
        return None
    
    def read_motor_voltage(self, servo_id: int) -> float | None:
        """Read voltage of a specific motor."""
        if servo_id < 1 or servo_id > self.num_joints:
            return None
        
        # ROS2 doesn't typically provide motor voltage info
        # Return default voltage from config
        return float(self.config.servos_voltage) if self.config else 12.0
    
    def write_motor_position(self, servo_id: int, units: int, **kwargs) -> None:
        """Write position to a specific motor."""
        if servo_id < 1 or servo_id > self.num_joints:
            return
            
        # Convert units to radians (servo_id is 1-indexed)
        angle_rad = (units / self.RESOLUTION) * 2 * np.pi - np.pi
        
        # Update the position array and send command
        new_positions = self._current_positions.copy()
        new_positions[servo_id - 1] = angle_rad
        self.set_motors_positions(new_positions)
    
    def read_motor_position(self, servo_id: int, **kwargs) -> int | None:
        """Read position of a specific motor in units."""
        if servo_id < 1 or servo_id > self.num_joints:
            return None
        
        # Convert radians to units (servo_id is 1-indexed)
        if len(self._current_positions) >= servo_id:
            angle_rad = self._current_positions[servo_id - 1]
            units = int((angle_rad + np.pi) / (2 * np.pi) * self.RESOLUTION)
            return max(0, min(self.RESOLUTION - 1, units))
        return None
    
    def calibrate_motors(self, **kwargs) -> None:
        """Calibrate motors by moving to calibration position."""
        logger.info("Starting ROS2 robot calibration")
        asyncio.create_task(self.move_to_initial_position())
    
    def read_group_motor_position(self) -> np.ndarray:
        """Read positions of all motors."""
        return self._current_positions.copy()
    
    def write_group_motor_position(self, q_target: np.ndarray, enable_gripper: bool) -> None:
        """Write positions to all motors."""
        self.set_motors_positions(q_target, enable_gripper)
    
    @property 
    def is_connected(self) -> bool:
        return self._is_connected
        
    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = value
    
    @classmethod
    def from_config(cls, namespace: str, **kwargs):
        """Factory method to create robot from configuration."""
        return cls(namespace=namespace, **kwargs)
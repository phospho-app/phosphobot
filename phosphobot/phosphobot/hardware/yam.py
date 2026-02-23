"""
YAM robot arm hardware driver.

Connects to YAM arm server processes via the `portal` RPC framework.
Each YamHardware instance controls a single YAM arm. For bimanual setups,
add two YamHardware instances (one per arm).

The YAM arm server must be running separately. See:
  phosphobot/scripts/run_yam_server.py

Dependencies:
  pip install portal-client
  pip install i2rt  (only needed on the machine running the server)
"""

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from loguru import logger

from phosphobot.models import RobotConfigStatus
from phosphobot.models.robot import BaseRobot


class YamHardware(BaseRobot):
    """
    Hardware driver for a single YAM robot arm via portal RPC.

    Connects to a YAM arm server process that exposes:
      - num_dofs() -> int
      - get_joint_pos() -> np.ndarray
      - command_joint_pos(joint_pos: np.ndarray) -> None
      - get_observations() -> dict[str, np.ndarray]

    Args:
        host: Server hostname (default: "localhost")
        port: Server port (default: 1234)
        arm_label: Label for this arm (e.g., "left", "right") for display purposes
    """

    name = "yam"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1234,
        arm_label: str = "",
        **kwargs: Dict[str, Any],
    ) -> None:
        self.host = host
        self.port = int(port)
        self.arm_label = arm_label or f"port-{self.port}"
        self._client = None
        self._num_dofs: Optional[int] = None
        self._is_connected = False
        self.is_moving = False
        self.device_name = f"yam-{self.arm_label}"

        # Cache for current joint positions
        self._current_joint_pos: Optional[np.ndarray] = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    async def connect(self) -> None:
        """Connect to the YAM arm server via portal RPC."""
        try:
            import portal  # type: ignore
        except ImportError:
            raise ImportError(
                "The 'portal' library is required for YAM robot support. "
                "Install it with: pip install portal-client"
            )

        try:
            logger.info(
                f"Connecting to YAM arm server at {self.host}:{self.port} "
                f"(label: {self.arm_label})"
            )
            self._client = portal.Client(f"{self.host}:{self.port}")

            # Verify connection by reading DOFs
            self._num_dofs = self._client.num_dofs().result()
            logger.info(
                f"Connected to YAM arm ({self.arm_label}): {self._num_dofs} DOFs"
            )

            # Read initial position
            self._current_joint_pos = self._client.get_joint_pos().result()
            self._is_connected = True

        except Exception as e:
            logger.error(f"Failed to connect to YAM arm server: {e}")
            self._client = None
            self._is_connected = False
            raise

    def disconnect(self) -> None:
        """Disconnect from the YAM arm server."""
        if self._client is not None:
            logger.info(f"Disconnecting from YAM arm ({self.arm_label})")
            self._client = None
        self._is_connected = False

    @property
    def num_dofs(self) -> int:
        """Number of degrees of freedom (typically 6 joints + 1 gripper = 7)."""
        return self._num_dofs or 7

    def get_observation(
        self, source: Literal["sim", "robot"], do_forward: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current observation from the YAM arm.

        Returns:
            - state: 7D array [x, y, z, roll, pitch, yaw, gripper_state]
              (approximate — YAM doesn't provide FK natively, so we return
               joint positions as a proxy state)
            - joints_position: np.array of all joint positions in radians
        """
        if not self._is_connected or self._client is None:
            logger.warning("YAM arm is not connected")
            return np.zeros(7), np.zeros(self.num_dofs)

        try:
            obs = self._client.get_observations().result()
            joint_pos = np.array(obs["joint_pos"], dtype=np.float64)

            # Include gripper if available
            gripper_val = 0.0
            if "gripper_pos" in obs:
                gripper_pos = obs["gripper_pos"]
                gripper_val = float(
                    gripper_pos[0] if hasattr(gripper_pos, "__len__") else gripper_pos
                )
                joint_pos = np.concatenate([joint_pos, [gripper_val]])

            self._current_joint_pos = joint_pos

            # Build a proxy state vector (no FK available without URDF)
            # Use first 3 joints as rough position proxy, next 3 as orientation
            state = np.zeros(7)
            n = min(len(joint_pos), 6)
            state[:n] = joint_pos[:n]
            state[6] = gripper_val

            return state, joint_pos

        except Exception as e:
            logger.error(f"Failed to read YAM arm observation: {e}")
            return np.zeros(7), np.zeros(self.num_dofs)

    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Command joint positions to the YAM arm.

        Args:
            q_target_rad: Target joint positions in radians
            enable_gripper: If True, include gripper joint; otherwise exclude last DOF
        """
        if not self._is_connected or self._client is None:
            logger.warning("YAM arm is not connected")
            return

        try:
            positions = np.array(q_target_rad, dtype=np.float64)
            if not enable_gripper and len(positions) > 6:
                # Strip gripper (last joint) if not enabled
                positions = positions[:6]
            self._client.command_joint_pos(positions)
        except Exception as e:
            logger.error(f"Failed to command YAM arm positions: {e}")

    def get_info_for_dataset(self) -> dict:
        """Return robot info for dataset recording."""
        return {
            "robot_type": "yam",
            "arm_label": self.arm_label,
            "host": self.host,
            "port": self.port,
            "num_dofs": self.num_dofs,
        }

    async def move_robot_absolute(
        self,
        target_position: np.ndarray,
        target_orientation_rad: Optional[np.ndarray],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Move the YAM arm to an absolute position.

        Since YAM arms don't have built-in IK, this directly commands
        joint positions if provided in kwargs, or does nothing.
        """
        if not self._is_connected:
            logger.warning("YAM arm is not connected")
            return

        # If joint positions are provided directly, use them
        joint_positions = kwargs.get("joint_positions")
        if joint_positions is not None:
            self.set_motors_positions(
                np.array(joint_positions), enable_gripper=True
            )

    async def move_to_initial_position(self) -> None:
        """Move the YAM arm to a neutral initial position (all zeros)."""
        if not self._is_connected or self._client is None:
            logger.warning("YAM arm is not connected")
            return

        try:
            initial = np.zeros(min(self.num_dofs, 6))
            self._client.command_joint_pos(initial)
            logger.info(f"YAM arm ({self.arm_label}) moved to initial position")
        except Exception as e:
            logger.error(f"Failed to move YAM arm to initial position: {e}")

    async def move_to_sleep(self) -> None:
        """Move the YAM arm to a safe sleep position (all zeros)."""
        if not self._is_connected or self._client is None:
            logger.warning("YAM arm is not connected")
            return

        try:
            sleep_pos = np.zeros(min(self.num_dofs, 6))
            self._client.command_joint_pos(sleep_pos)
            logger.info(f"YAM arm ({self.arm_label}) moved to sleep position")
        except Exception as e:
            logger.error(f"Failed to move YAM arm to sleep position: {e}")

    def enable_torque(self) -> None:
        """Enable torque (no-op for YAM — server handles motor control)."""
        pass

    def disable_torque(self) -> None:
        """Disable torque (no-op for YAM — server handles motor control)."""
        pass

    def status(self) -> RobotConfigStatus:
        """Get the robot status."""
        return RobotConfigStatus(
            name=self.name,
            robot_type="manipulator",
            device_name=f"{self.host}:{self.port} ({self.arm_label})",
        )

    def read_joints_position(
        self,
        unit: Literal["rad", "degrees", "motor_units"] = "rad",
        source: Optional[Literal["sim", "robot"]] = None,
    ) -> np.ndarray:
        """Read current joint positions."""
        if not self._is_connected or self._client is None:
            return np.zeros(self.num_dofs)

        try:
            joint_pos = self._client.get_joint_pos().result()
            joint_pos = np.array(joint_pos, dtype=np.float64)

            if unit == "degrees":
                joint_pos = np.rad2deg(joint_pos)

            return joint_pos
        except Exception as e:
            logger.error(f"Failed to read YAM arm joints: {e}")
            return np.zeros(self.num_dofs)

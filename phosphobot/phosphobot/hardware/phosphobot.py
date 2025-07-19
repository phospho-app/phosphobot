import asyncio
import httpx
import numpy as np
import threading

from phosphobot.hardware.base import BaseRobot
from loguru import logger

from phosphobot.models.robot import RobotConfigStatus


class RemotePhosphobot(BaseRobot):
    """
    Class to connect to another phosphobot server using HTTP
    """

    name = "phosphobot"

    def __init__(self, ip: str, port: int, robot_id: int, **kwargs):
        """
        Initialize connectio to phosphobot.

        Args:
            ip: IP address of the robot
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.ip = ip
        self.port = port
        self.client = httpx.Client(base_url=f"http://{ip}:{port}")
        self.async_client = httpx.AsyncClient(base_url=f"http://{ip}:{port}")
        self.current_position = np.zeros(3)
        self.current_orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.robot_id = robot_id
        self.initial_position: np.ndarray | None = None
        self.initial_orientation_rad: np.ndarray | None = None
        self.device_name = f"{self.ip}:{self.port}"
        
        # Cache for actuated joints to avoid repeated network calls
        self._cached_actuated_joints: list[int] | None = None
        self._joints_cache_lock = threading.Lock() 

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """
        Set the connection status of the robot.

        Args:
            value: True if connected, False otherwise
        """
        self._is_connected = value

    async def connect(self) -> None:
        """
        Initialize communication with the phosphobot server by calling /status

        Raises:
            Exception: If the connection fails
        """
        try:
            response = self.client.get("/status")
            response.raise_for_status()
            self.is_connected = True
            logger.info(f"Connected to remote phosphobot at {self.ip}:{self.port}")
        except httpx.RequestError as e:
            logger.warning(f"Failed to connect to remote phosphobot: {e}")
            raise Exception(f"Connection failed: {e}")

    def disconnect(self) -> None:
        """
        Close the connection to the robot.
        """
        try:
            self.client.close()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.async_client.aclose())
            else:
                asyncio.run(self.async_client.aclose())
            self.is_connected = False
            # if disconnected, clear the cache 
            self._cached_actuated_joints = None
            logger.info("Disconnected from remote phosphobot")
        except Exception as e:
            logger.warning(f"Failed to disconnect from remote phosphobot: {e}")
            raise Exception(f"Disconnection failed: {e}")

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.

        Returns:
            - state: [x, y, z, roll, pitch, yaw, gripper_state]
            - joints_position: np.array of joint positions
        """

        end_effector_position = self.client.post(
            "/end-effector/read", params={"robot_id": self.robot_id}
        ).json()
        joints = self.client.post(
            "/joints/read", json={"unit": "rad"}, params={"robot_id": self.robot_id}
        ).json()
        state = np.array(
            [
                end_effector_position["x"],
                end_effector_position["y"],
                end_effector_position["z"],
                end_effector_position["rx"],
                end_effector_position["ry"],
                end_effector_position["rz"],
                end_effector_position["open"],
            ]
        )
        joints_position = np.array(joints["angles"])

        return state, joints_position

    def set_motors_positions(
        self, positions: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Set the motor positions of the robot.
        """

        self.client.post(
            "/joints/write",
            json={"angles": positions.tolist(), "unit": "rad"},
            params={"robot_id": self.robot_id},
        )

    def get_info_for_dataset(self):
        """
        Get information about the remote robot for dataset creation.
        Assume dof = 6 by default. 
        
        Returns:
            BaseRobotInfo: Information about the robot including action and observation state shapes
        """
        from phosphobot.models.lerobot_dataset import BaseRobotInfo, FeatureDetails
        
        actual_dof = 6  # Default fallback
        
        if not self.is_connected:
            logger.warning("RemotePhosphobot: Robot is not connected. Using default 6 DOF.")
        else:
            try:
                joints_response = self.client.post(
                    "/joints/read", 
                    json={"unit": "rad"}, 
                    params={"robot_id": self.robot_id}
                )
                joints_response.raise_for_status()  
                joints_data = joints_response.json()
                
                if "angles" in joints_data and isinstance(joints_data["angles"], list):
                    if len(joints_data["angles"]) > 0:
                        actual_dof = len(joints_data["angles"])
                        logger.info(f"RemotePhosphobot: Detected {actual_dof} DOF from remote robot")
                    else:
                        logger.warning("RemotePhosphobot: Remote robot returned empty joints array. Using default 6 DOF.")
                else:
                    logger.warning("RemotePhosphobot: Invalid response format from remote robot. Using default 6 DOF.")
                    
            except Exception as e:
                logger.warning(f"RemotePhosphobot: Could not query remote robot DOF: {e}. Using default 6 DOF.")
        
        # Generate joint names based on actual DOF count
        joint_names = [f"joint_{i+1}" for i in range(actual_dof)]
        
        return BaseRobotInfo(
            robot_type=self.name,
            action=FeatureDetails(
                dtype="float32",
                shape=[actual_dof], 
                names=joint_names,
            ),
            observation_state=FeatureDetails(
                dtype="float32",
                shape=[actual_dof], 
                names=joint_names,
            ),
        )

    async def move_robot_absolute(
        self,
        target_position: np.ndarray,
        target_orientation_rad: np.ndarray | None,
        **kwargs,
    ) -> None:
        """
        Move the robot to the target position and orientation asynchronously.

        Args:
            target_position: Target position as [x, y, z]
            target_orientation_rad: Target orientation in radians [roll, pitch, yaw]
            **kwargs: Additional arguments
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return

        body = {
            "x": target_position[0] * 100,
            "y": target_position[1] * 100,
            "z": target_position[2] * 100,
        }
        if target_orientation_rad is not None:
            target_orientation_deg = np.rad2deg(target_orientation_rad)
            body = {
                **body,
                "rx": target_orientation_deg[0],
                "ry": target_orientation_deg[1],
                "rz": target_orientation_deg[2],
            }

        response = await self.async_client.post(
            "/move/absolute",
            json=body,
            params={"robot_id": self.robot_id},
        )
        if response.status_code != 200:
            logger.warning(
                f"Failed to move robot to absolute position: {response.text}"
            )
            raise Exception(f"Move failed: {response.text}")

    async def move_robot_relative(
        self, target_position: np.ndarray, target_orientation_rad: np.ndarray | None
    ) -> None:
        # Replace None values in target_position and target_orientation_rad with 0

        body = {
            "x": target_position[0] * 100 if target_position[0] is not None else None,
            "y": target_position[1] * 100 if target_position[1] is not None else None,
            "z": target_position[2] * 100 if target_position[2] is not None else None,
        }
        if target_orientation_rad is not None:
            body = {
                **body,
                "rx": np.rad2deg(target_orientation_rad[0])
                if target_orientation_rad[0] is not None
                else None,
                "ry": np.rad2deg(target_orientation_rad[1])
                if target_orientation_rad[1] is not None
                else None,
                "rz": np.rad2deg(target_orientation_rad[2])
                if target_orientation_rad[2] is not None
                else None,
            }

        response = await self.async_client.post(
            "/move/relative",
            json=body,
            params={"robot_id": self.robot_id},
        )
        if response.status_code != 200:
            logger.warning(
                f"Failed to move robot to relative position: {response.text}"
            )
            raise Exception(f"Move failed: {response.text}")

    def forward_kinematics(
        self, sync_robot_pos: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the end effector position and orientation of the robot.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return np.zeros(3), np.zeros(3)

        end_effector_position = self.client.post(
            "/end-effector/read", params={"robot_id": self.robot_id}
        ).json()

        current_effector_position = np.array(
            [
                end_effector_position["x"],
                end_effector_position["y"],
                end_effector_position["z"],
            ]
        )
        current_effector_orientation_deg = np.array(
            [
                end_effector_position["rx"],
                end_effector_position["ry"],
                end_effector_position["rz"],
            ]
        )
        # Convert to radians
        current_effector_orientation_rad = np.deg2rad(current_effector_orientation_deg)

        return (
            current_effector_position,
            current_effector_orientation_rad,
        )

    def status(self) -> RobotConfigStatus:
        """
        Get the status of the robot.

        Returns:
            RobotConfigStatus object
        """
        return RobotConfigStatus(
            name=self.name,
            device_name=self.ip,
        )

    async def move_to_initial_position(self) -> None:
        """
        Move the robot to its initial position.

        This puts the robot in a stand position ready for operation.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return

        await self.async_client.post("/move/init", params={"robot_id": self.robot_id})
        # Read the initial position and orientation after moving
        (
            self.initial_position,
            self.initial_orientation_rad,
        ) = self.forward_kinematics()

    async def move_to_sleep(self) -> None:
        """
        Move the robot to its sleep position.

        This makes the robot sit down before potentially disconnecting.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return
        await self.async_client.post("/move/sleep", params={"robot_id": self.robot_id})

    def enable_torque(self):
        """
        Enable the torque of the robot.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return

        self.client.post(
            "/torque/toggle",
            json={"torque_status": True},
            params={"robot_id": self.robot_id},
        )

    def disable_torque(self):
        """
        Disable the torque of the robot.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return

        self.client.post(
            "/torque/toggle",
            json={"torque_status": False},
            params={"robot_id": self.robot_id},
        )

    def control_gripper(self, open_command: float) -> None:
        """
        Control the gripper of the robot.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return

        # Move absolute with only open command
        self.client.post(
            "/move/absolute",
            json={"open": open_command},
            params={"robot_id": self.robot_id},
        )

    def current_torque(self) -> np.ndarray:
        """
        Read current torque /torque/read
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return np.zeros(6)

        response = self.client.post(
            "/torque/read",
            params={"robot_id": self.robot_id},
        )
        torque = response.json()
        return np.array(torque["current_torque"])

    def current_voltage(self) -> np.ndarray:
        """
        Read current voltage /voltage/read
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return np.zeros(6)

        response = self.client.post(
            "/voltage/read",
            params={"robot_id": self.robot_id},
        )
        voltage = response.json()
        return np.array(voltage["current_voltage"])

    def write_joint_positions(
        self,
        angles: list[float],
        unit: str = "rad",
        joints_ids: list[int] | None = None,
        **kwargs,
    ) -> None:
        """
        Write joint positions to the robot.

        Args:
            positions: Joint positions as a numpy array
            unit: Unit of the joint positions (default is 'rad')
            **kwargs: Additional keyword arguments
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            return

        self.client.post(
            "/joints/write",
            json={"angles": angles, "unit": unit, "joints_ids": joints_ids},
            params={"robot_id": self.robot_id},
        )

    def read_joints_position(self, unit: str = "rad") -> np.ndarray:
        """
        Read the current joint positions of the robot.

        Args:
            unit: Unit of the joint positions (default is 'rad')
        Returns:
            np.ndarray: Joint positions as a numpy array
        """
        if not self.is_connected:
            logger.warning("Robot is not connected")
            dof_count = len(self.actuated_joints)
            return np.zeros(dof_count)

        try:
            response = self.client.post(
                "/joints/read",
                json={"unit": unit},
                params={"robot_id": self.robot_id},
            )
            response.raise_for_status()
            joints_data = response.json()
            
            if "angles" in joints_data and isinstance(joints_data["angles"], list):
                return np.array(joints_data["angles"])
            else:
                logger.warning("RemotePhosphobot: Invalid response format when reading joint positions")
                dof_count = len(self.actuated_joints)
                return np.zeros(dof_count)
                
        except Exception as e:
            logger.warning(f"RemotePhosphobot: Failed to read joint positions: {e}")
            dof_count = len(self.actuated_joints)
            return np.zeros(dof_count)

    @property
    def actuated_joints(self) -> list[int]:
        """
        Get the list of actuated joints.

        Returns:
            list[int]: List of actuated joint IDs
        """
        with self._joints_cache_lock:
            if self._cached_actuated_joints is not None:
                return self._cached_actuated_joints
            
            # Default fallback
            default_joints = [1, 2, 3, 4, 5, 6]
            
            if not self.is_connected:
                logger.warning("RemotePhosphobot: Robot is not connected. Using default joints [1,2,3,4,5,6].")
                self._cached_actuated_joints = default_joints
                return self._cached_actuated_joints
            
            try:
                joints_response = self.client.post(
                    "/joints/read", 
                    json={"unit": "rad"}, 
                    params={"robot_id": self.robot_id}
                )
                joints_response.raise_for_status()  
                joints_data = joints_response.json()
                
                if "angles" in joints_data and isinstance(joints_data["angles"], list):
                    if len(joints_data["angles"]) > 0:
                        actual_dof = len(joints_data["angles"])
                        self._cached_actuated_joints = list(range(1, actual_dof + 1))
                        logger.info(f"RemotePhosphobot: Detected {actual_dof} joints: {self._cached_actuated_joints}")
                        return self._cached_actuated_joints
                    else:
                        logger.warning("RemotePhosphobot: Remote robot returned empty joints array. Using default joints.")
                else:
                    logger.warning("RemotePhosphobot: Invalid response format from remote robot. Using default joints.")
                    
            except Exception as e:
                logger.warning(f"RemotePhosphobot: Could not query remote robot joints: {e}. Using default joints.")
            
            self._cached_actuated_joints = default_joints
            return self._cached_actuated_joints

    def _set_pid_gains_motors(
        self, servo_id: int, p_gain: int, i_gain: int, d_gain: int
    ) -> None:
        """
        Set PID gains for a specific motor.

        Args:
            servo_id: ID of the servo motor
            p_gain: Proportional gain
            i_gain: Integral gain
            d_gain: Derivative gain
        """
        # TODO: Implement this method to set PID gains for the motors
        pass

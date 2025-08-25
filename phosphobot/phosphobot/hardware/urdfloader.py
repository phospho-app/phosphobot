import json
import threading
from typing import List, Literal, Optional

import numpy as np
import zmq
from loguru import logger

from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import RobotConfigStatus
from phosphobot.models.robot import BaseRobotConfig, BaseRobotPIDGains


class URDFLoader(BaseManipulator):
    name = "urdf_loader"

    RESOLUTION = 4096  # unused for now

    def __init__(
        self,
        urdf_path: str,
        end_effector_link_index: int,
        gripper_joint_index: int,
        zmq_server_url: str | None = None,
        zmq_topic: str | None = None,
        axis_orientation: list[int] | None = None,
    ):
        self.URDF_FILE_PATH = urdf_path
        self.END_EFFECTOR_LINK_INDEX = int(end_effector_link_index)
        self.GRIPPER_JOINT_INDEX = int(gripper_joint_index)
        self.zmq_server_url = zmq_server_url
        self.zmq_topic = zmq_topic

        # --- Threading and ZMQ Attributes ---
        self.zmq_context: zmq.Context | None = None
        self.zmq_socket: zmq.Socket | None = None
        self.zmq_latest_joint_positions: np.ndarray | None = None

        # Thread-safe mechanisms
        self.data_lock = threading.Lock()  # To safely write/read joint positions
        self.stop_event = threading.Event()  # To signal the thread to stop
        self.zmq_thread: threading.Thread | None = None

        if axis_orientation is not None:
            self.AXIS_ORIENTATION = axis_orientation
        else:
            self.AXIS_ORIENTATION = [0, 0, 0, 1]

        super().__init__(only_simulation=True)

    def _zmq_listen_loop(self) -> None:
        """
        This method runs in a separate thread, continuously listening for ZMQ messages.
        """
        poller = zmq.Poller()
        poller.register(self.zmq_socket, zmq.POLLIN)

        while not self.stop_event.is_set():
            # Wait for a message with a timeout of 100ms
            # This allows the loop to periodically check the stop_event
            socks = dict(poller.poll(100))
            if self.zmq_socket in socks:
                try:
                    topic, msg_bytes = self.zmq_socket.recv_multipart()
                    joint_data = json.loads(msg_bytes.decode("utf-8"))

                    if isinstance(joint_data, list) and len(joint_data) == len(
                        self.SERVO_IDS
                    ):
                        # Safely update the shared data
                        with self.data_lock:
                            self.zmq_latest_joint_positions = np.array(
                                joint_data, dtype=np.float32
                            )
                    else:
                        logger.warning(f"Received malformed ZMQ data: {joint_data}")
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Error decoding ZMQ message: {e}")
                except Exception as e:
                    # Catch other potential ZMQ errors if the context is terminated
                    if not self.stop_event.is_set():
                        logger.error(f"Error in ZMQ listen loop: {e}")

    async def connect(self) -> None:
        """
        Connect to the robot and initialize the ZMQ subscriber in a background thread.
        """
        if not self.zmq_server_url or not self.zmq_topic:
            self.is_connected = True
            return

        logger.info(
            f"Connecting to ZMQ server at {self.zmq_server_url} with topic '{self.zmq_topic}'"
        )
        try:
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.SUB)
            self.zmq_socket.connect(self.zmq_server_url)
            self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, self.zmq_topic)

            # Start the background listening thread
            self.stop_event.clear()
            self.zmq_thread = threading.Thread(
                target=self._zmq_listen_loop, daemon=True
            )
            self.zmq_thread.start()

            self.is_connected = True
            logger.info(
                "Successfully connected to ZMQ server and started listener thread."
            )
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ subscriber: {e}")
            self.is_connected = False

    def disconnect(self) -> None:
        """
        Disconnect the robot and gracefully shut down the ZMQ thread.
        """
        # Signal the thread to stop
        self.stop_event.set()

        # Wait for the thread to finish
        if self.zmq_thread and self.zmq_thread.is_alive():
            self.zmq_thread.join(timeout=1.0)

        # Clean up ZMQ resources
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        self.is_connected = False
        logger.info("ZMQ listener thread stopped and resources released.")

    def get_observation(
        self, do_forward: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot by reading the latest data from the ZMQ thread.
        """
        if not self.zmq_server_url or not self.zmq_topic:
            # If no ZMQ connection is set, fallback to the base class method
            return super().get_observation(do_forward=do_forward)

        # If a ZMQ connection is active, try to use its latest data
        with self.data_lock:
            if self.zmq_latest_joint_positions is not None:
                # Use a copy to ensure thread safety
                joints_position = self.zmq_latest_joint_positions.copy()

        state = np.full(6, np.nan)
        if do_forward:
            effector_position, effector_orientation_euler_rad = self.forward_kinematics(
                q=joints_position
            )
            state = np.concatenate((effector_position, effector_orientation_euler_rad))

        return state, joints_position

    def init_config(self) -> None:
        """
        This config is used for PID tuning, motors offsets, and other parameters.
        """
        self.config = self.get_default_base_robot_config()

    def get_default_base_robot_config(
        self, voltage: str = "6.0", raise_if_none: bool = False
    ) -> BaseRobotConfig:
        return BaseRobotConfig(
            name=self.name,
            servos_voltage=6.0,
            servos_offsets=[0] * len(self.SERVO_IDS),
            servos_calibration_position=[1] * len(self.SERVO_IDS),
            servos_offsets_signs=[1] * len(self.SERVO_IDS),
            pid_gains=[BaseRobotPIDGains(p_gain=0, i_gain=0, d_gain=0)]
            * len(self.SERVO_IDS),
            gripping_threshold=10,
            non_gripping_threshold=1,
        )

    def enable_torque(self):
        pass

    def disable_torque(self):
        pass

    def _set_pid_gains_motors(
        self, servo_id: int, p_gain: int = 32, i_gain: int = 0, d_gain: int = 32
    ):
        pass

    def read_joints_position(
        self,
        unit: Literal["rad", "motor_units", "degrees", "other"] = "rad",
        source: Literal["sim", "robot"] = "robot",
        joints_ids: Optional[List[int]] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        return super().read_joints_position(
            unit=unit,
            source="sim",
            joints_ids=joints_ids,
            min_value=min_value,
            max_value=max_value,
        )

    def read_motor_position(self, servo_id: int, **kwargs) -> int | None:
        pass

    def write_motor_position(self, servo_id: int, units: int, **kwargs) -> None:
        pass

    def write_group_motor_position(
        self, q_target: np.ndarray, enable_gripper: bool = True
    ) -> None:
        pass

    def read_group_motor_position(self) -> np.ndarray:
        return np.zeros(len(self.SERVO_IDS), dtype=np.int32)

    def read_motor_torque(self, servo_id: int, **kwargs) -> float | None:
        pass

    def read_motor_voltage(self, servo_id: int, **kwargs) -> float | None:
        pass

    def status(self) -> RobotConfigStatus:
        return RobotConfigStatus(
            name=self.name,
            device_name=self.URDF_FILE_PATH,
            temperature=None,
        )

    async def calibrate(self) -> tuple[Literal["success", "in_progress", "error"], str]:
        return "success", "Calibration not implemented yet."

    def calibrate_motors(self, **kwargs) -> None:
        pass

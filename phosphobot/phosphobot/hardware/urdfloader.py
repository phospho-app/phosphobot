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
    device_name = "urdf_loader"

    RESOLUTION = 4096  # unused for now

    def __init__(
        self,
        urdf_path: str,
        end_effector_link_index: int,
        gripper_joint_index: int,
        zmq_server_url: Optional[str] = None,
        zmq_topic: Optional[str] = None,
        axis_orientation: Optional[List[int]] = None,
    ):
        self.URDF_FILE_PATH = urdf_path
        self.END_EFFECTOR_LINK_INDEX = int(end_effector_link_index)
        self.GRIPPER_JOINT_INDEX = int(gripper_joint_index)
        self.zmq_server_url = (
            zmq_server_url if zmq_server_url and zmq_server_url.strip() else None
        )
        self.zmq_topic = zmq_topic if zmq_topic and zmq_topic.strip() else None
        self.zmq_context: Optional[zmq.Context] = None
        self.zmq_socket: Optional[zmq.Socket] = None
        self.zmq_latest_joint_positions: Optional[np.ndarray] = None
        self.zmq_thread: Optional[threading.Thread] = None
        self._zmq_initialized = False
        self._zmq_init_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.AXIS_ORIENTATION = (
            axis_orientation if axis_orientation is not None else [0, 0, 0, 1]
        )
        super().__init__(only_simulation=True)

    def _initialize_zmq(self) -> None:
        """Initializes the ZMQ PULL socket and starts the listener thread."""
        if not self.zmq_server_url or not self.zmq_topic:
            logger.warning("ZMQ server URL or topic not set. Skipping initialization.")
            return

        logger.info(
            f"Lazily initializing ZMQ PULL connection to {self.zmq_server_url} for topic '{self.zmq_topic}'"
        )
        try:
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PULL)
            self.zmq_socket.connect(self.zmq_server_url)

            self.stop_event.clear()
            self.zmq_thread = threading.Thread(
                target=self._zmq_listen_loop, daemon=True
            )
            self.zmq_thread.start()
            logger.success("ZMQ PULL listener thread started successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ PULL socket: {e}")
            self.zmq_context = None
            self.zmq_socket = None

    def _zmq_listen_loop(self) -> None:
        """Runs in a background thread, listening for and processing ZMQ messages."""
        if not self.zmq_socket:
            logger.error("ZMQ socket not initialized. Exiting listener thread.")
            return

        poller = zmq.Poller()
        poller.register(self.zmq_socket, zmq.POLLIN)
        logger.debug("ZMQ listener thread started. Waiting for messages...")

        while not self.stop_event.is_set():
            socks = dict(poller.poll(100))
            if not socks:
                # This is now an expected state if no messages for this topic are sent
                continue

            if self.zmq_socket in socks:
                try:
                    topic_bytes, msg_bytes = self.zmq_socket.recv_multipart()

                    if topic_bytes.decode() != self.zmq_topic:
                        continue  # Ignore messages not intended for this subscriber

                    json_string = msg_bytes.decode("utf-8")
                    obs_dict = json.loads(json_string)

                    if isinstance(obs_dict, dict) and "joints" in obs_dict:
                        joint_data = np.array(obs_dict["joints"], dtype=np.float32)

                        if joint_data.shape == (len(self.SERVO_IDS),):
                            with self.data_lock:
                                self.zmq_latest_joint_positions = joint_data
                        else:
                            logger.warning(
                                f"Received malformed 'joints' data shape: {joint_data.shape}"
                            )
                    else:
                        logger.warning(
                            f"ZMQ payload is not a valid observation dictionary: {obs_dict}"
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding ZMQ JSON message: {e}")
                except Exception as e:
                    if not self.stop_event.is_set():
                        logger.error(f"Error in ZMQ listen loop: {e}")

    async def connect(self) -> None:
        """
        Marks the robot as connected. ZMQ initialization is deferred to get_observation.
        """
        self.is_connected = True

    def disconnect(self) -> None:
        """
        Gracefully shuts down the ZMQ thread and cleans up resources if they were initialized.
        """
        if self._zmq_initialized:
            self.stop_event.set()
            if self.zmq_thread and self.zmq_thread.is_alive():
                self.zmq_thread.join(timeout=1.0)
            if self.zmq_socket:
                self.zmq_socket.close()
            if self.zmq_context:
                self.zmq_context.term()
            logger.info("ZMQ listener stopped and resources released.")
        self.is_connected = False

    def get_observation(
        self, do_forward: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets the robot's observation. If ZMQ is configured, it will initialize the
        connection on the first call and then read the latest data from the listener thread.
        """
        if self.zmq_server_url and self.zmq_topic:
            # Lazy initialization of ZMQ (if possible)
            with self._zmq_init_lock:
                if not self._zmq_initialized:
                    self._initialize_zmq()
                    self._zmq_initialized = True

        # If ZMQ has provided data, overwrite the simulation state
        if self._zmq_initialized:
            with self.data_lock:
                if self.zmq_latest_joint_positions is not None:
                    joints_position = self.zmq_latest_joint_positions.copy()
                else:
                    logger.warning(
                        "ZMQ latest joint positions not available, falling back to simulation state."
                    )
                    joints_position = self.read_joints_position(
                        unit="rad", source="sim"
                    )
        else:
            # Fallback to simulation state if ZMQ is not initialized
            joints_position = self.read_joints_position(unit="rad", source="sim")

        # --- Kinematics and Return ---
        state = np.full(6, np.nan)
        if do_forward:
            effector_position, effector_orientation_euler_rad = (
                self.forward_kinematics()
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

    def read_motor_position(self, servo_id: int, **kwargs) -> Optional[int]:
        pass

    def write_motor_position(self, servo_id: int, units: int, **kwargs) -> None:
        pass

    def write_group_motor_position(
        self, q_target: np.ndarray, enable_gripper: bool = True
    ) -> None:
        pass

    def read_group_motor_position(self) -> np.ndarray:
        return np.zeros(len(self.SERVO_IDS), dtype=np.int32)

    def read_motor_torque(self, servo_id: int, **kwargs) -> Optional[float]:
        pass

    def read_motor_voltage(self, servo_id: int, **kwargs) -> Optional[float]:
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

    def update_object_gripping_status(self):
        # The object is never gripped
        self.is_object_gripped = False

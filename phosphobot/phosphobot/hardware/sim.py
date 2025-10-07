"""
PyBullet Simulation wrapper class
"""

import io
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
from loguru import logger

from phosphobot.types import SimulationMode

sim = None


class PyBulletSimulation:
    """
    A comprehensive wrapper class for PyBullet simulation environment.
    """

    def __init__(
        self,
        sim_mode: SimulationMode = SimulationMode.headless,
    ) -> None:
        """
        Initialize the PyBullet simulation environment.

        Args:
            sim_mode (SimulationMode): Simulation mode - "headless" or "gui"
        """
        self.sim_mode = sim_mode
        self.connected = False
        self.robots: dict = {}  # Store loaded robots
        self._running = False
        self._step_thread: Optional[threading.Thread] = None

        self._pending_steps = 0
        self._lock = threading.Lock()

        self.init_simulation()
        self.start_stepping()

    def init_simulation(self) -> None:
        """
        Initialize the pybullet simulation environment based on the configuration.
        """
        if self.sim_mode == SimulationMode.headless:
            p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1.0 / 10)  # 10 Hz simulation
            self.connected = True
            logger.debug("Simulation: headless mode enabled")

        elif self.sim_mode == SimulationMode.gui:
            # Spin up a new process for the simulation
            absolute_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "simulation",
                    "pybullet",
                )
            )

            def _stream_to_console(pipe: io.BufferedReader) -> None:
                """Continuously read from *pipe* and write to stdout."""
                try:
                    with pipe:
                        for line in iter(pipe.readline, b""):
                            # decode bytes -> str and write to the console
                            sys.stdout.write(
                                "[gui sim] " + line.decode("utf-8", errors="replace")
                            )
                            sys.stdout.flush()
                except Exception as exc:
                    logger.warning(f"Error while reading child stdout: {exc}")

            self._gui_proc = subprocess.Popen(
                ["uv", "run", "--python", "3.8", "main.py"],
                cwd=absolute_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout
                bufsize=0,
            )
            t = threading.Thread(
                target=_stream_to_console, args=(self._gui_proc.stdout,), daemon=True
            )
            t.start()

            # Wait for 1 second to allow the simulation to start
            time.sleep(1)
            p.connect(p.SHARED_MEMORY)
            self.connected = True
            logger.debug("Simulation: GUI mode enabled")

        else:
            raise ValueError("Invalid simulation mode")

        try:
            import pybullet_data

            data_path = pybullet_data.getDataPath()
            p.setAdditionalSearchPath(data_path)
            logger.debug(f"Added pybullet_data search path: {data_path}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to set additional search path for pybullet_data: {e}")

    def start_stepping(self) -> None:
        if self._running:
            logger.debug("Simulation stepping already running")
            return

        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot start stepping")
            return

        self._running = True

        def _loop() -> None:
            while self._running and self.connected and p.isConnected():
                steps_to_do = 0
                with self._lock:
                    if self._pending_steps > 0:
                        steps_to_do = self._pending_steps
                        self._pending_steps = 0

                if steps_to_do > 0:
                    for _ in range(steps_to_do):
                        p.stepSimulation()
                else:
                    time.sleep(0.01)  # avoid busy loop

        self._step_thread = threading.Thread(target=_loop, daemon=True)
        self._step_thread.start()
        logger.info("Started background stepping thread with counter")

    def stop_stepping(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._step_thread is not None:
            self._step_thread.join(timeout=1)
            self._step_thread = None
        logger.info("Stopped background stepping thread")

    def step(self, steps: int = 60) -> None:
        """
        Increment the pending step counter (non-blocking).
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot enqueue step")
            return

        with self._lock:
            self._pending_steps += steps

    def stop(self) -> None:
        """
        Cleanup the simulation environment.
        """
        self.stop_stepping()
        if self.connected and p.isConnected():
            p.disconnect()
            self.connected = False
            logger.info("Simulation disconnected")

        if self.sim_mode == "gui":
            if hasattr(self, "_gui_proc") and self._gui_proc.poll() is None:
                self._gui_proc.terminate()
                try:
                    self._gui_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._gui_proc.kill()
            # Kill the simulation process: any instance of python 3.8
            # A bit invasive. Can we do something better?
            subprocess.run(["pkill", "-f", "python3.8"])

    def __del__(self) -> None:
        """
        Cleanup when object is destroyed.
        """
        self.stop()

    def reset(self) -> None:
        """
        Reset the simulation environment.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot reset")
            return

        p.resetSimulation()
        self.robots.clear()
        logger.info("Simulation reset")

    def set_joint_state(
        self, robot_id: int, joint_id: int, joint_position: float
    ) -> None:
        """
        Set the joint state of a robot in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_id (int): The ID of the joint to set.
            joint_position (float): The position to set the joint to.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot set joint state")
            return

        p.resetJointState(robot_id, joint_id, joint_position)

    def inverse_dynamics(
        self,
        robot_id: int,
        positions: List[float],
        velocities: List[float],
        accelerations: List[float],
    ) -> List[float]:
        """
        Perform inverse dynamics to compute joint angles from end-effector pose.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            positions (list): Joint positions
            velocities (list): Joint velocities
            accelerations (list): Joint accelerations

        Returns:
            list: Joint torques
        """
        if not self.connected or not p.isConnected():
            logger.warning(
                "Simulation is not connected, cannot perform inverse dynamics"
            )
            return []

        joint_angles = p.calculateInverseDynamics(
            robot_id, positions, velocities, accelerations
        )
        return joint_angles

    def load_urdf(
        self,
        urdf_path: str,
        axis: Optional[List[float]] = None,
        axis_orientation: Optional[List[int]] = None,
        use_fixed_base: bool = True,
        enable_self_collision: bool = False,
    ) -> Tuple[int, int, List[int]]:
        """
        Load a URDF file into the simulation.

        Args
            urdf_path (str): Path to the URDF file
            axis (list, optional): Base position of the robot in the simulation
            axis_orientation (list, optional): Base orientation of the robot in the simulation
            use_fixed_base (bool): Whether to use a fixed base for the robot

        Returns:
            tuple: (robot_id, num_joints, actuated_joints)
        """
        if axis_orientation is None:
            axis_orientation = [0, 0, 0, 1]

        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot load URDF")
            raise RuntimeError(
                f"Can't load URDF {urdf_path} - simulation not connected."
            )

        if enable_self_collision:
            flags = p.URDF_MAINTAIN_LINK_ORDER and p.URDF_USE_SELF_COLLISION
        else:
            flags = p.URDF_MAINTAIN_LINK_ORDER

        from phosphobot.utils import get_resources_path

        # Only load the plane when not on windows, Pybullet loads the urdf with "/" instead of "\" on windows
        # I wasn't able to find the root cause, maybe it's when we compile pybullet ?
        if os.name != "nt":
            plane_path_str = str(get_resources_path() / "urdf" / "plane" / "plane.urdf")
            p.loadURDF(plane_path_str)
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=axis,
            baseOrientation=axis_orientation,
            useFixedBase=use_fixed_base,
            flags=flags,
        )

        num_joints = p.getNumJoints(robot_id)
        actuated_joints = []

        for i in range(num_joints):
            joint_type = self.get_joint_info(robot_id, i)[2]
            # Consider only revolute joints
            if joint_type in [p.JOINT_REVOLUTE]:
                actuated_joints.append(i)
            else:
                logger.warning(
                    f"Joint {i} is not revolute, type: {joint_type} - skipping"
                )

        # Store robot info
        self.robots[robot_id] = {
            "urdf_path": urdf_path,
            "num_joints": num_joints,
            "actuated_joints": actuated_joints,
        }

        return robot_id, num_joints, actuated_joints

    def set_joints_states(
        self, robot_id: int, joint_indices: List[int], target_positions: List[float]
    ) -> None:
        """
        Set multiple joint states of a robot in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_indices (list[int]): The indices of the joints to set.
            target_positions (list[float]): The positions to set the joints to.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot set joint states")
            return

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
        )

    def get_joints_states(self, robot_id: int, joint_indices: List[int]) -> List[float]:
        """
        Get the states of multiple joints in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_indices (list[int]): The indices of the joints to get.

        Returns:
            list[float]: List of joint positions.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot get joint states")
            return []

        joint_states = p.getJointStates(robot_id, joint_indices)
        joint_positions = [state[0] for state in joint_states]
        return joint_positions

    def get_joint_state(self, robot_id: int, joint_index: int) -> List:
        """
        Get the state of a joint in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_index (int): The index of the joint to get.

        Returns:
            list: pybullet list describing the joint state.s
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot get joint state")
            return []

        joint_state = p.getJointState(robot_id, joint_index)
        return joint_state

    def inverse_kinematics(
        self,
        robot_id: int,
        end_effector_link_index: int,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        rest_poses: List,
        joint_damping: Optional[List] = None,
        lower_limits: Optional[List] = None,
        upper_limits: Optional[List] = None,
        joint_ranges: Optional[List] = None,
        max_num_iterations: int = 200,
        residual_threshold: float = 1e-6,
    ) -> List[float]:
        """
        Perform inverse kinematics to compute joint angles from end-effector pose.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            end_effector_link_index (int): The index of the end-effector link.
            target_position (list): The target position for the end-effector.
            target_orientation (list): The target orientation for the end-effector.
            rest_poses (list): Rest poses for the joints.
            joint_damping (list, optional): Damping for each joint.
            lower_limits (list, optional): Lower limits for each joint.
            upper_limits (list, optional): Upper limits for each joint.
            joint_ranges (list, optional): Joint ranges for each joint.
            max_num_iterations (int, optional): Maximum number of iterations for IK solver.
            residual_threshold (float, optional): Residual threshold for IK solver.

        Returns:
            list: Joint angles computed by inverse kinematics.
        """
        if not self.connected or not p.isConnected():
            logger.warning(
                "Simulation is not connected, cannot perform inverse kinematics"
            )
            return []

        if joint_damping is None:
            return p.calculateInverseKinematics(
                robot_id,
                end_effector_link_index,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                restPoses=rest_poses,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                maxNumIterations=max_num_iterations,
                residualThreshold=residual_threshold,
            )

        return p.calculateInverseKinematics(
            robot_id,
            end_effector_link_index,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            jointDamping=joint_damping,
            solver=p.IK_SDLS,
            restPoses=rest_poses,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            maxNumIterations=max_num_iterations,
            residualThreshold=residual_threshold,
        )

    def get_link_state(
        self, robot_id: int, link_index: int, compute_forward_kinematics: bool = False
    ) -> list:
        """
        Get the state of a link in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            link_index (int): The index of the link to get.
            compute_forward_kinematics (bool): Whether to compute forward kinematics.

        Returns:
            list: pybullet list describing the link state.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot get link state")
            return []

        link_state = p.getLinkState(
            robot_id, link_index, computeForwardKinematics=compute_forward_kinematics
        )
        return link_state

    def get_joint_info(self, robot_id: int, joint_index: int) -> List:
        """
        Get the information of a joint in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_index (int): The index of the joint to get.

        Returns:
            list: pybullet list describing the joint info.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot get joint info")
            return []

        joint_info = p.getJointInfo(robot_id, joint_index)
        return joint_info

    def add_debug_text(
        self, text: str, text_position: list, text_color_RGB: list, life_time: int = 3
    ) -> None:
        """
        Add debug text to the simulation.

        Args:
            text (str): The text to display.
            text_position (list): The position to display the text at.
            text_color_RGB (list): The color of the text in RGB format.
            life_time (int, optional): The lifetime of the debug text in seconds. Defaults to 3.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot add debug text")
            return

        p.addUserDebugText(
            text=text,
            textPosition=text_position,
            textColorRGB=text_color_RGB,
            lifeTime=life_time,
        )

    def add_debug_points(
        self,
        point_positions: List,
        point_colors_RGB: List,
        point_size: int = 4,
        life_time: int = 3,
    ) -> None:
        """
        Add debug points to the simulation.

        Args:
            point_positions (list): The positions of the points.
            point_colors_RGB (list): The colors of the points in RGB format.
            point_size (int, optional): The size of the points. Defaults to 4.
            life_time (int, optional): The lifetime of the debug points in seconds. Defaults to 3.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot add debug points")
            return

        p.addUserDebugPoints(
            pointPositions=point_positions,
            pointColorsRGB=point_colors_RGB,
            pointSize=point_size,
            lifeTime=life_time,
        )

    def add_debug_lines(
        self,
        line_from_XYZ: List,
        line_to_XYZ: List,
        line_color_RGB: List,
        line_width: int = 4,
        life_time: int = 3,
    ) -> None:
        """
        Add debug lines to the simulation.

        Args:
            line_from_XYZ (list): The starting position of the line.
            line_to_XYZ (list): The ending position of the line.
            line_color_RGB (list): The color of the line in RGB format.
            line_width (int, optional): The width of the line. Defaults to 4.
            life_time (int, optional): The lifetime of the debug line in seconds. Defaults to 3.
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot add debug lines")
            return

        p.addUserDebugLine(
            lineFromXYZ=line_from_XYZ,
            lineToXYZ=line_to_XYZ,
            lineColorRGB=line_color_RGB,
            lineWidth=line_width,
            lifeTime=life_time,
        )

    def get_robot_info(self, robot_id: int) -> dict:
        """
        Get information about a loaded robot.

        Args:
            robot_id (int): The ID of the robot

        Returns:
            dict: Robot information dictionary
        """
        return self.robots.get(robot_id, {})

    def get_all_robots(self) -> dict:
        """
        Get all loaded robots.

        Returns:
            dict: Dictionary of all loaded robots
        """
        return self.robots

    def is_connected(self) -> bool:
        """
        Check if the simulation is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and p.isConnected()

    def set_gravity(self, gravity_vector: Optional[List[float]] = None) -> None:
        """
        Set the gravity vector for the simulation.

        Args:
            gravity_vector (list): The gravity vector [x, y, z]
        """
        if gravity_vector is None:
            gravity_vector = [0, 0, -9.81]

        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot set gravity")
            return

        p.setGravity(*gravity_vector)

    def get_dynamics_info(self, robot_id: int, link_index: int = -1) -> List:
        """
        Get dynamics information for a robot body/link.

        Args:
            robot_id (int): The ID of the robot
            link_index (int): The link index (-1 for base)

        Returns:
            list: Dynamics information
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot get dynamics info")
            return []

        return p.getDynamicsInfo(robot_id, link_index)

    def change_dynamics(
        self, robot_id: int, link_index: int = -1, **kwargs: Dict[str, Any]
    ) -> None:
        """
        Change dynamics properties of a robot body/link.

        Args:
            robot_id (int): The ID of the robot
            link_index (int): The link index (-1 for base)
            **kwargs: Dynamics properties to change (mass, friction, etc.)
        """
        if not self.connected or not p.isConnected():
            logger.warning("Simulation is not connected, cannot change dynamics")
            return

        p.changeDynamics(robot_id, link_index, **kwargs)


def get_sim() -> PyBulletSimulation:
    global sim

    if sim is None:
        from phosphobot.configs import config

        sim = PyBulletSimulation(sim_mode=config.SIM_MODE)

    return sim

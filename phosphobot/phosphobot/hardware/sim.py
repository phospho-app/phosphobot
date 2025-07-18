"""
MuJoCo Simulation wrapper class
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from loguru import logger

sim = None


class MuJoCoSimulation:
    """
    A comprehensive wrapper class for MuJoCo simulation environment.
    """

    def __init__(self, sim_mode="headless"):
        """
        Initialize the MuJoCo simulation environment.

        Args:
            sim_mode (str): Simulation mode - "headless" or "gui"
        """
        self.sim_mode = sim_mode
        self.connected = False
        self.robots = {}  # Store loaded robots
        self.model = None
        self.data = None
        self.viewer = None
        self._gui_proc = None
        self.init_simulation()

    def init_simulation(self):
        """
        Initialize the MuJoCo simulation environment based on the configuration.
        """
        # Create a basic scene
        xml_content = """
        <mujoco model="basic_scene">
            <compiler angle="radian"/>
            <default>
                <joint damping="0.2" frictionloss="0.1"/>
                <geom friction="1.0 0.005 0.0001"/>
            </default>
            <worldbody>
                <geom pos="0 0 0" size="0 0 .125" type="plane" rgba="0.5 0.5 0.5 1" name="floor"/>
                <light pos="0 0 3" dir="0 0 -1" directional="false"/>
            </worldbody>
        </mujoco>
        """
        
        try:
            self.model = mujoco.MjModel.from_xml_string(xml_content)
            self.data = mujoco.MjData(self.model)
            self.connected = True
            
            if self.sim_mode == "headless":
                logger.debug("Simulation: headless mode enabled")
                
            elif self.sim_mode == "gui":
                self._start_gui_process()
                logger.debug("Simulation: GUI mode enabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize MuJoCo simulation: {e}")
            raise

    def _start_gui_process(self):
        """Start the GUI simulation process (similar to PyBullet approach)."""
        try:
            absolute_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "simulation",
                    "mujoco",
                )
            )

            def _stream_to_console(pipe):
                """Continuously read from *pipe* and write to stdout."""
                try:
                    with pipe:
                        for line in iter(pipe.readline, b""):
                            sys.stdout.write(
                                "[gui sim] " + line.decode("utf-8", errors="replace")
                            )
                            sys.stdout.flush()
                except Exception as exc:
                    logger.warning(f"Error while reading child stdout: {exc}")

            # start mujoco 
            self._gui_proc = subprocess.Popen(
                ["uv", "run", "python", "main.py"],
                cwd=absolute_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            t = threading.Thread(
                target=_stream_to_console, args=(self._gui_proc.stdout,), daemon=True
            )
            t.start()

            # Wait for simulation to start
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Failed to launch GUI process: {e}")
            self.sim_mode = "headless"

    def stop(self):
        """
        Cleanup the simulation environment.
        """
        if self.viewer and hasattr(self.viewer, 'is_running') and self.viewer.is_running():
            self.viewer.close()
            
        if self.sim_mode == "gui":
            if hasattr(self, "_gui_proc") and self._gui_proc.poll() is None:
                self._gui_proc.terminate()
                try:
                    self._gui_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._gui_proc.kill()
            
        self.connected = False
        logger.info("Simulation disconnected")

    def __del__(self):
        """
        Cleanup when object is destroyed.
        """
        self.stop()

    def reset(self):
        """
        Reset the simulation environment.
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot reset")
            return

        # Reset the simulation state
        mujoco.mj_resetData(self.model, self.data)
        self.robots.clear()
        logger.info("Simulation reset")

    def step(self, steps=960):
        """
        Step the simulation environment.

        Args:
            steps (int): Number of simulation steps to execute
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot step")
            return

        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            
        if self.viewer and hasattr(self.viewer, 'sync'):
            self.viewer.sync()

    def set_joint_state(self, robot_id, joint_id: int, joint_position: float):
        """
        Set the joint state of a robot in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_id (int): The ID of the joint to set.
            joint_position (float): The position to set the joint to.
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot set joint state")
            return

        try:
            robot_info = self.robots.get(robot_id)
            if robot_info and joint_id < len(robot_info["actuated_joints"]):
                mj_joint_id = robot_info["actuated_joints"][joint_id]
                if mj_joint_id < self.model.nq:
                    self.data.qpos[mj_joint_id] = joint_position
                    mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            logger.warning(f"Failed to set joint state: {e}")

    def inverse_dynamics(
        self, robot_id, positions: list, velocities: list, accelerations: list
    ):
        """
        Perform inverse dynamics to compute joint torques.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            positions (list): Joint positions
            velocities (list): Joint velocities
            accelerations (list): Joint accelerations

        Returns:
            list: Joint torques
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot perform inverse dynamics")
            return []

        try:
            qacc = np.array(accelerations, dtype=np.float64)
            qfrc = np.zeros(self.model.nv)
            mujoco.mj_inverse(self.model, self.data, qacc, qfrc)
            return qfrc.tolist()
        except Exception as e:
            logger.warning(f"Failed to compute inverse dynamics: {e}")
            return []

    def load_urdf(
        self,
        urdf_path: str,
        axis: list[float] | None = None,
        axis_orientation: list[int] = [0, 0, 0, 1],
        use_fixed_base: bool = True,
    ):
        """
        Load a robot model into the simulation.
        For MuJoCo, prefer MJCF format but support URDF conversion.

        Args:
            urdf_path (str): The path to the URDF/MJCF file.
            axis (list[float] | None): The position of the robot.
            axis_orientation (list[int]): The orientation of the robot.
            use_fixed_base (bool): Whether to use a fixed base for the robot.

        Returns:
            tuple: (robot_id, num_joints, actuated_joints)
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot load model")
            return None, 0, []

        try:
            # check if we have a MuJoCo XML version of this model
            urdf_dir = Path(urdf_path).parent
            mjcf_path = urdf_dir / "robot.xml"
            
            if mjcf_path.exists():
                logger.info(f"Loading MuJoCo model: {mjcf_path}")
                
                model_xml = mjcf_path.read_text()
                assets = {}
                
                for mesh_file in urdf_dir.glob("*.stl"):
                    assets[mesh_file.name] = mesh_file.read_bytes()
                
                model = mujoco.MjModel.from_xml_string(model_xml, assets)
                
                self.model = model
                self.data = mujoco.MjData(model)
                
                # restart viewer if in GUI mode
                if self.sim_mode == "gui" and self.viewer:
                    self.viewer.close()
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                
            else:
                # Fallback to basic simulation
                logger.warning(f"No MuJoCo XML found for {urdf_path}. Using basic simulation.")
                return None, 0, []

            actuated_joints = []
            for i in range(model.njnt):
                if model.jnt_type[i] in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                    actuated_joints.append(i)

            robot_id = len(self.robots)
            self.robots[robot_id] = {
                "model_path": str(mjcf_path) if mjcf_path.exists() else urdf_path,
                "num_joints": model.njnt,
                "actuated_joints": actuated_joints,
                "model": model,
            }

            logger.info(f"Loaded robot {robot_id} with {model.njnt} joints")
            return robot_id, model.njnt, actuated_joints
            
        except Exception as e:
            logger.error(f"Failed to load model {urdf_path}: {e}")
            return None, 0, []

    def set_joints_states(self, robot_id, joint_indices, target_positions):
        """
        Set multiple joint states of a robot in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_indices (list[int]): The indices of the joints to set.
            target_positions (list[float]): The positions to set the joints to.
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot set joint states")
            return

        try:
            robot_info = self.robots.get(robot_id)
            if robot_info:
                for i, pos in zip(joint_indices, target_positions):
                    if i < len(robot_info["actuated_joints"]):
                        mj_joint_id = robot_info["actuated_joints"][i]
                        if mj_joint_id < self.model.nq:
                            self.data.qpos[mj_joint_id] = pos
                            
                mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            logger.warning(f"Failed to set joint states: {e}")

    def get_joint_state(self, robot_id, joint_index: int) -> list:
        """
        Get the state of a joint in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_index (int): The index of the joint to get.

        Returns:
            list: [position, velocity, reaction_forces, applied_torque]
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot get joint state")
            return []

        try:
            robot_info = self.robots.get(robot_id)
            if robot_info and joint_index < len(robot_info["actuated_joints"]):
                mj_joint_id = robot_info["actuated_joints"][joint_index]
                
                position = self.data.qpos[mj_joint_id] if mj_joint_id < self.model.nq else 0.0
                velocity = self.data.qvel[mj_joint_id] if mj_joint_id < self.model.nv else 0.0
                
                return [position, velocity, 0.0, 0.0]
        except Exception as e:
            logger.warning(f"Failed to get joint state: {e}")
            
        return []

    def inverse_kinematics(
        self,
        robot_id,
        end_effector_link_index: int,
        target_position,
        target_orientation,
        rest_poses: list,
        joint_damping: list | None = None,
        lower_limits: list | None = None,
        upper_limits: list | None = None,
        joint_ranges: list | None = None,
        max_num_iterations: int = 200,
        residual_threshold: float = 1e-6,
    ) -> list:
        """
        Perform inverse kinematics using MuJoCo's IK solver.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            end_effector_link_index (int): The index of the end-effector link.
            target_position (list): The target position for the end-effector.
            target_orientation (list): The target orientation for the end-effector.
            rest_poses (list): Rest poses for the joints.
            ... (other args for compatibility)

        Returns:
            list: Joint angles computed by inverse kinematics.
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot perform inverse kinematics")
            return []

        try:
            # TODO: MuJoCo IK implementation would go here
            # For now, return rest poses as fallback
            logger.debug("MuJoCo IK not yet fully implemented, returning rest poses")
            return rest_poses
            
        except Exception as e:
            logger.warning(f"Failed to compute inverse kinematics: {e}")
            return []

    def get_link_state(
        self, robot_id, link_index: int, compute_forward_kinematics: bool = False
    ) -> list:
        """
        Get the state of a link in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            link_index (int): The index of the link to get.
            compute_forward_kinematics (bool): Whether to compute forward kinematics.

        Returns:
            list: [position, orientation, linear_velocity, angular_velocity, ...]
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot get link state")
            return []

        try:
            if compute_forward_kinematics:
                mujoco.mj_forward(self.model, self.data)
                
            if link_index < self.model.nbody:
                pos = self.data.xpos[link_index].copy()
                quat = self.data.xquat[link_index].copy()
                
                return [pos.tolist(), quat.tolist(), [0, 0, 0], [0, 0, 0]]
                
        except Exception as e:
            logger.warning(f"Failed to get link state: {e}")
            
        return []

    def get_joint_info(self, robot_id, joint_index: int) -> list:
        """
        Get the information of a joint in the simulation.

        Args:
            robot_id (int): The ID of the robot in the simulation.
            joint_index (int): The index of the joint to get.

        Returns:
            list: Joint information compatible with PyBullet format
        """
        if not self.connected:
            logger.warning("Simulation is not connected, cannot get joint info")
            return []

        try:
            robot_info = self.robots.get(robot_id)
            if robot_info and joint_index < len(robot_info["actuated_joints"]):
                mj_joint_id = robot_info["actuated_joints"][joint_index]
                
                joint_name = f"joint_{joint_index}".encode('utf-8')
                joint_type = self.model.jnt_type[mj_joint_id] if mj_joint_id < self.model.njnt else 0
                
                return [joint_index, joint_name, joint_type, -1, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, b'']
                
        except Exception as e:
            logger.warning(f"Failed to get joint info: {e}")
            
        return []

    def add_debug_text(self, text: str, text_position, text_color_RGB: list, life_time: int = 3):
        """Add debug text to the simulation."""
        logger.debug(f"Debug text: {text} at {text_position}")

    def add_debug_points(self, point_positions: list, point_colors_RGB: list, point_size: int = 4, life_time: int = 3):
        """Add debug points to the simulation."""
        logger.debug(f"Debug points: {len(point_positions)} points")

    def add_debug_lines(self, line_from_XYZ: list, line_to_XYZ: list, line_color_RGB: list, line_width: int = 4, life_time: int = 3):
        """Add debug lines to the simulation."""
        logger.debug(f"Debug line from {line_from_XYZ} to {line_to_XYZ}")

    def get_robot_info(self, robot_id):
        """Get information about a loaded robot."""
        return self.robots.get(robot_id, {})

    def get_all_robots(self):
        """Get all loaded robots."""
        return self.robots

    def is_connected(self):
        """Check if the simulation is connected."""
        return self.connected

    def set_gravity(self, gravity_vector: list = [0, 0, -9.81]):
        """Set the gravity vector for the simulation."""
        if not self.connected:
            logger.warning("Simulation is not connected, cannot set gravity")
            return

        try:
            self.model.opt.gravity[:] = gravity_vector
            logger.debug(f"Gravity set to {gravity_vector}")
        except Exception as e:
            logger.warning(f"Failed to set gravity: {e}")

    def get_dynamics_info(self, robot_id, link_index: int = -1):
        """Get dynamics information for a robot body/link."""
        if not self.connected:
            logger.warning("Simulation is not connected, cannot get dynamics info")
            return []
        return []

    def change_dynamics(self, robot_id, link_index: int = -1, **kwargs):
        """Change dynamics properties of a robot body/link."""
        if not self.connected:
            logger.warning("Simulation is not connected, cannot change dynamics")
            return
        logger.debug(f"Change dynamics called for robot {robot_id}, link {link_index}")


def get_sim() -> MuJoCoSimulation:
    """Get the global simulation instance."""
    global sim

    if sim is None:
        from phosphobot.configs import config
        sim = MuJoCoSimulation(sim_mode=config.SIM_MODE)

    return sim

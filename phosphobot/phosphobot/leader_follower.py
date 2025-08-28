import asyncio
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from loguru import logger

from phosphobot.control_signal import ControlSignal
from phosphobot.hardware import (
    SO100Hardware,
    PiperHardware,
    RemotePhosphobot,
    get_sim,
    URDFLoader,
)
from phosphobot.utils import background_task_log_exceptions


@dataclass
class RobotPair:
    leader: SO100Hardware | PiperHardware | RemotePhosphobot | URDFLoader
    follower: SO100Hardware | PiperHardware | RemotePhosphobot | URDFLoader


class LeaderFollowerRunner:
    """
    This class encapsulates the leader-follower logic and is designed to be
    run within a dedicated thread.
    """

    def __init__(
        self,
        robot_pairs: list[RobotPair],
        control_signal: ControlSignal,
        invert_controls: bool,
        enable_gravity_compensation: bool,
        compensation_values: Optional[Dict[str, int]],
        sim=get_sim(),
    ):
        self.robot_pairs = robot_pairs
        self.control_signal = control_signal
        self.invert_controls = invert_controls
        self.enable_gravity_compensation = enable_gravity_compensation
        self.compensation_values = compensation_values
        self.sim = sim

    def run_in_thread(self):
        """
        Sets up a new asyncio event loop for the current thread and runs
        the main control loop. This method is the target for the thread.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_async_loop())
        finally:
            loop.close()

    async def _run_async_loop(self):
        """
        The original async logic of the leader-follower control loop.
        - Applies gravity compensation to the leader
        - Makes the follower mirror the leader's current joint positions
        """
        logger.info(
            f"Starting leader-follower control in a new thread with {len(self.robot_pairs)} pairs of robots:"
            + ", ".join(
                f"{pair.leader.name} -> {pair.follower.name}"
                for pair in self.robot_pairs
            )
            + f"\ninvert_controls={self.invert_controls}\nenable_gravity_compensation={self.enable_gravity_compensation}"
            + (
                f"\ncompensation_values={self.compensation_values}"
                if self.compensation_values
                else ""
            )
        )
        loop_period = 1 / 150 if not self.enable_gravity_compensation else 1 / 60

        # Check if the initial position is set, otherwise move them
        wait_for_initial_position = False
        for pair in self.robot_pairs:
            for robot in [pair.leader, pair.follower]:
                if (
                    robot.initial_position is None
                    or robot.initial_orientation_rad is None
                ):
                    logger.warning(
                        f"Initial position or orientation not set for {robot.name} {robot.device_name}. "
                        "Moving to initial position before starting leader-follower control."
                    )
                    robot.enable_torque()
                    await robot.move_to_initial_position()
                    wait_for_initial_position = True
        if wait_for_initial_position:
            # Give some time for the robots to move to initial position
            await asyncio.sleep(1)

        # Enable torque and set PID gains
        default_p_gains, default_d_gains = await self._setup_robots()

        # We display a warning only once if the leader has more joints than the follower
        warning_dropping_joints_displayed = False
        # Main control loop
        while self.control_signal.is_in_loop():
            start_time = time.perf_counter()

            for pair in self.robot_pairs:
                if not self.enable_gravity_compensation:
                    await self._simple_follow(pair, warning_dropping_joints_displayed)
                else:
                    await self._gravity_compensated_follow(
                        pair, warning_dropping_joints_displayed
                    )

            # Maintain loop frequency
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, loop_period - elapsed)
            await asyncio.sleep(sleep_time)

        # Cleanup
        await self._cleanup(default_p_gains, default_d_gains)
        logger.info("Leader-follower control stopped")

    async def _setup_robots(self):
        """Initializes robots, enabling torque and setting PID gains."""
        # Default gains are returned for cleanup
        default_p_gains = [12, 20, 20, 20, 20, 20]
        default_d_gains = [36, 36, 36, 32, 32, 32]

        for pair in self.robot_pairs:
            leader, follower = pair.leader, pair.follower
            follower.enable_torque()

            if not self.enable_gravity_compensation:
                leader.disable_torque()
                p_gains = [12, 12, 12, 12, 12, 12]
                d_gains = [32, 32, 32, 32, 32, 32]
                if isinstance(follower, SO100Hardware):
                    for i in range(6):
                        follower._set_pid_gains_motors(
                            servo_id=i + 1,
                            p_gain=p_gains[i],
                            i_gain=0,
                            d_gain=d_gains[i],
                        )
                        await asyncio.sleep(0.05)
            else:
                await self._setup_gravity_compensation(leader)

        return default_p_gains, default_d_gains

    async def _setup_gravity_compensation(self, leader):
        """Sets up a leader robot for gravity compensation."""
        assert isinstance(
            leader, SO100Hardware
        ), "Gravity compensation is only supported for SO100Hardware."

        leader_current_voltage = leader.current_voltage()
        if (
            leader_current_voltage is None
            or np.isnan(np.mean(leader_current_voltage))
            or np.mean(leader_current_voltage) < 10
        ):
            logger.warning(
                "Leader motor voltage is NaN. Please calibrate the robot and check the USB connection."
            )
            self.control_signal.stop()
            return

        voltage = "6V" if np.mean(leader_current_voltage) < 9.0 else "12V"
        p_gains = [3, 6, 6, 3, 3, 3]
        d_gains = [9, 9, 9, 9, 9, 9]

        if voltage == "12V":
            p_gains = [int(p / 2) for p in p_gains]
            d_gains = [int(d / 2) for d in d_gains]

        leader.enable_torque()
        for i in range(6):
            leader._set_pid_gains_motors(
                servo_id=i + 1, p_gain=p_gains[i], i_gain=0, d_gain=d_gains[i]
            )
            await asyncio.sleep(0.05)

    async def _simple_follow(self, pair: RobotPair, warning_displayed: bool):
        """Follower mirrors leader's position without gravity compensation."""
        leader, follower = pair.leader, pair.follower
        pos_rad = leader.read_joints_position(unit="rad")

        if any(np.isnan(pos_rad)):
            logger.warning("Leader joint positions contain NaN values. Skipping.")
            return

        if self.invert_controls:
            pos_rad[0] = -pos_rad[0]

        follower.control_gripper(
            open_command=leader._rad_to_open_command(
                pos_rad[leader.GRIPPER_JOINT_INDEX]
            )
        )

        if len(pos_rad) > len(follower.SERVO_IDS):
            if not warning_displayed:
                logger.warning(
                    f"Leader has more joints than follower ({len(pos_rad)} > {len(follower.SERVO_IDS)}). Dropping extra joints."
                )
                warning_displayed = True
            pos_rad = pos_rad[: len(follower.SERVO_IDS)]

        follower.set_motors_positions(q_target_rad=pos_rad, enable_gripper=False)

    async def _gravity_compensated_follow(
        self, pair: RobotPair, warning_displayed: bool
    ):
        """Follower mirrors leader with gravity compensation applied to the leader."""
        leader, follower = pair.leader, pair.follower
        assert isinstance(leader, SO100Hardware) and isinstance(
            follower, SO100Hardware
        ), "Gravity compensation is only supported for SO100Hardware."

        num_joints = len(leader.actuated_joints)
        pos_rad = leader.read_joints_position(unit="rad")

        if any(np.isnan(pos_rad)):
            logger.warning("Leader joint positions contain NaN values. Skipping.")
            return

        for i, idx in enumerate(range(num_joints)):
            self.sim.set_joint_state(leader.p_robot_id, idx, pos_rad[i])

        tau_g = self.sim.inverse_dynamics(
            leader.p_robot_id,
            positions=list(pos_rad),
            velocities=[0.0] * num_joints,
            accelerations=[0.0] * num_joints,
        )

        if self.compensation_values:
            tau_g = self._apply_compensation(list(tau_g))

        alpha = np.array([0, 0.2, 0.2, 0.1, 0.2, 0.2])
        theta_des_rad = pos_rad + alpha[:num_joints] * np.array(tau_g)

        leader.write_joint_positions(theta_des_rad, unit="rad")

        if self.invert_controls:
            theta_des_rad[0] = -theta_des_rad[0]

        follower.control_gripper(
            open_command=leader._rad_to_open_command(
                theta_des_rad[leader.GRIPPER_JOINT_INDEX]
            )
        )

        if len(pos_rad) > len(follower.SERVO_IDS):
            if not warning_displayed:
                logger.warning(
                    f"Leader has more joints than follower ({len(pos_rad)} > {len(follower.SERVO_IDS)}). Dropping extra joints."
                )
                warning_displayed = True
            theta_des_rad = theta_des_rad[: len(follower.SERVO_IDS)]

        follower.set_motors_positions(q_target_rad=theta_des_rad, enable_gripper=False)

    def _apply_compensation(self, tau_g: list):
        """Applies custom compensation values to gravity torque vector."""
        for key, value in self.compensation_values.items():
            if key == "shoulder":
                tau_g[1] *= value / 100
            elif key == "elbow":
                tau_g[2] *= value / 100
            elif key == "wrist":
                tau_g[3] *= value / 100
            else:
                logger.debug(f"Unknown compensation key: {key}")
        return tau_g

    async def _cleanup(self, default_p_gains, default_d_gains):
        """Resets PID gains and disables torque on all robots."""
        for pair in self.robot_pairs:
            leader, follower = pair.leader, pair.follower
            leader.enable_torque()
            if isinstance(leader, SO100Hardware):
                for i in range(6):
                    leader._set_pid_gains_motors(
                        servo_id=i + 1,
                        p_gain=default_p_gains[i],
                        i_gain=0,
                        d_gain=default_d_gains[i],
                    )
                    await asyncio.sleep(0.05)
            leader.disable_torque()
            follower.disable_torque()


@background_task_log_exceptions
async def start_leader_follower_loop(
    robot_pairs: list[RobotPair],
    control_signal: ControlSignal,
    invert_controls: bool,
    enable_gravity_compensation: bool,
    compensation_values: Optional[Dict[str, int]],
    sim=get_sim(),
):
    """
    FastAPI background task that spins up a dedicated thread to run the
    leader-follower control loop, ensuring the main async loop is not blocked.
    """
    runner = LeaderFollowerRunner(
        robot_pairs=robot_pairs,
        control_signal=control_signal,
        invert_controls=invert_controls,
        enable_gravity_compensation=enable_gravity_compensation,
        compensation_values=compensation_values,
        sim=sim,
    )

    # Create and start the dedicated thread
    thread = threading.Thread(target=runner.run_in_thread, daemon=True)
    thread.start()
    logger.info(f"Leader-follower thread started.")

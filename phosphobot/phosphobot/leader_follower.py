import asyncio
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from phosphobot.control_signal import ControlSignal
from phosphobot.hardware import SO100Hardware, PiperHardware, RemotePhosphobot, get_sim


@dataclass
class RobotPair:
    leader: SO100Hardware | PiperHardware | RemotePhosphobot
    follower: SO100Hardware | PiperHardware | RemotePhosphobot


async def leader_follower_loop(
    robot_pairs: list[RobotPair],
    control_signal: ControlSignal,
    invert_controls: bool,
    enable_gravity_compensation: bool,
    compensation_values: dict[str, int] | None,
    sim=get_sim(),
):
    """
    Background task that implements leader-follower control:
    - Applies gravity compensation to the leader
    - Makes the follower mirror the leader's current joint positions
    """
    logger.info("Starting leader-follower control.")
    loop_period = 1 / 150 if not enable_gravity_compensation else 1 / 60

    # Check if the initial position is set, otherwise move them
    wait_for_initial_position = False
    for pair in robot_pairs:
        for robot in [pair.leader, pair.follower]:
            if robot.initial_position is None or robot.initial_orientation_rad is None:
                logger.warning(
                    f"Initial position or orientation not set for {robot.name} {robot.device_name}"
                    "Moving to initial position before starting leader-follower control."
                )
                robot.enable_torque()
                await robot.move_to_initial_position()
                wait_for_initial_position = True
    if wait_for_initial_position:
        # Give some time for the robots to move to initial position
        await asyncio.sleep(1)

    # Enable torque if using gravity compensation, and on the follower
    for pair in robot_pairs:
        leader = pair.leader
        follower = pair.follower

        follower.enable_torque()
        if not enable_gravity_compensation:
            leader.disable_torque()
            if isinstance(follower, SO100Hardware):
                p_gains = [12, 12, 12, 12, 12, 12]
                d_gains = [32, 32, 32, 32, 32, 32]
                default_p_gains = [12, 20, 20, 20, 20, 20]
                default_d_gains = [36, 36, 36, 32, 32, 32]

                for i in range(6):
                    follower._set_pid_gains_motors(
                        servo_id=i + 1,
                        p_gain=p_gains[i],
                        i_gain=0,
                        d_gain=d_gains[i],
                    )
                    await asyncio.sleep(0.05)
        else:
            assert isinstance(
                leader, SO100Hardware
            ), "Gravity compensation is only supported for SO100Hardware."
            assert isinstance(
                follower, SO100Hardware
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
                control_signal.stop()
                return

            voltage = "6V" if np.mean(leader_current_voltage) < 9.0 else "12V"

            # Define PID gains for all six motors
            p_gains = [3, 6, 6, 3, 3, 3]
            d_gains = [9, 9, 9, 9, 9, 9]
            default_p_gains = [12, 20, 20, 20, 20, 20]
            default_d_gains = [36, 36, 36, 32, 32, 32]
            alpha = np.array([0, 0.2, 0.2, 0.1, 0.2, 0.2])

            if voltage == "12V":
                p_gains = [int(p / 2) for p in p_gains]
                d_gains = [int(d / 2) for d in d_gains]
                default_p_gains = [6, 6, 6, 10, 10, 10]
                default_d_gains = [30, 15, 15, 30, 30, 30]
            leader.enable_torque()

            # Apply custom PID gains to leader for all six motors
            for i in range(6):
                leader._set_pid_gains_motors(
                    servo_id=i + 1,
                    p_gain=p_gains[i],
                    i_gain=0,
                    d_gain=d_gains[i],
                )
                await asyncio.sleep(0.05)

    # Main control loop
    while control_signal.is_in_loop():
        start_time = time.perf_counter()

        for pair in robot_pairs:
            leader = pair.leader
            follower = pair.follower

            # Control loop parameters
            num_joints = len(leader.actuated_joints)
            joint_indices = list(range(num_joints))

            # Get leader's current joint positions
            pos_rad = leader.read_joints_position(unit="rad")

            if any(np.isnan(pos_rad)):
                logger.warning(
                    "Leader joint positions contain NaN values. Skipping this iteration."
                )
                continue

            if not enable_gravity_compensation:
                # Simple leader-follower control: follower mirrors leader's position
                if invert_controls:
                    pos_rad[0] = -pos_rad[0]
                follower.set_motors_positions(
                    q_target_rad=pos_rad, enable_gripper=False
                )
                # Get the leader gripper position and set it to the follower
                follower.control_gripper(
                    open_command=leader._rad_to_open_command(
                        pos_rad[leader.GRIPPER_JOINT_INDEX]
                    )
                )

            else:
                assert isinstance(
                    leader, SO100Hardware
                ), "Gravity compensation is only supported for SO100Hardware."
                assert isinstance(
                    follower, SO100Hardware
                ), "Gravity compensation is only supported for SO100Hardware."
                # Calculate gravity compensation torque
                # Update PyBullet simulation for gravity calculation
                for i, idx in enumerate(joint_indices):
                    sim.set_joint_state(leader.p_robot_id, idx, pos_rad[i])

                positions = list(pos_rad)
                velocities = [0.0] * num_joints
                accelerations = [0.0] * num_joints
                tau_g = sim.inverse_dynamics(
                    leader.p_robot_id,
                    positions=positions,
                    velocities=velocities,
                    accelerations=accelerations,
                )
                tau_g = list(tau_g)

                if compensation_values is not None:
                    for key in compensation_values.keys():
                        if key == "shoulder":
                            tau_g[1] = compensation_values[key] * tau_g[1] / 100
                        elif key == "elbow":
                            tau_g[2] = compensation_values[key] * tau_g[2] / 100
                        elif key == "wrist":
                            tau_g[3] = compensation_values[key] * tau_g[3] / 100
                        else:
                            logger.debug(f"Unknow key: {key}")
                            continue

                # Apply gravity compensation to leader
                theta_des_rad = pos_rad + alpha[:num_joints] * np.array(tau_g)
                leader.write_joint_positions(theta_des_rad, unit="rad")
                if invert_controls:
                    theta_des_rad[0] = -theta_des_rad[0]
                follower.set_motors_positions(
                    q_target_rad=theta_des_rad, enable_gripper=False
                )
                # Get the leader gripper position and set it to the follower
                follower.control_gripper(
                    open_command=leader._rad_to_open_command(
                        theta_des_rad[leader.GRIPPER_JOINT_INDEX]
                    )
                )

        # Maintain loop frequency
        elapsed = time.perf_counter() - start_time
        sleep_time = max(0, loop_period - elapsed)
        await asyncio.sleep(sleep_time)

    # Cleanup: Reset leader's PID gains to default for all six motors
    for pair in robot_pairs:
        leader = pair.leader
        follower = pair.follower

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

        # Disable torque
        leader.disable_torque()
        follower.disable_torque()

    logger.info("Leader-follower control stopped")

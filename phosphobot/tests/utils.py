import time

import numpy as np
import pytest
from phosphobot.hardware import get_sim
from phosphobot.hardware.base import BaseManipulator


def compare_angles_radian(
    angle1: float | np.ndarray,
    angle2: float | np.ndarray,
    atol=1e-2,  # 1e-2 is roughly 1 degree
):
    """
    Compare two angles in radians, accounting for modulo 2π equivalence.

    Parameters:
        angle1 (float or np.ndarray): First angle(s) in radians.
        angle2 (float or np.ndarray): Second angle(s) in radians.
        atol (float): Absolute tolerance for the comparison (default 1e-2).

    Returns:
        bool or np.ndarray: True if angles are equivalent within the tolerance, otherwise False.
    """
    # Compute the difference and normalize to [0, 2π)
    angle_diff = angle1 - angle2
    angle_diff_mod = np.mod(angle_diff, 2 * np.pi)

    # Determine closeness to either 0 or 2π
    closest_zero_or_2pi = np.minimum(angle_diff_mod, 2 * np.pi - angle_diff_mod)

    # Compare using np.allclose (for scalars or arrays)
    max_error = np.max(closest_zero_or_2pi)
    return np.allclose(closest_zero_or_2pi, 0, atol=atol), max_error


@pytest.mark.asyncio
async def move_robot_testing(
    robot: BaseManipulator,
    delta_position: np.ndarray,
    delta_orientation_rad: np.ndarray,
    atol_pos=0.05,  # in meters
    atol_rot=10,  # in degrees
):
    """
    Utils function used to test the movement of a robot given a delta position and orientation.
    """
    # We now test if the robot can move 1 cm in the x direction
    robot.set_simulation_positions(np.zeros(6))

    # This steps the simulation to update the robot's position
    robot.sim.step(steps=600)
    time.sleep(0.1)  # Allow some time for the simulation to update

    # Calculate the start position
    start_position, start_orientation = robot.forward_kinematics()

    # Move the robot 1 cm in the x direction
    theoretical_position = start_position + delta_position
    theoretical_rotation = start_orientation + delta_orientation_rad

    await robot.move_robot_absolute(
        target_position=theoretical_position,
        target_orientation_rad=theoretical_rotation,
    )

    robot.sim.step(steps=600)
    time.sleep(0.1)  # Allow some time for the simulation to update

    updated_position, updated_rotation = robot.forward_kinematics()

    max_position_diff = np.max(np.abs(updated_position - theoretical_position))
    assert (
        max_position_diff < atol_pos
    ), f"{robot.name} *move* error: {max_position_diff * 100:.3f}cm. Theoretical: {theoretical_position}, Actual: {updated_position}"

    angles_are_close, max_angle_diff = compare_angles_radian(
        updated_rotation, theoretical_rotation, atol=atol_rot
    )
    # Convert to degree
    max_angle_diff = np.degrees(max_angle_diff)

    assert (
        max_angle_diff < atol_rot
    ), f"{robot.name} *rotate* error: {max_angle_diff:.3f}° Theoretical: {theoretical_rotation}, Actual: {updated_rotation}"

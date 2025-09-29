"""
Tests for the teleop Base Robot class.

```
run uv pytest tests/test_base.py
```
"""

import os
import sys
import time

import numpy as np
import pytest
from loguru import logger
from utils import move_robot_testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phosphobot.configs import config
from phosphobot.hardware import KochHardware, SO100Hardware, get_sim
from phosphobot.hardware.base import BaseManipulator
from phosphobot.types import SimulationMode


# Create robot pytest fixture
@pytest.fixture
def robot(request):
    """
    Enables us to feed our different robot classes to the test functions
    """
    # Initialize the simulation
    config.SIM_MODE = SimulationMode.headless
    get_sim()

    return request.getfixturevalue(request.param)


@pytest.fixture
def koch():
    """
    Create a Koch robot instance
    """
    robot = KochHardware()

    return robot


@pytest.fixture
def so100():
    """
    Create a SO100 robot instance
    """
    robot = SO100Hardware()

    return robot


@pytest.mark.parametrize("robot", ["so100"], indirect=True)
@pytest.mark.asyncio
async def test_inverse_kinematics(robot: BaseManipulator):
    """
    Assert the function inverse_kinematics returns the correct angles
    """

    # Move to the initial position
    await robot.move_to_initial_position()
    robot.sim.step(steps=600)
    time.sleep(0.1)  # Allow some time for the simulation to update

    position = robot.initial_position
    orientation = robot.initial_orientation_rad

    assert (
        position is not None
    ), "Initial position should not be None after initialization"
    assert (
        orientation is not None
    ), "Initial orientation should not be None after initialization"

    q_robot_reference_rad = robot.read_joints_position()
    logger.info(f"q_robot_reference_rad: {q_robot_reference_rad}")

    q_robot_rad = robot.inverse_kinematics(position, orientation)
    logger.info(f"q_robot_rad: {q_robot_rad}")

    assert np.allclose(
        q_robot_rad, q_robot_reference_rad, rtol=0, atol=1e-2
    ), f"Forward + Inverse kinematics should be the same. {q_robot_rad} != {q_robot_reference_rad}"


@pytest.mark.parametrize("robot", ["so100"], indirect=True)
def test_forward_inverse_kinematics(robot: BaseManipulator):
    """
    Assert the functions forward kinematics is the inverse of inverse kinematics
    """

    q_robot_rad_reference = robot.read_joints_position()

    position, orientation = robot.forward_kinematics()

    # Inverse kinematics
    q_robot_rad = robot.inverse_kinematics(position, orientation)

    logger.info(f"q_robot_rad: {q_robot_rad}")
    logger.info(f"q_robot_rad_reference: {q_robot_rad_reference}")

    assert np.allclose(
        q_robot_rad, q_robot_rad_reference, rtol=0, atol=1e-3
    ), f"Joint angles should be the same. {q_robot_rad} != {q_robot_rad_reference}"


@pytest.mark.parametrize("robot", ["so100"], indirect=True)
def test_initial_position(robot: BaseManipulator):
    current_joint_positions = robot.read_joints_position()

    assert np.allclose(
        current_joint_positions,
        [0, 0, 0, 0, 0, 0],
    ), "Initial joint positions are not zero"


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_move_robot_no_move(robot: BaseManipulator):
    await move_robot_testing(robot, np.array([0, 0, 0]), np.array([0, 0, 0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_move_robot_forward(robot: BaseManipulator):
    await move_robot_testing(robot, np.array([0.015, 0, 0]), np.array([0, 0, 0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_move_robot_backward(robot: BaseManipulator):
    await move_robot_testing(robot, np.array([-0.02, 0, 0]), np.array([0, 0, 0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_move_robot_right(robot: BaseManipulator):
    """
    The SO100 robot can't move right without rotation its basis on the Z axis.
    """

    await move_robot_testing(
        robot,
        np.array([0, -0.1, 0]),
        np.deg2rad([0, 0, -30]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_move_robot_left(robot: BaseManipulator):
    await move_robot_testing(robot, np.array([0, 0.01, 0]), np.array([0, 0, 0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_move_robot_up(robot: BaseManipulator):
    await move_robot_testing(robot, np.array([0, 0, 0.02]), np.array([0, 0, 0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_move_robot_down(robot: BaseManipulator):
    await move_robot_testing(robot, np.array([0, 0, -0.02]), np.array([0, 0, 0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_rotate_robot_x(robot: BaseManipulator):
    await move_robot_testing(
        robot, np.array([0, 0, 0]), np.array([0.1, 0, 0]), atol_pos=1.5e-2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_rotate_robot_y(robot: BaseManipulator):
    await move_robot_testing(robot, np.array([0, 0, 0]), np.array([0, 0.1, 0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("robot", ["so100"], indirect=True)
async def test_rotate_robot_z(robot: BaseManipulator):
    await move_robot_testing(
        robot, np.array([0, 0, 0]), np.array([0, 0, 0.1]), atol_pos=2e-2
    )

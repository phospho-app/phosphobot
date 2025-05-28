import asyncio
from copy import copy
import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import time
from typing import Dict, Literal, Optional, Tuple, cast

import numpy as np
from fastapi import WebSocket
from loguru import logger
from pydantic import ValidationError

from phosphobot.hardware import BaseManipulator
from phosphobot.hardware.base import BaseMobileRobot
from phosphobot.models import AppControlData, RobotStatus, UDPServerInformationResponse
from phosphobot.robot import RobotConnectionManager
from phosphobot.utils import get_local_network_ip


@dataclass
class RobotState:
    last_update: datetime = datetime.now()
    last_timestamp: float = 0.0
    gripped: bool = False


class TeleopManager:
    robot_id: int | None
    rcm: RobotConnectionManager
    states: Dict[Literal["left", "right"], RobotState]
    action_counter: int
    last_report: datetime
    MOVE_TIMEOUT: float = 1.0  # seconds
    MAX_INSTRUCTIONS_PER_SEC: int = 200

    def __init__(self, rcm: RobotConnectionManager, robot_id: int | None = None):
        self.rcm = rcm
        self.states: Dict[str, RobotState] = {
            "left": RobotState(),
            "right": RobotState(),
        }
        self.action_counter = 0
        self.last_report = datetime.now()
        self.robot_id = robot_id

        # rate limiting window
        self._window_start: datetime = datetime.now()
        self._instr_in_window: int = 0

        self._robots: list[BaseManipulator | BaseMobileRobot] = []
        self.is_initializing: bool = False

    def allow_instruction(self) -> bool:
        """Simple 1-second sliding window rate limiter."""
        if self.is_initializing:
            logger.debug("Initialization in progress, skipping instruction allowance")
            return False

        now = datetime.now()
        if (now - self._window_start).total_seconds() >= 1.0:
            self._window_start = now
            self._instr_in_window = 0

        if self._instr_in_window < self.MAX_INSTRUCTIONS_PER_SEC:
            self._instr_in_window += 1
            return True

        return False

    async def get_manipulator_robot(self, source: str) -> Optional[BaseManipulator]:
        """
        Get the manipulator robot based on source
        - "right" returns the first manipulator robot
        - "left" returns the second manipulator robot

        This is a convention because 90% of people are right-handed.
        """
        right_robot = None
        left_robot = None

        for robot in self._robots:
            if isinstance(robot, BaseManipulator):
                if right_robot is None:
                    right_robot = robot
                elif left_robot is None:
                    left_robot = robot
                    break

        if source == "right":
            return right_robot
        elif source == "left":
            return left_robot
        else:
            logger.error(f"Unknown source '{source}' for manipulator robot")
            return None

    async def get_mobile_robot(self, source: str) -> Optional[BaseMobileRobot]:
        """
        Get the mobile robot.
        Currently, only one mobile robot is supported, the one linked to the "right" source.
        This means that only the "right" joystick can control the mobile robot.
        """
        if source == "right":
            for robot in self._robots:
                if isinstance(robot, BaseMobileRobot):
                    return robot

        return None

    async def move_init(self, robot_id: int | None = None) -> None:
        """
        Move the robot to the initial position.
        """
        if self.is_initializing:
            logger.debug("Initialization already in progress, skipping move_init")
            return

        self.is_initializing = True
        for i, robot in enumerate(await self.rcm.robots):
            if robot_id is not None and i != robot_id:
                continue
            robot.init_config()
            robot.enable_torque()
            # For Agilex Piper, we need to connect after enabling torque
            if robot.name == "agilex-piper":
                robot.connect()
            await robot.move_to_initial_position()

        # Hard block the code to wait for the robot to reach the initial position
        if any(robot.name == "agilex-piper" for robot in (await self.rcm.robots)):
            await asyncio.sleep(2.5)
        else:
            await asyncio.sleep(0.3)

        for i, robot in enumerate(await self.rcm.robots):
            if robot_id is not None and i != robot_id:
                continue
            if hasattr(robot, "forward_kinematics"):
                initial_position, initial_orientation_rad = robot.forward_kinematics()
                robot.initial_position = initial_position
                robot.initial_orientation_rad = initial_orientation_rad

        self._robots = copy(await self.rcm.robots)
        self.is_initializing = False

    async def _process_control_data_manipulator(
        self, control_data: AppControlData, robot: BaseManipulator
    ):
        """
        We transform the control data into a target position, orientation and gripper state.
        We then move the robot to that position and orientation.
        """
        if robot.initial_position is None or robot.initial_orientation_rad is None:
            await self.move_init()
        # Convert and execute command
        (
            target_pos,
            target_orient_deg,
            target_open,
        ) = control_data.to_robot(robot_name=robot.name)

        # TODO: Raise an error if initial_position or initial_orientation_rad is None ?
        initial_position = getattr(robot, "initial_position", None)
        initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)

        target_position = (
            target_pos + initial_position
            if initial_position is not None
            else target_pos
        )
        target_orientation_rad = (
            np.deg2rad(target_orient_deg) + initial_orientation_rad
            if initial_orientation_rad is not None
            else np.deg2rad(target_orient_deg)
        )

        # if robot.is_moving, wait for it to stop
        start_wait_time = time.perf_counter()
        while (
            robot.is_moving
            and time.perf_counter() - start_wait_time < self.MOVE_TIMEOUT
        ):
            await asyncio.sleep(0.00001)
        if time.perf_counter() - start_wait_time >= self.MOVE_TIMEOUT:
            logger.warning(
                f"Robot {robot.name} is still moving after {self.MOVE_TIMEOUT}s; skipping this command"
            )
            # skip gripper & counting if move failed
            return False

        try:
            # off-load blocking move_robot into threadpool + enforce timeout
            await asyncio.wait_for(
                robot.move_robot_absolute(target_position, target_orientation_rad),
                timeout=self.MOVE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"move_robot timed out after {self.MOVE_TIMEOUT}s; skipping this command"
            )
            # skip gripper & counting if move failed
            return False

        robot.control_gripper(open_command=target_open)
        robot.update_object_gripping_status()
        self.action_counter += 1

    async def _process_control_data_mobile_robot(
        self, control_data: AppControlData, robot: BaseMobileRobot
    ):
        """
        We use the control_data.direction_x and control_data.direction_y to move the mobile robot.
        These values are between -1 and 1, where 0 means no movement.
        - direction_y: forward/backward movement
        - direction_x: rotate left/right (rz axis rotation)
        """
        # TODO:
        # - deadzone detection
        # - rescaling (the joystick sensitivity is likely not linear)
        # - some trig ?
        # - progressive acceleration ?

        # deadzone: zero if below 0.3
        if abs(control_data.direction_x) < 0.3:
            control_data.direction_x = 0.0
        if abs(control_data.direction_y) < 0.3:
            control_data.direction_y = 0.0

        rz = -control_data.direction_x * np.pi
        x = control_data.direction_y

        # Adjust for max speed
        # TODO: better params
        rz *= 0.5
        x *= 0.4

        await robot.move_robot_absolute(
            target_position=np.array([x, 0, 0]),
            target_orientation_rad=np.array([0, 0, rz]),
        )
        self.action_counter += 1

    async def process_control_data(self, control_data: AppControlData) -> bool:
        """
        Process control data
        Returns:
            True if the command was processed successfully, False otherwise.
        """
        if self.is_initializing:
            logger.debug("Initialization in progress, skipping control data processing")
            return False

        state = self.states[control_data.source]
        if self._robots == []:
            self._robots = copy(await self.rcm.robots)

        # Check timestamp freshness
        if control_data.timestamp is not None:
            if control_data.timestamp <= state.last_timestamp:
                return False

            state.last_timestamp = control_data.timestamp

        # If robot_id is set, get the specific robot and move it accordingly
        if self.robot_id is not None:
            # robot = await self.rcm.get_robot(self.robot_id)
            try:
                robot = self._robots[self.robot_id]
            except IndexError:
                logger.warning(f"Robot ID {self.robot_id} not found in the robot list")
                return False
            if isinstance(robot, BaseManipulator):
                await self._process_control_data_manipulator(control_data, robot)
                return True
            elif isinstance(robot, BaseMobileRobot):
                await self._process_control_data_mobile_robot(control_data, robot)
                return True
            else:
                logger.error(f"Unknown robot type for robot_id {self.robot_id}")
                return False

        # Otherwise (ex: VR control), fetch the manipulators and mobile robots based on control_data.source
        manipulator_robot = await self.get_manipulator_robot(control_data.source)
        mobile_robot = await self.get_mobile_robot(control_data.source)

        # No robot -> nothing to do
        if manipulator_robot is None and mobile_robot is None:
            return False

        # Manipulator -> move it
        if manipulator_robot is not None:
            await self._process_control_data_manipulator(
                control_data, manipulator_robot
            )

        # Mobile robot -> move it
        if mobile_robot is not None:
            await self._process_control_data_mobile_robot(control_data, mobile_robot)

        return True

    async def send_status_updates(
        self, websocket: Optional[WebSocket] = None
    ) -> list[RobotStatus]:
        """Generate and optionally send status updates"""
        updates = []
        now = datetime.now()

        for source, state in self.states.items():
            robot = await self.get_manipulator_robot(source)
            if robot and (now - state.last_update).total_seconds() > 0.033:
                if isinstance(robot, BaseManipulator):
                    if state.gripped != robot.is_object_gripped:
                        state.gripped = robot.is_object_gripped
                        updates.append(
                            RobotStatus(
                                is_object_gripped=state.gripped,
                                is_object_gripped_source=source,
                                nb_actions_received=self.action_counter,
                            )
                        )

                state.last_update = now

        # Send periodic action count
        if (now - self.last_report).total_seconds() > 1:
            updates.append(RobotStatus(nb_actions_received=self.action_counter))
            self.action_counter = 0
            self.last_report = now

        # Send updates if websocket is provided
        if websocket:
            for update in updates:
                await websocket.send_text(update.model_dump_json())

        return updates


teleop_manager = None
udp_server = None


class _TeleopProtocol(asyncio.DatagramProtocol):
    def __init__(self, manager: TeleopManager):
        self.manager = manager
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport):
        self.transport = cast(asyncio.DatagramTransport, transport)
        sockname = transport.get_extra_info("sockname")
        logger.info(f"UDP socket opened on {sockname}")

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        # fire-and-forget per-packet handling
        asyncio.create_task(self._handle(data, addr))

    async def _handle(self, data: bytes, addr: Tuple[str, int]):
        if self.transport is None:
            logger.error("Transport is None, cannot handle datagram")
            return

        if not self.manager.allow_instruction():
            err = {
                "error": "rate_limited",
                "detail": f"Exceeded {self.manager.MAX_INSTRUCTIONS_PER_SEC} msgs/sec",
            }
            self.transport.sendto(json.dumps(err).encode("utf-8"), addr)
            return

        # 1) decode
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            err = {"error": "invalid_encoding", "detail": str(e)}
            self.transport.sendto(json.dumps(err).encode(), addr)
            logger.error(f"Decoding error from {addr}: {e}")
            return

        # 2) parse JSON
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            err = {"error": "invalid_json", "detail": e.msg}
            self.transport.sendto(json.dumps(err).encode(), addr)
            logger.error(f"JSON parse error from {addr}: {e.msg}")
            return

        # 3) validate schema
        try:
            control = AppControlData.model_validate(raw)
        except ValidationError as e:
            err = {"error": "validation_error", "detail": str(e)}
            self.transport.sendto(json.dumps(err).encode(), addr)
            logger.error(f"Schema validation failed from {addr}: {e}")
            return

        # 4) process and respond
        try:
            await self.manager.process_control_data(control)
            updates = await self.manager.send_status_updates()
            for u in updates:
                self.transport.sendto(u.model_dump_json().encode(), addr)
        except Exception as e:
            err = {"error": "internal_server_error", "detail": str(e)}
            self.transport.sendto(json.dumps(err).encode(), addr)
            logger.exception("Error while processing control data")


class UDPServer:
    def __init__(self, rcm: RobotConnectionManager):
        self.manager = get_teleop_manager()
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[_TeleopProtocol] = None
        self.bound_port: Optional[int] = None

    async def init(
        self, port: Optional[int] = None, restart: bool = False
    ) -> UDPServerInformationResponse:
        """
        Initialize (or re-init) the UDP server via asyncio.create_datagram_endpoint.
        Returns the bound host/port.
        """
        if self.transport is not None and not restart:
            host, bound_port = self.transport.get_extra_info("sockname")
            return UDPServerInformationResponse(host=host, port=bound_port)

        loop = asyncio.get_running_loop()
        local_ip = get_local_network_ip()

        # choose port
        if port is None:
            for p in range(5000, 6000):
                try:
                    transport, protocol = await loop.create_datagram_endpoint(
                        lambda: _TeleopProtocol(self.manager),
                        local_addr=(local_ip, p),
                    )
                    self.transport = transport
                    self.protocol = protocol
                    self.bound_port = p
                    logger.info(f"Bound UDP server to {local_ip}:{p}")
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("Could not bind to any port between 5000 and 6000")
        else:
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _TeleopProtocol(self.manager),
                local_addr=(local_ip, port),
            )
            self.transport = transport
            self.protocol = protocol
            self.bound_port = port
            logger.info(f"Bound UDP server to {local_ip}:{port}")

        host, bound_port = self.transport.get_extra_info("sockname")
        return UDPServerInformationResponse(host=host, port=bound_port)

    def stop(self) -> None:
        """
        Close the transport; no more packets will be received.
        """
        if self.transport:
            self.transport.close()
            logger.info("UDP server transport closed")
            self.transport = None
            self.protocol = None
            self.bound_port = None


@lru_cache()
def get_udp_server() -> UDPServer:
    """
    Get the UDP server instance.
    If it doesn't exist, create it.
    """
    from phosphobot.robot import get_rcm

    global udp_server
    if udp_server is None:
        udp_server = UDPServer(get_rcm())
    return udp_server


@lru_cache()
def get_teleop_manager() -> TeleopManager:
    """
    Get the TeleopManager instance.
    If it doesn't exist, create it.
    """
    from phosphobot.robot import get_rcm

    global teleop_manager
    if teleop_manager is None:
        teleop_manager = TeleopManager(get_rcm())
    return teleop_manager

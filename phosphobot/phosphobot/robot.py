import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Set

from async_property import async_property
from fastapi import HTTPException
from loguru import logger
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.configs import config
from phosphobot.hardware import (
    BaseRobot,
    URDFLoader,
    KochHardware,
    LeKiwi,
    PiperHardware,
    RemotePhosphobot,
    SO100Hardware,
    UnitreeGo2,
    WX250SHardware,
    get_sim,
)
from phosphobot.models import RobotConfigStatus
from phosphobot.utils import is_can_plugged

rcm = None

robot_name_to_class = {
    SO100Hardware.name: SO100Hardware,
    KochHardware.name: KochHardware,
    WX250SHardware.name: WX250SHardware,
    UnitreeGo2.name: UnitreeGo2,
    LeKiwi.name: LeKiwi,
    PiperHardware.name: PiperHardware,
    RemotePhosphobot.name: RemotePhosphobot,
    URDFLoader.name: URDFLoader,
}


@dataclass
class NewAndOldPorts:
    new_ports: List[ListPortInfo]
    old_ports: List[ListPortInfo]
    new_can_ports: List[str]
    old_can_ports: List[str]


class RobotConnectionManager:
    _all_robots: list[BaseRobot]
    _manually_added_robots: list[BaseRobot]

    available_ports: List[ListPortInfo]
    available_can_ports: List[str]
    last_scan_time: float

    def __init__(self):
        self.available_ports = []
        self.available_can_ports = []
        self.last_scan_time = 0

        self._all_robots = []
        self._manually_added_robots = []

    def __del__(self):
        # Disconnect all robots
        for robot in self._all_robots:
            robot.disconnect()

    def _scan_ports(self) -> tuple[list, list]:
        """
        Scan USB and CAN ports.
        """

        available_ports = list_ports.comports()

        if config.ENABLE_CAN:
            # Look for CAN ports
            can_ports = []
            for i in range(2):  # Adjust based on maximum expected CAN interfaces
                can_name = f"can{i}"
                if is_can_plugged(can_name):
                    can_ports.append(can_name)
            available_can_ports = can_ports
        else:
            available_can_ports = []

        self.last_scan_time = time.time()
        return available_ports, available_can_ports

    def difference_new_and_old_ports(
        self,
        new_ports: list[ListPortInfo],
        old_ports: list[ListPortInfo],
        new_can_ports: list[str],
        old_can_ports: list[str],
    ) -> NewAndOldPorts:
        """
        Make a difference between new and old ports.
        """
        # For ports, look at port.device for comparison
        new_ports_set = {port.device for port in new_ports}
        old_ports_set = {port.device for port in old_ports}

        # For CAN ports, look at the port name for comparison
        new_can_ports_set = set(new_can_ports)
        old_can_ports_set = set(old_can_ports)

        # Find the difference between the new and old ports
        new_ports_difference = new_ports_set.difference(old_ports_set)
        old_ports_difference = old_ports_set.difference(new_ports_set)

        # Find the difference between the new and old CAN ports
        new_can_ports_difference = new_can_ports_set.difference(old_can_ports_set)
        old_can_ports_difference = old_can_ports_set.difference(new_can_ports_set)

        return NewAndOldPorts(
            new_ports=[
                port for port in new_ports if port.device in new_ports_difference
            ],
            old_ports=[
                port for port in old_ports if port.device in old_ports_difference
            ],
            new_can_ports=list(new_can_ports_difference),
            old_can_ports=list(old_can_ports_difference),
        )

    async def _find_robots(self) -> None:
        """
        Loop through all available ports and try to connect to a robot.

        Use self.scan_ports() before to update self.available_ports and self.available_can_ports
        """

        sim = get_sim()
        sim.reset()
        self._all_robots = []

        # If we are only simulating, we can just use the SO100Hardware class
        if config.ONLY_SIMULATION:
            logger.debug("ONLY_SIMULATION is set to True. Using SO-100 in simulation.")
            self._all_robots = [SO100Hardware(only_simulation=True)]
            return

        # Keep track of connected devices by port name and serial to avoid duplicates
        connected_devices: Set[str] = set()
        connected_serials: Set[str] = set()

        # Try each serial port exactly once
        for port in self.available_ports:
            serial_num = getattr(port, "serial_number", None)
            # Skip if this port or its serial has already been connected
            if port.device in connected_devices or (
                serial_num and serial_num in connected_serials
            ):
                logger.debug(f"Skipping {port.device}: already connected (or alias).")
                continue

            for robot_class in [
                WX250SHardware,
                KochHardware,
                SO100Hardware,
            ]:
                if not hasattr(robot_class, "name") or not hasattr(
                    robot_class, "from_port"
                ):
                    continue

                logger.debug(
                    f"Trying to connect to {robot_class.name} on {port.device}."
                )
                robot = robot_class.from_port(port)
                if robot is None:
                    logger.debug(
                        f"Failed to create robot from {robot_class.name} on {port.device}."
                    )
                    continue
                logger.debug(f"Robot created: {robot}")
                await robot.connect()

                if robot is not None:
                    logger.success(f"Connected to {robot_class.name} on {port.device}.")
                    self._all_robots.append(robot)
                    # Mark both device and serial as connected
                    connected_devices.add(port.device)
                    if serial_num:
                        connected_serials.add(serial_num)
                    break  # stop trying other classes on this port

        # Detect CAN-based Agilex Piper robots
        if config.ENABLE_CAN:
            for can_name in self.available_can_ports:
                logger.info(f"Attempting to connect to Agilex Piper on {can_name}")
                try:
                    robot = PiperHardware.from_can_port(can_name=can_name)
                    if robot is None:
                        logger.debug(
                            f"Failed to create PiperHardware from {can_name}. Skipping."
                        )
                        continue
                    await robot.connect()
                except Exception as e:
                    logger.warning(
                        f"Error connecting to Agilex Piper on {can_name}: {e}. Skipping."
                    )
                    continue
                if robot is not None:
                    self._all_robots.append(robot)
                    logger.success(f"Connected to Agilex Piper on {can_name}")

        # Add manually added robots
        self._all_robots.extend(self._manually_added_robots)

        if not self._all_robots:
            logger.info("No robot connected.")

    @async_property
    async def robots(self) -> list[BaseRobot]:
        """
        Return all connected robots.
        """

        # In simulation, we call _find_robots() only once. It doesn't need to be updated.
        if config.ONLY_SIMULATION:
            if self._all_robots:
                return self._all_robots
            await self._find_robots()
            return self._all_robots

        # If we are not in simulation, we check the ports
        if time.time() - self.last_scan_time > 1:
            logger.info("Scanning ports.")
            ports, can_ports = self._scan_ports()

            # If new ports are detected or old ports are deleted, we refresh the list of robots
            difference = self.difference_new_and_old_ports(
                ports, self.available_ports, can_ports, self.available_can_ports
            )
            if (
                difference.new_ports
                or difference.old_ports
                or difference.new_can_ports
                or difference.old_can_ports
            ):
                # First, disconnect all robots
                for robot in self._all_robots:
                    robot.disconnect()
                self.available_ports = ports
                self.available_can_ports = can_ports
                await self._find_robots()

        # Return the stored list of robots
        return self._all_robots

    async def get_robot(self, robot_id: int = 0) -> BaseRobot:
        """
        Return the currently connected robot.
        """
        robot = None

        if not isinstance(robot_id, int):
            raise ValueError("robot_id must be an integer.")

        if self._all_robots and len(self._all_robots) > robot_id:
            return self._all_robots[robot_id]

        robots = await self.robots
        if robot_id >= len(robots):
            raise HTTPException(
                status_code=400,
                detail=f"Robot ID {robot_id} is out of range. Only {len(robots)} robots connected.",
            )

        robot = robots[robot_id] if robots else None
        if robot is None:
            raise HTTPException(
                status_code=400, detail=f"No robot with ID {robot_id} connected."
            )

        return robot

    def get_id_from_robot(self, robot: BaseRobot) -> int:
        """
        Return the ID of the robot.
        """

        return self._all_robots.index(robot)

    async def status(self) -> list[RobotConfigStatus]:
        """
        Return the status of all robots. Used at server startup
        """
        robots_status = [robot.status() for robot in await self.robots]
        return [status for status in robots_status if status is not None]

    async def add_connection(
        self, robot_name: str, connection_details: dict[str, Any]
    ) -> tuple[int, BaseRobot]:
        """
        Manually add a connection to a robot using the robot type and connection details.
        Useful when detecting the robot is more complex than just a serial port.
        Eg: IP address, etc.

        :param robot_name: Name of the robot to connect to.
        :param connection_details: Dictionary containing connection details specific to the robot.

        :return: Tuple containing the robot ID and the connected robot instance.
        """
        robot_class = robot_name_to_class.get(robot_name)
        if robot_class is None:
            raise HTTPException(
                status_code=400,
                detail=f"Robot {robot_name} not supported. Supported robots: {list(robot_name_to_class.keys())}",
            )
        robot = robot_class(**connection_details)
        await robot.connect()
        self._all_robots.append(robot)
        self._manually_added_robots.append(robot)
        logger.success(
            f"Connected to {robot.name} with robot_id {len(self._all_robots) - 1}."
        )
        robot_id = len(self._all_robots) - 1
        return robot_id, robot

    async def remove_connection(self, robot_id: int):
        """
        Remove a connection to a robot by its ID.
        """
        if not isinstance(robot_id, int):
            raise ValueError("robot_id must be an integer.")

        if robot_id < 0 or robot_id >= len(self._all_robots):
            raise HTTPException(
                status_code=400,
                detail=f"Robot ID {robot_id} is out of range. Only {len(self._all_robots)} robots connected.",
            )

        robot = await self.get_robot(robot_id=robot_id)
        robot.disconnect()
        self._all_robots.remove(robot)
        if robot in self._manually_added_robots:
            # Remove from manually added robots if it was added manually
            self._manually_added_robots.remove(robot)
        logger.success(f"Disconnected and removed robot with ID {robot_id}.")


@lru_cache()
def get_rcm() -> RobotConnectionManager:
    global rcm

    if rcm is None:
        rcm = RobotConnectionManager()

    return rcm

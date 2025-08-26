import asyncio
import concurrent.futures
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from loguru import logger

from phosphobot.hardware.base import BaseMobileRobot
from phosphobot.models import RobotConfigStatus


@dataclass
class MovementCommand:
    position: np.ndarray
    orientation: Optional[np.ndarray] = None


class UnitreeGo2(BaseMobileRobot):
    name = "unitree-go2"

    UNITREE_MAC_PREFIXES = ["78:22:88"]

    def __init__(self, ip: str, max_history_len: int = 100, **kwargs):
        """
        Initialize the UnitreeGo2 robot.

        Args:
            ip: Local network IP address of the robot. Eg: 192.168.1.42
            **kwargs: Additional keyword arguments
        """
        self.ip = ip
        self.conn = None
        self.current_position = np.zeros(3)  # [x, y, z]
        self.current_orientation = np.zeros(3)  # [roll, pitch, yaw]
        self._is_connected = False
        self._connection_loop = None
        self._connection_thread = None
        self._loop_thread = None
        self._shutdown_event = threading.Event()
        self.is_moving = False

        # Status variables about the robot
        self.lowstate = None
        self.sportmodstate = None
        self.last_movement = 0.0

        # Track movement instructions
        self.movement_queue: Deque[MovementCommand] = deque(maxlen=max_history_len)
        super().__init__(**kwargs)

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        status = (
            self._is_connected
            and self.conn is not None
            and getattr(self.conn, "isConnected", True)
        )
        return status

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """
        Set the connection status of the robot.

        Args:
            value: True if connected, False otherwise
        """
        if value and self.conn is None:
            raise ValueError("Cannot set is_connected to True without a connection")
        self._is_connected = value

    def _create_event_loop_thread(self):
        """Create a dedicated event loop thread for WebRTC operations."""

        def loop_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._connection_loop = loop

            try:
                # Keep the loop running until shutdown
                loop.run_until_complete(self._loop_runner())
            except Exception as e:
                logger.error(f"Event loop error: {e}")
            finally:
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()

                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )

                    loop.close()
                except Exception as e:
                    logger.error(f"Error cleaning up event loop: {e}")

        self._loop_thread = threading.Thread(target=loop_thread, daemon=True)
        self._loop_thread.start()

        # Wait for loop to be created
        max_wait = 5
        start_time = time.time()
        while self._connection_loop is None and (time.time() - start_time) < max_wait:
            time.sleep(0.1)

        if self._connection_loop is None:
            raise RuntimeError("Failed to create event loop")

    async def _loop_runner(self):
        """Keep the event loop alive until shutdown."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)

    def _run_async_safely(self, coro):
        """
        Helper method to run async code safely in the dedicated event loop.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        if self._connection_loop is None:
            self._create_event_loop_thread()

        # Use the dedicated event loop
        future = asyncio.run_coroutine_threadsafe(coro, self._connection_loop)
        try:
            return future.result(timeout=30)  # 30 second timeout
        except concurrent.futures.TimeoutError:
            logger.error("Async operation timed out")
            raise
        except Exception as e:
            logger.error(f"Async operation failed: {e}")
            raise

    async def connect(self):
        """
        Initialize communication with the robot.

        Current understanding: The robot needs to be first switched to AI mode, then
        disconnected, and then reconnected to ensure it operates in the correct mode.
        """
        try:
            # Create connection and connect
            try:
                self.conn = Go2WebRTCConnection(
                    WebRTCConnectionMethod.LocalSTA, ip=self.ip
                )
                if self.conn is None:
                    raise Exception("Failed to create WebRTC connection: conn is None")
                await asyncio.wait_for(self.conn.connect(), timeout=10.0)
            except Exception as e:
                logger.warning(f"Failed to connect using IP {self.ip}: {e}")
                raise Exception(f"Failed to connect to UnitreeGo2 at {self.ip}: {e}")

            # Switch to AI mode
            # await self.conn.datachannel.pub_sub.publish_request_new(
            #     RTC_TOPIC["MOTION_SWITCHER"],
            #     {"api_id": 1002, "parameter": {"name": "ai"}},
            # )
            # await asyncio.sleep(5)

            # # Shutdown the connection
            # self.disconnect()

            # # Connect again
            # self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
            # await asyncio.wait_for(self.conn.connect(), timeout=10.0)
            # await self._ensure_moving_mode()

            def lowstate_callback(message):
                self.lowstate = message["data"]

            # Connect to the lowstate topic to receive battery and sensor data
            self.conn.datachannel.pub_sub.subscribe(
                RTC_TOPIC["LOW_STATE"], lowstate_callback
            )

            def sportmodestatus_callback(message):
                self.sportmodstate = message["data"]

            self.conn.datachannel.pub_sub.subscribe(
                RTC_TOPIC["LF_SPORT_MOD_STATE"], sportmodestatus_callback
            )

            # await self.conn.datachannel.pub_sub.publish_request_new(
            #     RTC_TOPIC["MOTION_SWITCHER"],
            #     {"api_id": 1002, "parameter": {"name": "ai"}},
            # )

            self._is_connected = True

        except Exception as e:
            # Clean up connection on failure
            raise e

    def disconnect(self) -> None:
        """
        Close the connection to the robot.
        """
        try:
            # Set shutdown flag
            self._shutdown_event.set()

            if self.conn:
                try:
                    # Disconnect the WebRTC connection
                    if self._connection_loop and not self._connection_loop.is_closed():
                        future = asyncio.run_coroutine_threadsafe(
                            self.conn.disconnect(), self._connection_loop
                        )
                        future.result(timeout=5)
                except Exception as e:
                    logger.warning(f"Error during WebRTC disconnect: {e}")

                self.conn = None

            # Clean up the event loop thread
            if self._loop_thread and self._loop_thread.is_alive():
                try:
                    self._loop_thread.join(timeout=2)
                except Exception as e:
                    logger.warning(f"Error joining loop thread: {e}")

            self.is_connected = False
            self._connection_loop = None
            self._loop_thread = None
            logger.info("Disconnected from UnitreeGo2")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            # Force cleanup
            self.conn = None
            self.is_connected = False
            self._connection_loop = None

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.

        Returns:
            - state: Last movement command position [x, y, z] and orientation [roll, pitch, yaw]
            - joints_position: Empty array as we're not tracking joint positions
        """
        # Return the last movement command as observation
        # Combine position and orientation into a 6D state vector
        state = np.concatenate([self.current_position, self.current_orientation])
        # Return empty array for joints since we're not tracking them
        joints_position = np.array([])

        return state, joints_position

    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Not implemented
        """
        pass

    def get_info_for_dataset(self):
        """
        Not implemented
        """
        raise NotImplementedError

    async def _ensure_moving_mode(self):
        """
        The go2 has multiple motions modes.
        "mcf" is the mode introduced in the 1.1.7 firmware update that allows the robot to move
        with great stability.
        On older firmware versions, you need to use the "normal" mode to achieve similar results.
        In our testing, the "mcf" mode is the most stable and reliable for movement.
        """
        # # Get the current motion_switcher status
        # response = await self.conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
        # )

        # if response["data"]["header"]["status"]["code"] == 0:
        #     data = json.loads(response["data"]["data"])
        #     current_motion_switcher_mode = data["name"]
        #     logger.debug(f"Current motion mode: {current_motion_switcher_mode}")

        # # Switch to "normal" mode if not already
        # if current_motion_switcher_mode != "mcf":
        #     logger.debug(
        #         f"Switching motion mode from {current_motion_switcher_mode} to 'mcf'..."
        #     )
        #     await self.conn.datachannel.pub_sub.publish_request_new(
        #         RTC_TOPIC["MOTION_SWITCHER"],
        #         {"api_id": 1002, "parameter": {"name": "mcf"}},
        #     )
        #     await asyncio.sleep(5)  # Wait while it stands up

        # await asyncio.sleep(1)  # Allow time for mode switch to take effect

    async def _move_robot(
        self,
        x: float,
        y: float,
        rz: float,
    ):
        """
        Move the robot to the specified position and orientation asynchronously.

        Current understanding: The payload to Unitree expects x, y, z:
        x: forward/backward movement
        y: left/right movement
        rz: rotate right/left
        The values should be between -1 and 1, where 1 is maximum movement in that direction,
        and -1 the maximum movement in the opposite direction.
        """

        current_time = time.perf_counter()
        # Rate limit
        if self.last_movement != 0.0 and (current_time - self.last_movement < 0.1):
            logger.warning(
                f"Skipping move command due to rate limiting: last_movement={self.last_movement}, current_time={current_time}"
            )
            return
        self.last_movement = current_time

        if not self.is_connected or self.conn is None:
            logger.warning(
                f"Robot is not connected: conn={self.conn} is_connected={self.is_connected}"
            )
            return

        # Clamp the values to the expected range
        rz = np.clip(rz, -90, 90)
        x = np.clip(x, -1, 1)
        y = np.clip(y, -1, 1)
        # Convert to float´
        x, y, rz = float(x), float(y), float(rz)

        try:
            self.is_moving = True

            # This is how to use the SPORT_MOD topic to move the robot
            # payload = {"x": x, "y": y, "z": rz}
            # await self.conn.datachannel.pub_sub.publish_request_new(
            #     RTC_TOPIC["SPORT_MOD"],
            #     {
            #         "api_id": SPORT_CMD["Move"],
            #         "parameter": payload,
            #     },
            # )

            # Instead, we use the WIRELESS_CONTROLLER topic to move the robot. This is a bit smoother
            payload = {"lx": y, "ly": x, "rx": -rz, "ry": 0, "keys": 0}
            logger.debug(f"Sending payload for position: {payload}")
            await asyncio.wait_for(
                self.conn.datachannel.pub_sub.publish(
                    RTC_TOPIC["WIRELESS_CONTROLLER"],
                    payload,
                ),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            logger.warning("UnitreeGo2 move command timed out")
        except Exception as e:
            logger.error(f"Error during move: {e}")
            raise e
        finally:
            self.is_moving = False

    async def move_robot_absolute(
        self,
        target_position: np.ndarray,
        target_orientation_rad: Optional[np.ndarray],
        **kwargs,
    ) -> None:
        """
        Move the robot to the target position and orientation asynchronously.

        Args:
            target_position: Target position as [x, y, z]
            target_orientation_rad: Target orientation in radians [roll, pitch, yaw]
            **kwargs: Additional arguments
        """
        self.current_position = target_position.copy()
        self.current_orientation = (
            target_orientation_rad.copy()
            if target_orientation_rad is not None
            else np.zeros(3)
        )
        self.movement_queue.append(
            MovementCommand(
                position=target_position.copy(),
                orientation=target_orientation_rad.copy()
                if target_orientation_rad is not None
                else None,
            )
        )
        await self._move_robot(
            x=target_position[0] * 100,
            y=target_position[1] * 100,
            rz=np.rad2deg(target_orientation_rad[2])
            if target_orientation_rad is not None
            else 0.0,
        )

    async def move_robot_relative(
        self,
        target_position: np.ndarray,
        target_orientation_rad: Optional[np.ndarray],
        **kwargs,
    ) -> None:
        """
        Move the robot to the target position and orientation asynchronously.

        Args:
            target_position: Target position as [x, y, z]
            target_orientation_rad: Target orientation in radians [roll, pitch, yaw]
            **kwargs: Additional arguments
        """
        self.current_position += target_position.copy()
        if target_orientation_rad is not None:
            self.current_orientation += target_orientation_rad.copy()
        else:
            self.current_orientation = np.zeros(3)
        self.movement_queue.append(
            MovementCommand(
                position=target_position.copy(),
                orientation=target_orientation_rad.copy()
                if target_orientation_rad is not None
                else None,
            )
        )
        await self._move_robot(
            x=target_position[0] * 100,
            y=target_position[1] * 100,
            rz=np.rad2deg(target_orientation_rad[2])
            if target_orientation_rad is not None
            else 0.0,
        )

    def status(self) -> RobotConfigStatus:
        """
        Get the status of the robot.

        Returns:
            RobotConfigStatus object
        """
        return RobotConfigStatus(
            name=self.name, device_name=self.ip, robot_type="mobile"
        )

    async def move_to_initial_position(self) -> None:
        """
        Move the robot to its initial position.

        This puts the robot in a stand position ready for operation.
        """
        # Zero-in the initial position and orientation
        self.initial_position = np.zeros(3)
        self.initial_orientation_rad = np.zeros(3)

    def move_to_initial_position_sync(self) -> None:
        """
        Synchronous wrapper for move_to_initial_position.
        Move the robot to its initial position.
        """
        self._run_async_safely(self.move_to_initial_position())

    async def move_to_sleep(self) -> None:
        """
        Move the robot to its sleep position.

        This makes the robot sit down before potentially disconnecting.
        """
        if not self.is_connected or self.conn is None:
            logger.warning("Robot is not connected")
            return

        try:
            # Make the robot sit
            await asyncio.wait_for(
                self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["StandDown"], "parameter": {"data": True}},
                ),
                timeout=10.0,
            )

            # Wait for the robot to sit
            await asyncio.sleep(3)

            logger.info("Robot moved to sleep position")

        except asyncio.TimeoutError:
            logger.error("Move to sleep position timed out")
            raise
        except Exception as e:
            logger.error(f"Error moving to sleep position: {e}")
            raise

    def move_to_sleep_sync(self) -> None:
        """
        Synchronous wrapper for move_to_sleep.
        Move the robot to its sleep position.
        """
        self._run_async_safely(self.move_to_sleep())

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, "_is_connected") and self._is_connected:
                self.disconnect()
        except Exception:
            pass

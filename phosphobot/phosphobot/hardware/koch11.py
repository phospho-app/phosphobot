from typing import Optional

import numpy as np
from dynamixel_sdk import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)
from loguru import logger
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.hardware.base import BaseManipulator
from phosphobot.utils import get_resources_path


class KochHardware(BaseManipulator):
    name: str = "koch-v1.1"

    URDF_FILE_PATH = str(get_resources_path() / "urdf" / "koch" / "robot.urdf")

    AXIS_ORIENTATION = [0, 0, 1, -1]

    TORQUE_ENABLE = 1  # Value to enable torque
    TORQUE_DISABLE = 0  # Value to disable torque

    # Control table addresses for your Dynamixel model (e.g., XM430-W210)
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_POSITION_D_GAIN = 80
    ADDR_POSITION_I_GAIN = 82
    ADDR_POSITION_P_GAIN = 84
    ADDR_PRESENT_CURRENT = 126  # For XM430-W350 (check your motor's manual)
    ADDR_PRESENT_VOLTAGE = 144
    LENGTH_PRESENT_CURRENT = 2  # Current is 2 bytes (16-bit)
    PWM_LIMIT_ADDR = 36  # Replace with the correct address for your motor model

    # Dynamixel settings
    SERVO_IDS = [1, 2, 3, 4, 5, 6]  # Dynamixel IDs
    BAUDRATE = 1000000  # Baud rate
    RESOLUTION = 4096  # 12-bit resolution

    # Measured offset on the servos
    CALIBRATION_POSITION = [
        np.pi / 2,
        -np.pi / 2,
        +np.pi / 2,
        0,
        +np.pi,
        -np.pi / 2,
    ]

    END_EFFECTOR_LINK_INDEX = 4
    GRIPPER_JOINT_INDEX = 5

    GRIPPING_THRESHOLD = 100
    NON_GRIPPING_THRESHOLD = 10

    @classmethod
    def from_port(cls, port: ListPortInfo, **kwargs) -> Optional["KochHardware"]:
        """
        Detect if the device is a Koch v1.1 robot.
        """
        if port.pid == 21971 and port.serial_number in {"58CD176940"}:
            return cls(device_name=port.device, serial_id=port.serial_number)
        return None

    async def connect(self):
        # Initialize PortHandler and PacketHandler
        self.portHandler = PortHandler(self.device_name)
        self.packetHandler = PacketHandler(protocol_version=2.0)

        # Open port
        if not self.portHandler.openPort():
            logger.warning("Failed to open the port")
            raise Exception("Failed to open the port")

        # Set port baud rate
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            logger.warning("Failed to set the baud rate")
            raise Exception("Failed to set the baud rate")

        self.is_connected = True

    def disconnect(self):
        if self.portHandler.is_open:
            self.portHandler.closePort()
        self.is_connected = False

    def _set_pid_gain_motor(
        self, dxl_id: int, p_gain: float, i_gain: float, d_gain: float
    ) -> None:
        """
        Set the PID gains for a motor.
        Note: You have to call this function AFTER enabling torque.
        """
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return

        # Set Position D Gain
        self.packetHandler.write2ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_POSITION_D_GAIN, d_gain
        )
        # Set Position I Gain
        self.packetHandler.write2ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_POSITION_I_GAIN, i_gain
        )
        # Set Position P Gain
        self.packetHandler.write2ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_POSITION_P_GAIN, p_gain
        )

    def _set_pid_gains_group(self) -> None:
        """
        Sets the PID gains for all motors using GroupSyncWrite for efficiency.

        This function should be called AFTER enabling torque.
        """
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return

        # Create a GroupSyncWrite instance for 2-byte values (D, I, and P gains)
        groupSyncWriteD = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_POSITION_D_GAIN, 2
        )
        groupSyncWriteI = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_POSITION_I_GAIN, 2
        )
        groupSyncWriteP = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_POSITION_P_GAIN, 2
        )

        if not self.config:
            raise ValueError(
                "Configuration is not set. Please set the configuration before calling this method."
            )

        # Loop through all servo configurations and add them to the SyncWrite buffer
        pid_params = np.array(
            [(c.d_gain, c.i_gain, c.p_gain) for c in self.config.pid_gains],
            dtype=np.uint16,
        )

        # Convert PID values to bytes and add to GroupSyncWrite
        successD = all(
            [  # type: ignore
                groupSyncWriteD.addParam(servo_id, [DXL_LOBYTE(d), DXL_HIBYTE(d)])  # type: ignore
                for servo_id, (d, _, _) in zip(self.SERVO_IDS, pid_params)
            ]
        )
        successI = all(
            [  # type: ignore
                groupSyncWriteI.addParam(servo_id, [DXL_LOBYTE(i), DXL_HIBYTE(i)])  # type: ignore
                for servo_id, (_, i, _) in zip(self.SERVO_IDS, pid_params)
            ]
        )
        successP = all(
            [  # type: ignore
                groupSyncWriteP.addParam(servo_id, [DXL_LOBYTE(p), DXL_HIBYTE(p)])  # type: ignore
                for servo_id, (_, _, p) in zip(self.SERVO_IDS, pid_params)
            ]
        )

        if not (successP and successI and successD):
            logger.warning("Failed to add one or more motors to sync write buffer")
            return

        # Send all sync writes
        for group, name in zip(
            [groupSyncWriteD, groupSyncWriteI, groupSyncWriteP], ["D", "I", "P"]
        ):
            if group.txPacket() != COMM_SUCCESS:
                logger.warning(f"Sync Write failed for {name}-Gain")
            group.clearParam()

    def enable_torque(self):
        """
        Enable torque for the motors.
        """
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return

        # Create a GroupSyncWrite instance for 1-byte values (Torque Enable)
        groupSyncWrite = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_TORQUE_ENABLE, 1
        )

        # Add all motors to the SyncWrite buffer
        success = all(
            [
                groupSyncWrite.addParam(servo_id, [self.TORQUE_ENABLE])
                for servo_id in self.SERVO_IDS
            ]
        )

        if not success:
            logger.warning("Failed to add one or more motors to sync write buffer")
            return

        # Send the sync write command to enable torque for all motors at once
        if groupSyncWrite.txPacket() != COMM_SUCCESS:
            logger.warning("Sync Write failed for enabling torque")

        # Clear the buffer
        groupSyncWrite.clearParam()

        # Set PID gains for each motor
        if self.config.pid_gains != []:
            self._set_pid_gains_group()

    def disable_torque(self):
        """
        Disable torque for the motors.
        """
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return

        # Create a GroupSyncWrite instance for 1-byte values (Torque Disable)
        groupSyncWrite = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            self.ADDR_TORQUE_ENABLE,
            self.TORQUE_ENABLE,
        )

        # Add all motors to the SyncWrite buffer
        success = all(
            [
                groupSyncWrite.addParam(servo_id, [self.TORQUE_DISABLE])
                for servo_id in self.SERVO_IDS
            ]
        )

        if not success:
            logger.warning("Failed to add one or more motors to sync write buffer")
            return

        # Send the sync write command to disable torque for all motors at once
        if groupSyncWrite.txPacket() != COMM_SUCCESS:
            logger.warning("Sync Write failed for disabling torque")

        # Clear the buffer
        groupSyncWrite.clearParam()

    def write_motor_position(self, servo_id: int, units: int, **kwargs) -> None:
        # Write goal position
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return

        self.packetHandler.write4ByteTxRx(
            self.portHandler,
            servo_id,
            self.ADDR_GOAL_POSITION,
            int(units),
        )

    def write_group_motor_position(
        self, q_target: np.ndarray, enable_gripper: bool
    ) -> None:
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return

        # Filter out the gripper servo if needed
        servo_ids = np.array(self.SERVO_IDS)
        if not enable_gripper:
            mask = servo_ids != self.gripper_servo_id
            servo_ids = servo_ids[mask]
            q_target = q_target[mask]

        # Create a Sync Write group for goal position (4 bytes per motor)
        groupSyncWrite = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, 4
        )

        # Convert goal positions to bytes in vectorized form
        # Ensure uint32 for correct byte conversion
        q_target = q_target.astype(np.uint32)
        param_goal_position = np.column_stack(
            [
                DXL_LOBYTE(DXL_LOWORD(q_target)),
                DXL_HIBYTE(DXL_LOWORD(q_target)),
                DXL_LOBYTE(DXL_HIWORD(q_target)),
                DXL_HIBYTE(DXL_HIWORD(q_target)),
            ]
        ).tolist()

        # Directly add all motors to SyncWrite buffer
        success = all(
            groupSyncWrite.addParam(servo_id, param)
            for servo_id, param in zip(servo_ids, param_goal_position)
        )

        if not success:
            logger.warning("Failed to add one or more motors to sync write buffer")
            return

        # Send the sync write command (sets all motor positions at once)
        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            logger.warning(
                f"Sync Write failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
            )
        # Clear the buffer
        groupSyncWrite.clearParam()

    def read_group_motor_position(self) -> np.ndarray:
        """
        Reads the current positions of all motors using GroupSyncRead.

        Returns:
            np.ndarray: An array of motor positions in Dynamixel units.
        """
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return np.full(len(self.SERVO_IDS), np.nan)

        # Create a Sync Read group for present position (4 bytes per motor)
        groupSyncRead = GroupSyncRead(
            self.portHandler, self.packetHandler, self.ADDR_PRESENT_POSITION, 4
        )

        # Attempt to add all motors to SyncRead buffer in a single step
        if not all(groupSyncRead.addParam(servo_id) for servo_id in self.SERVO_IDS):
            logger.warning("Failed to add one or more motors to sync read buffer")
            return np.full(len(self.SERVO_IDS), np.nan)  # Return NaN array on failure

        # Send the sync read command
        if groupSyncRead.txRxPacket() != COMM_SUCCESS:
            logger.warning(
                f"Sync Read failed: {self.packetHandler.getTxRxResult(groupSyncRead.txRxPacket())}"
            )
            return np.full(len(self.SERVO_IDS), np.nan)  # Return NaN array on failure

        # Extract motor positions in a vectorized way
        positions = np.array(
            [
                (
                    groupSyncRead.getData(servo_id, self.ADDR_PRESENT_POSITION, 4)
                    if groupSyncRead.isAvailable(
                        servo_id, self.ADDR_PRESENT_POSITION, 4
                    )
                    else np.nan
                )
                for servo_id in self.SERVO_IDS
            ],
            dtype=np.float64,
        )

        # Clear the buffer
        groupSyncRead.clearParam()

        return positions

    def read_motor_position(self, servo_id: int, **kwargs) -> Optional[int]:
        """
        Read the position of a Dynamixel servo.
        """
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return None

        try:
            (
                position,
                dxl_comm_result,
                dxl_error,
            ) = self.packetHandler.read4ByteTxRx(
                self.portHandler, servo_id, self.ADDR_PRESENT_POSITION
            )
            if dxl_comm_result != COMM_SUCCESS:
                logger.warning(
                    f"Communication Error for motor {servo_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
                )
            elif dxl_error != 0:
                logger.warning(
                    f"Hardware Error for motor {servo_id}: {self.packetHandler.getRxPacketError(dxl_error)}"
                )
            else:
                return position
        except Exception as e:
            logger.error(f"Error reading present position for motor {servo_id}: {e}")

        return None

    def read_motor_torque(self, servo_id: int, **kwargs) -> Optional[float]:
        """
        Read the torque of a Dynamixel servo.
        """
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return None

        try:
            (
                current_torque,
                dxl_comm_result,
                dxl_error,
            ) = self.packetHandler.read2ByteTxRx(
                self.portHandler, servo_id, self.ADDR_PRESENT_CURRENT
            )  # Current reading is 2 bytes
            if dxl_comm_result != COMM_SUCCESS:
                logger.warning(
                    f"Communication Error for motor {servo_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
                )
                return None
            elif dxl_error != 0:
                logger.warning(
                    f"Hardware Error for motor {servo_id}: {self.packetHandler.getRxPacketError(dxl_error)}"
                )
                return None
            else:
                return current_torque * 2.69e-3  # conversion value from dynamixel
        except Exception as e:
            logger.error(f"Error reading present position for motor {servo_id}: {e}")
            return None

    def read_motor_voltage(self, servo_id: int, **kwargs) -> None:
        if not self.is_connected:
            logger.warning("KochHardware: Not connected. Run .connect() first.")
            return

        # Read voltage value (2-byte unsigned integer)
        voltage_raw, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
            self.portHandler, servo_id, self.ADDR_PRESENT_VOLTAGE
        )

        if dxl_comm_result != COMM_SUCCESS:
            logger.warning(
                f"Failed to read voltage from motor {servo_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
            )
            return None

        if dxl_error:
            logger.warning(
                f"Error reading voltage from motor {servo_id}: {self.packetHandler.getRxPacketError(dxl_error)}"
            )
            return None

        # Convert raw value to voltage (Dynamixel units are in 0.1V steps)
        voltage = voltage_raw / 10.0  # Convert from 0.1V units to Volts

        return voltage

    def calibrate_motors(self, **kwargs) -> None:
        """
        This is called during the calibration phase of the robot.
        It sets the offset of all motors to self.RESOLUTION/2.
        """
        # TODO
        return None

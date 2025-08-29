from typing import Any, Optional

import numpy as np
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.hardware.koch11 import KochHardware
from phosphobot.utils import get_resources_path


class WX250SHardware(KochHardware):
    """
    Hardware interface for an Interbotix WX-250 arm powered by XM430 and XL430 Dynamixel servos.
    """

    # Robot identity
    name: str = "wx-250s"

    # URDF and Configuration
    URDF_FILE_PATH = str(
        get_resources_path() / "urdf" / "wx-250s" / "urdf" / "wx250s.urdf"
    )

    # Axis orientation
    AXIS_ORIENTATION = [0, 0, 0, 1]  # Adjust if needed

    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    BAUDRATE = 3000000
    RESOLUTION = 4096  # XM430 resolution
    OPERATING_MODE_ADDR = 11
    POSITION_CONTROL_MODE = 3
    PWM_LIMIT_VALUE = 300

    # Dynamixel settings
    motors = {
        # name: (index, model)
        "waist": [1, "XM430-W350"],
        "shoulder": [2, "M430-W350"],
        "elbow": [4, "M430-W350"],
        "forearm_roll": [6, "M430-W350"],
        "wrist_angle": [7, "M430-W350"],
        "wrist_rotate": [8, "sts3215"],
        "gripper": [9, "sts3215"],
    }

    SERVO_IDS = [1, 2, 4, 6, 7, 8, 9]

    # Calibration Offsets (to be updated based on real-world calibration)
    CALIBRATION_POSITION = [
        np.pi / 2,
        +np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        +np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
    ]

    SLEEP_POSITION = [
        0.0889926124093812,
        -1.8059362897558908,
        1.5650424940960141,
        -0.07978648009116934,
        -1.5779852248187624,
        -0.046030661591059244,
        1.0126745550033034,
    ]

    END_EFFECTOR_LINK_INDEX = 11  # This is the beginning of the fingers

    # If your code checks torque readouts to see if something is grasped,
    # you can set these thresholds. They might differ from the Koch’s defaults.
    GRIPPING_THRESHOLD = 100
    NON_GRIPPING_THRESHOLD = 10

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.set_gripper_in_position_control()

    def set_gripper_in_position_control(self) -> None:
        """
        This functions ensures the gripper motor is in position control mode since factory mode is PWM.
        """

        # First limit the PWM speed
        self.packetHandler.write2ByteTxRx(
            self.portHandler,
            self.gripper_servo_id,
            self.PWM_LIMIT_ADDR,
            self.PWM_LIMIT_VALUE,
        )

        # Then ensure it is in position control mdoe :
        self.packetHandler.write2ByteTxRx(
            self.portHandler,
            self.gripper_servo_id,
            self.OPERATING_MODE_ADDR,
            self.POSITION_CONTROL_MODE,
        )

    @classmethod
    def from_port(cls, port: ListPortInfo, **kwargs: Any) -> Optional["WX250SHardware"]:
        """
        Detect if the device is a WX-250s
        """
        # Serialid: FT94W6U7
        if port.pid == 24596:
            return cls(device_name=port.device, serial_id=port.serial_number)
        return None

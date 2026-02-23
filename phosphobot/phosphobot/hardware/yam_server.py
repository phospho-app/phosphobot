#!/usr/bin/env python3
"""
Server script for YAM robot arms using i2rt and portal.

This script starts server processes for YAM arms (follower and/or leader).
The phosphobot dashboard connects to these servers via the YamHardware driver.

Requirements:
    pip install portal-client i2rt

Example usage (single arm):
    python -m phosphobot.hardware.yam_server \
        --can_channel can0 --gripper v3 --mode follower --server_port 1234

Example usage (bimanual - all 4 arms):
    python -m phosphobot.hardware.yam_server \
        --mode bimanual \
        --right_follower_can can_follower_r \
        --left_follower_can can_follower_l \
        --right_leader_can can_leader_r \
        --left_leader_can can_leader_l

Ports (defaults):
    Right follower: 1234
    Left follower:  1235
    Right leader:   5001
    Left leader:    5002
"""

import sys
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro


def _check_dependencies() -> None:
    """Check that required dependencies are installed."""
    missing = []
    try:
        import portal  # noqa: F401
    except ImportError:
        missing.append("portal-client")
    try:
        import i2rt  # noqa: F401
    except ImportError:
        missing.append("i2rt")

    if missing:
        print(f"Error: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


class ServerRobot:
    """A portal RPC server wrapping a robot arm."""

    def __init__(self, robot, port: int, name: str):
        import portal

        self._robot = robot
        self._name = name
        self._port = port
        self._server = portal.Server(port)
        self._running = False

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("get_observations", self._robot.get_observations)

        # Optional methods
        if hasattr(self._robot, "command_joint_state"):
            self._server.bind("command_joint_state", self._robot.command_joint_state)
        if hasattr(self._robot, "update_kp_kd"):
            self._server.bind("update_kp_kd", self._robot.update_kp_kd)

    def serve(self) -> None:
        """Start serving (blocking)."""
        print(f"[{self._name}] Serving on port {self._port}")
        self._running = True
        self._server.start()

    def serve_background(self) -> threading.Thread:
        """Start serving in a background thread."""

        def _run():
            try:
                self._running = True
                self._server.start()
            except Exception as e:
                print(f"[{self._name}] Server error: {e}")
            finally:
                self._running = False

        thread = threading.Thread(target=_run, daemon=True, name=self._name)
        thread.start()
        return thread

    @property
    def is_running(self) -> bool:
        return self._running


class YAMLeaderRobot:
    """Wrapper for YAM leader arm with teaching handle (encoder) support."""

    def __init__(self, robot, name: str = "leader"):
        self._robot = robot
        self._motor_chain = robot.motor_chain
        self._name = name

    def num_dofs(self) -> int:
        return 7  # 6 joints + 1 gripper

    def get_joint_pos(self) -> np.ndarray:
        qpos = self._robot.get_observations()["joint_pos"]
        try:
            encoder_obs = self._motor_chain.get_same_bus_device_states()
            time.sleep(0.01)
            gripper_pos = 1 - encoder_obs[0].position
        except Exception:
            gripper_pos = 1.0
        return np.concatenate([qpos, [gripper_pos]])

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        if len(joint_pos) > self._robot.num_dofs():
            joint_pos = joint_pos[: self._robot.num_dofs()]
        self._robot.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: dict) -> None:
        self._robot.command_joint_state(joint_state)

    def get_observations(self) -> dict:
        qpos = self._robot.get_observations()["joint_pos"]
        try:
            encoder_obs = self._motor_chain.get_same_bus_device_states()
            time.sleep(0.01)
            gripper_pos = 1 - encoder_obs[0].position
            io_inputs = encoder_obs[0].io_inputs
        except Exception:
            gripper_pos = 1.0
            io_inputs = np.array([0.0])
        return {
            "joint_pos": qpos,
            "gripper_pos": np.array([gripper_pos]),
            "io_inputs": io_inputs,
        }

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


@dataclass
class SingleArmArgs:
    """Arguments for single arm server mode."""

    gripper: Literal[
        "crank_4310",
        "linear_3507",
        "linear_4310",
        "yam_teaching_handle",
        "no_gripper",
        "v3",
    ] = "v3"
    """Type of gripper attached to the arm."""

    mode: Literal["follower", "leader"] = "follower"
    """Operating mode: follower (execution) or leader (teaching)."""

    server_port: int = 1234
    """Port number for the server."""

    can_channel: str = "can0"
    """CAN interface name (e.g., can0, can_follower_r)."""


@dataclass
class BimanualArgs:
    """Arguments for bimanual server mode (all 4 arms)."""

    mode: Literal["full", "follower_only"] = "full"
    """'full' = followers + leaders, 'follower_only' = just followers."""

    right_follower_can: str = "can_follower_r"
    left_follower_can: str = "can_follower_l"
    right_follower_port: int = 1234
    left_follower_port: int = 1235

    follower_gripper: Literal[
        "crank_4310", "linear_3507", "linear_4310", "no_gripper", "v3"
    ] = "v3"

    right_leader_can: str = "can_leader_r"
    left_leader_can: str = "can_leader_l"
    right_leader_port: int = 5001
    left_leader_port: int = 5002

    leader_gripper: Literal[
        "crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "v3"
    ] = "yam_teaching_handle"


@dataclass
class Args:
    """YAM arm server launcher."""

    mode: Literal["single", "bimanual"] = "single"
    """Launch mode: 'single' for one arm, 'bimanual' for all 4 arms."""

    single: SingleArmArgs = SingleArmArgs()
    bimanual: BimanualArgs = BimanualArgs()


def run_single_arm(args: SingleArmArgs) -> None:
    """Run a single arm server."""
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_name = "linear_4310" if args.gripper == "v3" else args.gripper
    gripper_type = GripperType.from_string_name(gripper_name)
    robot = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type)

    if args.mode == "leader":
        robot = YAMLeaderRobot(robot, name="leader")

    server = ServerRobot(robot, args.server_port, f"{args.mode}_{args.can_channel}")
    print(f"Starting {args.mode} server on port {args.server_port} (CAN: {args.can_channel})")
    print("Press Ctrl+C to stop.")
    server.serve()


def run_bimanual(args: BimanualArgs) -> None:
    """Run all arm servers for a bimanual setup."""
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    follower_gripper_name = "linear_4310" if args.follower_gripper == "v3" else args.follower_gripper
    leader_gripper_name = "linear_4310" if args.leader_gripper == "v3" else args.leader_gripper

    follower_gripper_type = GripperType.from_string_name(follower_gripper_name)
    leader_gripper_type = GripperType.from_string_name(leader_gripper_name)

    servers = []
    threads = []

    try:
        # Follower arms
        print("Connecting to follower arms...")
        right_follower = get_yam_robot(
            channel=args.right_follower_can, gripper_type=follower_gripper_type
        )
        servers.append(
            ServerRobot(right_follower, args.right_follower_port, "right_follower")
        )

        left_follower = get_yam_robot(
            channel=args.left_follower_can, gripper_type=follower_gripper_type
        )
        servers.append(
            ServerRobot(left_follower, args.left_follower_port, "left_follower")
        )

        # Leader arms
        if args.mode == "full":
            print("Connecting to leader arms...")
            right_leader = get_yam_robot(
                channel=args.right_leader_can, gripper_type=leader_gripper_type
            )
            right_leader_wrapped = YAMLeaderRobot(right_leader, "right_leader")
            servers.append(
                ServerRobot(right_leader_wrapped, args.right_leader_port, "right_leader")
            )

            left_leader = get_yam_robot(
                channel=args.left_leader_can, gripper_type=leader_gripper_type
            )
            left_leader_wrapped = YAMLeaderRobot(left_leader, "left_leader")
            servers.append(
                ServerRobot(left_leader_wrapped, args.left_leader_port, "left_leader")
            )

        # Start all servers
        for server in servers:
            thread = server.serve_background()
            threads.append(thread)
            time.sleep(0.2)

        time.sleep(1.0)
        if not all(s.is_running for s in servers):
            print("Error: Some servers failed to start")
            sys.exit(1)

        print("\nBimanual YAM server running:")
        print(f"  Right follower: port {args.right_follower_port} (CAN: {args.right_follower_can})")
        print(f"  Left follower:  port {args.left_follower_port} (CAN: {args.left_follower_can})")
        if args.mode == "full":
            print(f"  Right leader:   port {args.right_leader_port} (CAN: {args.right_leader_can})")
            print(f"  Left leader:    port {args.left_leader_port} (CAN: {args.left_leader_can})")
        print("\nPress Ctrl+C to stop.")

        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        print("\nShutting down...")


def main() -> None:
    _check_dependencies()
    args = tyro.cli(Args)

    if args.mode == "single":
        run_single_arm(args.single)
    elif args.mode == "bimanual":
        run_bimanual(args.bimanual)


if __name__ == "__main__":
    main()

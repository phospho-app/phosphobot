import asyncio

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from phosphobot.configs import config
from phosphobot.utils import get_local_ip
from phosphobot.models import ChatRequest, ChatResponse


class PhosphobotClient:
    def __init__(self) -> None:
        self.server_url = f"http://{get_local_ip()}:{config.PORT}"
        self.client = httpx.AsyncClient(base_url=self.server_url, timeout=5.0)

    async def status(self) -> Dict[str, str]:
        """
        Get the status of the robot.
        """
        response = await self.client.get("/status", timeout=10.0)
        response.raise_for_status()
        return response.json()

    async def move_joints(self, joints: List[float]) -> None:
        """
        Move the robot joints to the specified angles.
        """
        response = await self.client.post("/joints/write", json={"joints": joints})
        response.raise_for_status()

    async def move_relative(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        open: Optional[float] = None,
    ) -> None:
        response = await self.client.post(
            "/move/relative",
            json={
                "x": x,
                "y": y,
                "z": z,
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "open": open,
            },
        )
        response.raise_for_status()

    async def get_camera_image(
        self,
        camera_ids: Optional[List[int]] = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Dict[int, str]:
        """
        Get an image from the specified camera.
        """
        params = {}
        if resize is not None:
            params["resize_x"] = resize[0]
            params["resize_y"] = resize[1]

        response = await self.client.get(
            "/frames",
            params=params,
            timeout=3.0,
        )
        response.raise_for_status()
        reponse_json = response.json()

        output: Dict[int, str] = {}
        for camera_id in camera_ids or reponse_json.keys():
            if not isinstance(camera_id, int):
                try:
                    camera_id = int(camera_id)
                except ValueError:
                    logger.error(
                        f"Invalid camera ID: {camera_id}. Must be an integer. Ignoring."
                    )
                    continue

            if str(camera_id) in reponse_json:
                image_b64 = reponse_json[str(camera_id)]
                output[camera_id] = image_b64
            else:
                logger.warning(f"Camera {camera_id} not found in response.")

        return output

    async def move_init(self) -> None:
        """
        Initialize the robot's position.
        """
        response = await self.client.post("/move/init")
        response.raise_for_status()

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """
        Send a chat request to the AI model.

        :param prompt: The text prompt to send.
        :param images: List of base64 encoded images to include in the request.
        """

        response = await self.client.post(
            "/ai-control/chat",
            json=chat_request.model_dump(mode="json"),
        )
        response.raise_for_status()
        return ChatResponse.model_validate(response.json())


class RoboticAgent:
    def __init__(
        self,
        images_sizes: Optional[Tuple[int, int]] = (256, 256),
        task_description: str = "Pick up white foam",
        manual_control: bool = False,
    ):
        self.resize = images_sizes
        self.phosphobot_client = PhosphobotClient()
        self.task_description = task_description
        self.manual_control = manual_control
        self.manual_command: Optional[str] = None

    def set_manual_command(self, command: str) -> None:
        """
        Set a manual command for the robot.
        """
        self.manual_command = command

    def get_manual_command(self) -> Optional[str]:
        """
        Get the current manual command and clear it.
        """
        command = self.manual_command
        self.manual_command = None
        return command

    def toggle_control_mode(self) -> str:
        """
        Toggle between AI and manual control modes.
        Returns the new mode.
        """
        self.manual_control = not self.manual_control
        mode = "manual" if self.manual_control else "AI"
        logger.info(f"Switched to {mode} control mode")
        return mode

    async def execute_command(self, chat_response: Optional[ChatResponse]) -> None:
        """
        Execute the AI command by moving the robot.
        """
        if chat_response is None:
            logger.warning("No chat response received. Skipping execution.")
            return None

        if chat_response.endpoint == "move_relative":
            if not chat_response.endpoint_params:
                logger.warning("No parameters provided for move_relative command.")
                return None
            await self.phosphobot_client.move_relative(**chat_response.endpoint_params)
        else:
            logger.warning(
                f"Unsupported command received: {chat_response.endpoint}. Skipping execution."
            )
            return None

        return None

    async def execute_manual_command(self, command: str) -> Optional[Dict[str, float]]:
        """
        Execute a manual command by moving the robot.
        """
        command_map = {
            "move_left": {"rz": 10.0},
            "move_right": {"rz": -10.0},
            "move_forward": {"x": 5.0},
            "move_backward": {"x": -5.0},
            "move_up": {"z": 5.0},
            "move_down": {"z": -5.0},
            "move_gripper_up": {"rx": 10.0},
            "move_gripper_down": {"rx": -10.0},
            "close_gripper": {"open": 0.0},
            "open_gripper": {"open": 1.0},
        }
        next_robot_move = command_map.get(command)
        if next_robot_move is None:
            logger.warning(
                f"Invalid manual command received: {command}. Skipping execution."
            )
            return None
        # Call the phosphobot client to move the robot
        await self.phosphobot_client.move_relative(**next_robot_move)
        return next_robot_move

    async def run(self) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """
        An async generator that yields events for the UI to handle.
        Events are tuples of (event_type: str, payload: dict).
        """
        yield "start_step", {"desc": "Checking robot status."}
        self.robot_status = await self.phosphobot_client.status()
        yield "step_output", {"desc": f"Robot status: {self.robot_status}"}
        yield "step_done", {"success": True}

        # Manual control setup
        if self.manual_control:
            yield "start_step", {"desc": "Manual control mode enabled."}
            yield (
                "step_output",
                {
                    "desc": "Manual control active. Use set_manual_command() to control the robot."
                },
            )
            yield "step_done", {"success": True}

        step_count = 0
        max_steps = 50

        while step_count < max_steps:
            current_mode = "manual" if self.manual_control else "AI"

            if self.manual_control:
                # MANUAL MODE: Wait for manual commands without consuming steps
                manual_command = self.get_manual_command()
                if manual_command:
                    step_count += 1
                    yield (
                        "log",
                        {
                            "text": f"Step {step_count} of {max_steps} - Mode: {current_mode}"
                        },
                    )
                    yield "step_output", {"output": f"Manual command: {manual_command}"}
                    execution_result = await self.execute_manual_command(
                        command=manual_command
                    )
                    yield (
                        "step_output",
                        {"output": f"Execution result: {execution_result}"},
                    )
                else:
                    # Wait for user input without consuming a step
                    await asyncio.sleep(0.1)
                    continue  # Don't increment step_count, just wait
            else:
                # AI MODE: Only run AI processing
                step_count += 1
                yield (
                    "log",
                    {
                        "text": f"Step {step_count} of {max_steps} - Mode: {current_mode}"
                    },
                )
                # Run the agent
                images = await self.phosphobot_client.get_camera_image(
                    resize=self.resize
                )
                if not images:
                    yield (
                        "step_output",
                        {"output": "No images received from cameras. Skipping step."},
                    )
                    continue  # Skip this step if no images

                chat_response = await self.phosphobot_client.chat(
                    chat_request=ChatRequest(
                        prompt=self.task_description,
                        # Convert dict to list of base64 strings
                        images=list(images.values()),
                    )
                )
                yield (
                    "step_output",
                    {"output": f"AI command: {chat_response.model_dump()}"},
                )
                # Execute the command
                await self.execute_command(chat_response=chat_response)

        yield "step_done", {"success": True}
        control_mode = "manual" if self.manual_control else "AI"
        yield "log", {"text": f"Robotic agent run completed in {control_mode} mode."}

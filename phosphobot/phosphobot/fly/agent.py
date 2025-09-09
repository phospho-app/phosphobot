import asyncio
import base64
import os
from typing import Dict, List, Literal, Optional, Tuple

import cv2
from google import genai
from google.genai.errors import ClientError, ServerError
from loguru import logger
import httpx
import numpy as np
from pydantic import BaseModel


class PhosphobotClient:
    def __init__(self, server_url: str = "http://localhost:80"):
        self.server_url = server_url
        self.client = httpx.AsyncClient(base_url=self.server_url)

    async def status(self) -> Dict[str, str]:
        """
        Get the status of the robot.
        """
        response = await self.client.get("/status")
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
        gripper: Optional[float] = None,
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
                "open": gripper,
            },
        )
        response.raise_for_status()

    async def get_camera_image(
        self, camera_ids: Optional[List[int]] = None
    ) -> Dict[int, str]:
        """
        Get an image from the specified camera.
        """

        response = await self.client.get("/frames")
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


class GeminiAgentResponse(BaseModel):
    # Alternative solution: https://ai.google.dev/gemini-api/docs/function-calling?example=meetings
    # A tool call could be more powerful to let the agent pick the amplitude of movement
    # and maybe more directions at once.

    next_robot_move: Literal[
        "rotate_left",
        "rotate_right",
        "move_forward",
        "move_backward",
        "move_up",
        "move_down",
        "move_gripper_up",
        "move_gripper_down",
        "close_gripper",
        "open_gripper",
        "nothing",
    ]


class GeminiAgent:
    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        task_description: str = "Pick up white foam",
        thinking_budget: int = 0,
    ):
        """
        Robot-controlling agent using Gemini VLM.
        """
        from dotenv import load_dotenv

        load_dotenv()

        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.genai_client = genai.Client()
        self.thinking_budget = thinking_budget
        self.model_id = model_id
        self.phosphobot_client = PhosphobotClient()
        self.task_description = task_description

    @property
    def prompt(self) -> str:
        prompt = f"""You control a robot with an ego-centric camera view. You must guide the robot to complete a task using \
step by step instructions to complete the task. The task is: {self.task_description}
The robot can move in 3D space and has a gripper that you can fully open or close. 
"""

        # if len(self.previous_commands) > 0:
        #     prompt += (
        #         "\n Your previous commands: " + "\n".join(self.previous_commands) + "\n"
        #     )
        return prompt

    @property
    def config(self) -> genai.types.GenerateContentConfig:
        """
        Get the configuration for the model.
        """
        return genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            ),
            response_mime_type="application/json",
            response_schema=GeminiAgentResponse,
        )

    async def run(
        self, images: List[genai.types.Part]
    ) -> Tuple[Optional[GeminiAgentResponse], Optional[str]]:
        """
        Run the agent for 1 step.
        """

        # Build the content list with prompt and images
        contents = [self.prompt]
        contents.extend(images)

        # Generate response with retry logic for ServerError and ClientError
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = await self.genai_client.aio.models.generate_content(
                    model=self.model_id, contents=contents, config=self.config
                )
                break  # Success, exit retry loop
            except ServerError as e:
                if "503" in str(e) and "overloaded" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Model overloaded (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds..."
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Model overloaded after {max_retries} attempts. Giving up."
                        )
                        raise
                else:
                    # Re-raise if it's a different ServerError
                    raise
            except ClientError as e:
                error_dict = e.args[1] if len(e.args) > 1 else {}

                # Handle 429 RESOURCE_EXHAUSTED errors
                if e.code == 429:
                    # Extract retry delay from error details if available
                    custom_retry_delay = retry_delay
                    if "error" in error_dict and "details" in error_dict["error"]:
                        for detail in error_dict["error"]["details"]:
                            if (
                                detail.get("@type")
                                == "type.googleapis.com/google.rpc.RetryInfo"
                            ):
                                retry_delay_str = detail.get("retryDelay", "5s")
                                # Parse delay string (e.g., "34s" -> 34)
                                custom_retry_delay = int(retry_delay_str.rstrip("s"))
                                break

                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {custom_retry_delay} seconds..."
                        )
                        await asyncio.sleep(custom_retry_delay)
                    else:
                        logger.error(
                            f"Rate limit exceeded after {max_retries} attempts. Giving up."
                        )
                        raise
                else:
                    # Re-raise if it's a different ClientError
                    raise

        # # Add to chat history (Gemini format)
        # self.chat_history.append(
        #     {
        #         "role": "user",
        #         "parts": contents[:-1],  # Exclude the command instruction
        #     }
        # )
        # self.chat_history.append({"role": "model", "parts": [response.text]})
        raw = response.text
        command: Optional[GeminiAgentResponse] = response.parsed

        return command, raw


class RoboticAgent:
    """
    This robotic agent
    """

    def __init__(
        self,
        images_sizes: Optional[Tuple[int, int]] = (256, 256),
        task_description: str = "Pick up white foam",
    ):
        self.images_sizes = images_sizes
        self.phosphobot_client = PhosphobotClient()
        self.task_description = task_description

    async def get_images(self) -> List[genai.types.Part]:
        """
        Get images from the cameras.
        """

        frames = await self.phosphobot_client.get_camera_image()
        resized_frames = []
        for camera_id, frame in frames.items():
            # If the images_sizes is not None, decode the base64 image and resize it
            if self.images_sizes:
                # Use cv2 to decode the base64 image
                image_data = base64.b64decode(frame)
                # image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                image = cv2.imdecode(
                    np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
                )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if image is not None:
                    image = cv2.resize(
                        image, self.images_sizes, interpolation=cv2.INTER_AREA
                    )
                    # Convert back to bytes
                    _, resized_data = cv2.imencode(".jpg", image)
                    frame = genai.types.Part.from_bytes(
                        data=resized_data.tobytes(), mime_type="image/jpeg"
                    )
                else:
                    logger.error(f"Failed to decode image for camera {camera_id}.")
                    continue
            else:
                # If images_sizes is None, just convert the base64 string to a Part
                frame = genai.types.Part.from_text(text=frame)

            resized_frames.append(frame)

        return resized_frames

    async def execute_command(
        self, command: Optional[GeminiAgentResponse]
    ) -> Optional[Dict[str, float]]:
        """
        Execute the command by moving the robot.
        """
        if command is None:
            logger.warning("No command received. Skipping execution.")
            return None
        if command == "nothing":
            logger.info("Received 'nothing' command. Skipping execution.")
            return None

        # Use a mapping to convert command strings to function calls
        command_map = {
            "rotate_left": {"rz": 10},
            "rotate_right": {"rz": -10},
            "move_forward": {"x": 0.1},
            "move_backward": {"x": -5},
            "move_up": {"z": 5},
            "move_down": {"z": -5},
            "move_gripper_up": {"rx": 10},
            "move_gripper_down": {"rx": -10},
            "close_gripper": {"open": 1},
            "open_gripper": {"open": 0},
        }
        next_robot_move = command_map.get(command.next_robot_move)
        if next_robot_move is None:
            logger.warning(
                f"Invalid command received: {command.next_robot_move}. Skipping execution."
            )
            return
        # Call the phosphobot client to move the robot
        await self.phosphobot_client.move_relative(**next_robot_move)
        return next_robot_move

    async def run(self):
        """
        An async generator that yields events for the UI to handle.
        Events are tuples of (event_type: str, payload: dict).
        """
        yield "start_step", {"desc": "Checking robot status."}
        self.robot_status = await self.phosphobot_client.status()
        yield "step_output", {"desc": f"Robot status: {self.robot_status}"}
        yield "step_done", {"success": True}

        yield "start_step", {"desc": "Initializing the AI agent."}
        # TODO: Here, you'd need to interpret the task description to make sure the agent understands what to do.
        try:
            self.gemini_agent = GeminiAgent(task_description=self.task_description)
        except Exception as e:
            yield "step_error", {"error": f"Failed to initialize agent: {str(e)}"}
            return

        for i in range(10):
            yield "log", {"text": f"Step {i + 1} of 10."}
            # Get images
            images = await self.get_images()
            if not images:
                yield (
                    "step_error",
                    {"error": "No images received from cameras. Skipping step."},
                )
                continue
            # Run the Gemini agent
            next_command, raw = await self.gemini_agent.run(images=images)
            yield (
                "step_output",
                {"output": f"Next command: {next_command} (raw: {raw})"},
            )
            # Execute the command
            execution_result = await self.execute_command(next_command)
            yield (
                "step_output",
                {"output": f"Execution result: {execution_result}"},
            )

        yield "step_done", {"success": True}
        yield "log", {"text": "Robotic agent run completed."}

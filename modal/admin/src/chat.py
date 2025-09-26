import asyncio
from typing import Dict, List, Literal, Optional, Union

from google import genai
from google.genai.errors import ClientError, ServerError
from loguru import logger
from pydantic import BaseModel

from phosphobot.models import ChatRequest, ChatResponse


class GeminiAgentResponse(BaseModel):
    # Alternative solution: https://ai.google.dev/gemini-api/docs/function-calling?example=meetings
    # A tool call could be more powerful to let the agent pick the amplitude of movement
    # and maybe more directions at once.

    next_robot_move: Literal[
        "move_left",
        "move_right",
        "move_forward",
        "move_backward",
        "move_up",
        "move_down",
        "move_gripper_up",
        "move_gripper_down",
        "close_gripper",
        "open_gripper",
        # "nothing",
    ]


class GeminiAgent:
    def __init__(
        self,
        model_id: str = "models/gemini-robotics-er-1.5-preview",  # "gemini-2.5-flash",
        thinking_budget: int = 0,
    ):
        """
        Robot-controlling agent using Gemini VLM.
        """

        self.genai_client = genai.Client()
        self.thinking_budget = thinking_budget
        self.model_id = model_id

    def get_prompt(self, command_history: Optional[List[str]]) -> str:
        prompt = f"""You control a green robot arm from an ego-centric 3D point of view. You must guide the robot arm using \
step by step commands to complete the task: "{self.task_description}"
You control the robot arm in 3D space by moving the end effector. The end effector of the robot arm has a gripper that you can open and close. \
The end effector is your main tool to interact with the environment.
In the chat, the image provided is from a camera recording of the current position of the robot arm and its surroundings. \
Use the image to localize the end effector, understand the task, and give the command for the next step.
"""

        if command_history and len(command_history) > 0:
            prompt += (
                "\n Your previous commands were: " + "\n".join(command_history) + "\n"
            )
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

    async def run(self, chat_request: ChatRequest) -> ChatResponse:
        """
        Run the agent.
        """

        self.task_description = chat_request.prompt
        images = chat_request.images
        if not images:
            images = []

        # Build the content list with prompt and images
        contents: List[Union[genai.types.Part, str]] = [
            self.get_prompt(command_history=chat_request.command_history)
        ]
        # Images are base64 encoded strings
        contents.extend([genai.types.Part.from_text(text=image) for image in images])

        # Generate response with retry logic for ServerError and ClientError
        max_retries = 3
        retry_delay = 5
        response = None

        for attempt in range(max_retries):
            try:
                response = await self.genai_client.aio.models.generate_content(
                    model=self.model_id,
                    contents=contents,  # type: ignore
                    config=self.config,
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

        if response is None:
            logger.error("Failed to get response after all retry attempts.")
            return ChatResponse(command=None, endpoint=None, endpoint_params=None)

        # # Add to chat history (Gemini format)
        # self.chat_history.append(
        #     {
        #         "role": "user",
        #         "parts": contents[:-1],  # Exclude the command instruction
        #     }
        # )
        # self.chat_history.append({"role": "model", "parts": [response.text]})
        raw = response.text
        command: Optional[GeminiAgentResponse] = response.parsed  # type: ignore

        return ChatResponse(
            endpoint="move_relative",
            endpoint_params=self._get_movement_parameters(command),
            command=command.next_robot_move if command else None,
        )

    def _get_movement_parameters(
        self, command: Optional[GeminiAgentResponse]
    ) -> Optional[Dict[str, float]]:
        """
        Get movement parameters for a given command string.
        """
        if command is None:
            logger.warning("Received None GeminiAgentResponse, returning None.")
            return None

        if command.next_robot_move is None:
            logger.warning("Received None next_robot_move, returning None.")
            return None

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
        return command_map.get(command.next_robot_move)

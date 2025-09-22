import asyncio
import contextlib
from collections import deque
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import httpx
from loguru import logger

from phosphobot.configs import config
from phosphobot.models import ChatRequest, ChatResponse
from phosphobot.utils import get_local_ip


class PhosphobotClient:
    def __init__(
        self,
        write_to_log: Optional[Callable[[str, str], None]] = None,
        log_to_ui: bool = True,
    ) -> None:
        """
        :param write_to_log: Callback like screen._write_to_log(content, who).
        :param log_to_ui: Enable/disable logging to UI.
        """
        self.server_url = f"http://{get_local_ip()}:{config.PORT}"
        self.client = httpx.AsyncClient(base_url=self.server_url, timeout=5.0)

        self._write_to_log = write_to_log
        self._log_to_ui = log_to_ui

    def _log(self, message: str, who: str = "system") -> None:
        """Internal logging helper."""
        if self._log_to_ui and self._write_to_log:
            self._write_to_log(message, who)

    async def _safe_request(
        self, method: str, url: str, **kwargs: Any
    ) -> Optional[httpx.Response]:
        """Wrapper around httpx requests with error handling + logging."""
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            msg = f"API error {e.response.status_code} on {url}: {e.response.text}"
            self._log(f"[bold red]Error:[/bold red] [red]{msg}[/red]")
            return None
        except httpx.RequestError as e:
            msg = f"Request error calling {url}: {e}"
            self._log(f"[bold red]Error:[/bold red] [red]{msg}[/red]")
            return None

    # -----------------------
    # Public API methods
    # -----------------------

    async def status(self) -> Optional[Dict[str, Any]]:
        response = await self._safe_request("GET", "/status", timeout=10.0)
        return response.json() if response else None

    async def move_joints(self, joints: List[float]) -> None:
        await self._safe_request("POST", "/joints/write", json={"joints": joints})

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
        await self._safe_request(
            "POST",
            "/move/relative",
            json={"x": x, "y": y, "z": z, "rx": rx, "ry": ry, "rz": rz, "open": open},
        )

    async def get_camera_image(
        self,
        camera_ids: Optional[List[int]] = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[int, str]]:
        params = {}
        if resize:
            params["resize_x"], params["resize_y"] = resize

        response = await self._safe_request(
            "GET", "/frames", params=params, timeout=3.0
        )
        if response is None:
            return None

        reponse_json = response.json()
        output: Dict[int, str] = {}

        for camera_id in camera_ids or reponse_json.keys():
            try:
                camera_id = int(camera_id)
            except ValueError:
                self._log(f"Invalid camera ID: {camera_id}. Skipping.")
                continue

            if str(camera_id) in reponse_json:
                output[camera_id] = reponse_json[str(camera_id)]
            else:
                self._log(f"Camera {camera_id} not found in response.")

        return output

    async def move_init(self) -> None:
        await self._safe_request("POST", "/move/init")

    async def chat(self, chat_request: ChatRequest) -> Optional[ChatResponse]:
        response = await self._safe_request(
            "POST",
            "/ai-control/chat",
            json=chat_request.model_dump(mode="json"),
        )
        if response is None:
            return None
        return ChatResponse.model_validate(response.json())

    async def log_chat(self, chat_request: ChatRequest) -> None:
        await self._safe_request(
            "POST",
            "/ai-control/chat/log",
            json=chat_request.model_dump(mode="json"),
        )

    async def start_recording(self, dataset_name: str, instruction: str) -> None:
        await self._safe_request(
            "POST",
            "/recording/start",
            json={
                "dataset_name": dataset_name,
                "instruction": instruction,
                "save_cartesian": True,
            },
        )

    async def stop_recording(self) -> None:
        await self._safe_request("POST", "/recording/stop", json={"save": True})


class RoboticAgent:
    def __init__(
        self,
        images_sizes: Optional[Tuple[int, int]] = (256, 256),
        write_to_log: Optional[Callable[[str, str], None]] = None,
    ):
        self.resize = images_sizes

        self.task_description: Optional[str] = None
        self.dataset_name: str = "chat_dataset"

        self.phosphobot_client = PhosphobotClient(write_to_log=write_to_log)
        self.control_mode: Literal["ai", "keyboard"] = "ai"
        self.action_queue: deque = deque(maxlen=100)
        self.chat_history: List[Union[ChatRequest, ChatResponse]] = []
        self.command_history: List[str] = []

        # For managing AI API call cancellation
        self._current_ai_task: Optional[asyncio.Task] = None
        self._ai_cancellation_event = asyncio.Event()

    def add_action(self, action: Union[ChatResponse, str]) -> None:
        """
        Add an action to the queue to be executed.
        """
        self.action_queue.append(action)

    def toggle_control_mode(self) -> str:
        """
        Toggle between AI and keyboard control modes.
        Returns the new mode.
        """
        old_mode = self.control_mode
        self.control_mode = "keyboard" if self.control_mode == "ai" else "ai"
        logger.info(f"Switched to {self.control_mode} control mode")

        # If switching from AI to keyboard, cancel any ongoing AI API call
        if old_mode == "ai" and self.control_mode == "keyboard":
            self._cancel_ai_task()

        return self.control_mode

    def set_control_mode(self, mode: Literal["ai", "keyboard"]) -> None:
        """
        Set the control mode directly.
        """
        old_mode = self.control_mode
        self.control_mode = mode
        logger.info(f"Set control mode to {self.control_mode}")

        # If switching from AI to keyboard, cancel any ongoing AI API call
        if old_mode == "ai" and self.control_mode == "keyboard":
            self._cancel_ai_task()

    def _cancel_ai_task(self) -> None:
        """
        Cancel the current AI API call if it's running.
        We set the cancellation event first (so _get_ai_response can notice it),
        then cancel the task as a fallback.
        """
        # Signal cancellation to any waiter
        self._ai_cancellation_event.set()

        # If there's a running task, cancel it (the _get_ai_response handler will await it and swallow CancelledError)
        if self._current_ai_task and not self._current_ai_task.done():
            try:
                self._current_ai_task.cancel()
            except Exception:
                # be defensive: log and ignore any unexpected cancellation errors
                logger.exception("Failed to cancel current AI task")
        logger.info("Signalled cancellation for ongoing AI API call")

    async def _get_ai_response(
        self, chat_request: ChatRequest
    ) -> Optional[ChatResponse]:
        """
        Get AI response with cancellation support.

        Ensures that cancellation of the AI *API* call is handled locally and
        does not raise CancelledError out of this function.
        """
        # Reset cancellation event
        self._ai_cancellation_event.clear()

        # Create task for the API call that can be cancelled
        self._current_ai_task = asyncio.create_task(
            self.phosphobot_client.chat(chat_request=chat_request)
        )

        # Create a task for the cancellation event (avoid passing a bare coroutine to asyncio.wait)
        cancel_wait_task = asyncio.create_task(self._ai_cancellation_event.wait())

        try:
            done, pending = await asyncio.wait(
                [self._current_ai_task, cancel_wait_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # If cancellation event finished first -> cancel API task and return None
            if cancel_wait_task in done:
                if not self._current_ai_task.done():
                    self._current_ai_task.cancel()
                    try:
                        await self._current_ai_task
                    except asyncio.CancelledError:
                        logger.info("AI API call was cancelled by user")
                    except Exception:
                        logger.exception("AI API call raised while cancelling")
                else:
                    # if api task already done but cancel event raced in, be safe:
                    if self._current_ai_task.cancelled():
                        logger.info("AI API task was cancelled")
                    else:
                        # If the API task already finished (rare race), try to get result safely
                        try:
                            return self._current_ai_task.result()
                        except Exception:
                            logger.exception(
                                "Error getting AI API call result after cancellation"
                            )
                return None

            # If API call completed first
            if self._current_ai_task in done:
                # If the task was cancelled, return None
                if self._current_ai_task.cancelled():
                    logger.info("AI API task ended up cancelled")
                    return None
                # Otherwise return the result, catching exceptions so they don't propagate CancelledError upward
                try:
                    return self._current_ai_task.result()
                except Exception:
                    logger.exception("AI API call raised an exception")
                    return None

            # Fallback: no result
            return None
        finally:
            # Cleanup the cancellation waiter task if it's still pending
            if not cancel_wait_task.done():
                cancel_wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_wait_task

    async def execute_chat_response(
        self, chat_response: Optional[ChatResponse]
    ) -> None:
        """
        Execute the AI command by moving the robot.
        """
        if chat_response is None:
            logger.warning("No chat response received. Skipping execution.")
            return None

        if chat_response.command is not None:
            self.command_history.append(chat_response.command)

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

    async def execute_command(self, command: str) -> Optional[Dict[str, float]]:
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

    def add_to_chat_history(
        self, chat_request: ChatRequest, chat_response: ChatResponse
    ) -> None:
        """
        Add the chat request and response to the chat history.
        """
        self.chat_history.extend([chat_request, chat_response])

    async def process_action_queue(self) -> bool:
        """
        Process one action from the queue if available.
        Returns True if an action was processed, False otherwise.
        """
        if not self.action_queue:
            return False

        action = self.action_queue.popleft()

        if isinstance(action, str):  # Manual command
            await self.execute_command(action)
        elif isinstance(action, ChatResponse):  # AI command
            await self.execute_chat_response(action)

        return True

    async def run(self) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """
        An async generator that yields events for the UI to handle.
        Events are tuples of (event_type: str, payload: dict).
        """
        if not self.task_description:
            yield (
                "log",
                {
                    "text": "No task description provided. Please set a task description."
                },
            )
            return

        yield "start_step", {"desc": "Checking robot status."}
        self.robot_status = await self.phosphobot_client.status()
        yield "step_output", {"desc": f"Robot status: {self.robot_status}"}
        yield "step_done", {"success": True}

        # Control mode setup
        yield (
            "start_step",
            {"desc": f"{self.control_mode.upper()} control mode enabled."},
        )
        yield "step_done", {"success": True}

        # Start recording
        yield "start_step", {"desc": "🔴 Starting recording."}
        await self.phosphobot_client.start_recording(
            dataset_name=self.dataset_name, instruction=self.task_description
        )

        step_count = 0
        max_steps = 50
        chat_logged = False

        while step_count < max_steps:
            # Process any available actions from the queue
            action_processed = await self.process_action_queue()

            if action_processed is True:
                continue  # Skip to the next iteration if an action was processed

            # Queue is empty: Add a new action based on the current mode
            if self.control_mode == "ai":
                # AI MODE: Generate new action if queue is empty
                step_count += 1
                yield (
                    "log",
                    {
                        "text": f"Step {step_count}/{max_steps} - AI mode - Generating command..."
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

                # Check again if we've been switched to keyboard mode
                if self.control_mode == "keyboard":
                    continue

                chat_request = ChatRequest(
                    prompt=self.task_description,
                    # Convert dict to list of base64 strings
                    images=list(images.values()),
                    command_history=self.command_history,
                )
                if not chat_logged:
                    # Log the initial chat request
                    await self.phosphobot_client.log_chat(chat_request=chat_request)
                    chat_logged = True

                # Get AI response with cancellation support
                chat_response = await self._get_ai_response(chat_request)

                # Check if we've been switched to keyboard mode during the API call
                if self.control_mode == "keyboard" or chat_response is None:
                    continue

                yield (
                    "step_output",
                    {"output": f"AI command: {chat_response.model_dump()}"},
                )
                self.add_to_chat_history(
                    chat_request=chat_request, chat_response=chat_response
                )
                # Add the AI command to the queue for execution
                self.add_action(chat_response)
            else:
                # KEYBOARD MODE: Wait for user input without consuming a step
                await asyncio.sleep(0.1)

        # Stop recording
        yield "start_step", {"desc": "🔴 Recording stopped."}
        await self.phosphobot_client.stop_recording()

        yield "step_done", {"success": True}
        yield (
            "log",
            {"text": f"Robotic agent run completed in {self.control_mode} mode."},
        )

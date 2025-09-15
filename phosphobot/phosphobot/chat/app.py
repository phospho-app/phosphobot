import asyncio
import datetime
import logging
from typing import Iterable, Optional

from rich.text import Text
from textual.app import App, ComposeResult, SystemCommand
from textual.events import Key
from textual.message import Message
from textual.reactive import var
from textual.screen import Screen
from textual.widgets import Footer, Input, RichLog
from textual.worker import Worker

from phosphobot.chat.agent import RoboticAgent
from phosphobot.configs import config
from phosphobot.utils import get_local_ip
from phosphobot.chat.utils import ascii_test_tube, KEYBOARD_CONTROl_TEXT


class AgentScreen(Screen):
    """
    The main screen for the phosphobot chat interface.
    This screen handles user input, displays chat logs, and manages agent interactions.
    """

    def compose(self) -> ComposeResult:
        """Create the UI layout and widgets."""
        yield RichLog(id="chat-log", wrap=True, highlight=True)
        yield Input(
            placeholder="Click here, type a prompt and press Enter to send",
            id="chat-input",
        )
        yield Footer()

    def on_key(self, event: Key) -> None:
        """Handle key presses at screen level to bypass input focus for keyboard control."""
        app = self.app
        if not isinstance(app, AgentApp) or not app.current_agent:
            return

        # Keyboard control keys should work regardless of focus
        if app.current_agent.control_mode == "keyboard":
            # Movement keys
            if event.key == "up":
                app.action_keyboard_forward()
                event.prevent_default()
            elif event.key == "down":
                app.action_keyboard_backward()
                event.prevent_default()
            elif event.key == "left":
                app.action_keyboard_left()
                event.prevent_default()
            elif event.key == "right":
                app.action_keyboard_right()
                event.prevent_default()
            elif event.key == "d":
                app.action_keyboard_up()
                event.prevent_default()
            elif event.key == "c":
                app.action_keyboard_down()
                event.prevent_default()
            # Gripper toggle
            elif event.key == "space":
                app.action_keyboard_gripper()
                event.prevent_default()

        # Toggle key always works
        if event.key == "ctrl+t":
            app.action_toggle_control_mode()
            event.prevent_default()

    def _write_welcome_message(self) -> None:
        self._write_to_log(
            f"""🧪 Welcome to phosphobot chat!

{ascii_test_tube()}

[grey46]Access the phosphobot dashboard here: http://{get_local_ip()}:{config.PORT}

💡 Tip: Press Ctrl+T for keyboard control, Ctrl+S to stop the agent, and Ctrl+P for commands.[/grey46]
""",
            "system",
        )
        self._write_to_log("Type a prompt and press Enter to start.", "agent")

    def on_mount(self) -> None:
        """
        Display welcome message and initial instructions when the screen is mounted.
        """
        self._write_welcome_message()
        self.query_one(Input).focus()

    def set_running_state(self, running: bool) -> None:
        """Update UI based on agent running state."""
        input_widget = self.query_one(Input)
        app = self.app

        # Check if we're in keyboard control mode
        keyboard_control_mode = (
            isinstance(app, AgentApp)
            and app.current_agent
            and app.current_agent.control_mode == "keyboard"
        )

        if keyboard_control_mode:
            self.app.sub_title = "Keyboard Control Active - See command layout below"
            input_widget.disabled = True
            input_widget.placeholder = "Keyboard control active - keys control robot"
            # Show command layout
            self._write_to_log(KEYBOARD_CONTROl_TEXT.strip(), "system")
            # Remove focus from input so keys work
            # self.focus()
        elif running:
            self.app.sub_title = "Agent is running..."
            input_widget.disabled = running
            input_widget.placeholder = "Agent running... (Ctrl+S to stop)"
        else:
            self.app.sub_title = "Ready"
            input_widget.disabled = False
            input_widget.placeholder = "Type a prompt and press Enter..."
            input_widget.focus()

    def _write_to_log(self, content: str, who: str) -> None:
        """Write a formatted message to the RichLog."""
        log = self.query_one(RichLog)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        style, prefix = "", ""
        if who == "user":
            style, prefix = "bold white", f"[{timestamp} YOU] "
        elif who == "agent":
            style, prefix = "bold green", f"[{timestamp} AGENT] "
        elif who == "system":
            style, prefix = "italic green", f"[{timestamp} SYS] "
        log.write(Text(prefix, style=style) + Text.from_markup(content))


class RichLogHandler(logging.Handler):
    def __init__(self, rich_log: RichLog) -> None:
        super().__init__()
        self.rich_log = rich_log

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self.rich_log.write(f"[DIM]{record.name}[/DIM] - {message}")


class AgentApp(App):
    """
    The main application class for the phosphobot chat interface.
    This app manages the agent lifecycle, user input, and UI updates.
    """

    TITLE = "phosphobot chat"
    SUB_TITLE = "Ready"

    SCREENS = {"main": AgentScreen}

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+p", "command_palette", "Commands"),
        ("ctrl+s", "stop_agent", "Stop Agent"),
    ]

    CSS = """
    #chat-log {
        height: 1fr;
        border: round $accent;
        margin: 1 2;
    }
    #chat-input {
        dock: bottom;
        height: 8;
        margin: 0 2 1 2;
    }
    """

    is_agent_running: var[bool] = var(False)
    worker: Optional[Worker] = None
    current_agent: RoboticAgent
    gripper_is_open: bool = True  # Track gripper state

    class AgentUpdate(Message):
        def __init__(self, event_type: str, payload: dict) -> None:
            self.event_type = event_type
            self.payload = payload
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self.current_agent = RoboticAgent()

    def _get_main_screen(self) -> Optional[AgentScreen]:
        """Safely gets the main screen instance, returning None if not ready."""
        try:
            screen = self.get_screen("main")
            if isinstance(screen, AgentScreen):
                return screen
        except KeyError:
            return None
        return None

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.push_screen("main")

    def watch_is_agent_running(self, running: bool) -> None:
        """Update the main screen's UI based on the running state."""
        screen = self._get_main_screen()
        if screen and screen.is_mounted:
            screen.set_running_state(running)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return None

        screen = self._get_main_screen()
        if not screen:
            return None

        screen.query_one(Input).clear()
        self._handle_prompt(prompt, screen)

    def _handle_prompt(self, prompt: str, screen: AgentScreen) -> None:
        if self.is_agent_running:
            screen._write_to_log("An agent is already running.", "system")
            return None
        screen._write_to_log(prompt, "user")

        if prompt.strip() == "/init":
            screen._write_to_log("Moving robot to initial position", "system")
            asyncio.create_task(self.current_agent.phosphobot_client.move_init())
            return None

        # Edit prompt of the agent
        self.current_agent.task_description = prompt
        self.worker = self.run_worker(
            self._run_agent(self.current_agent), exclusive=True
        )

    async def _run_agent(self, agent: RoboticAgent) -> None:
        self.is_agent_running = True
        try:
            async for event_type, payload in agent.run():
                self.post_message(self.AgentUpdate(event_type, payload))
        except asyncio.CancelledError:
            self.post_message(
                self.AgentUpdate(
                    "log", {"text": "asyncio.CancelledError: Agent stopped."}
                )
            )
            # Call stop recording
            await agent.phosphobot_client.stop_recording()
            self.post_message(
                self.AgentUpdate("log", {"text": "🔴 Recording stopped."})
            )

        finally:
            self.is_agent_running = False

    def on_agent_app_agent_update(self, message: AgentUpdate) -> None:
        self._handle_agent_event(message.event_type, message.payload)

    def _handle_agent_event(self, event_type: str, payload: dict) -> None:
        screen = self._get_main_screen()
        if not screen:
            return

        log = screen.query_one(RichLog)
        if event_type == "log":
            screen._write_to_log(payload.get("text", ""), "system")
        elif event_type == "start_step":
            screen._write_to_log(f"Starting: {payload['desc']}", "agent")
        elif event_type == "step_output":
            log.write(payload.get("output", ""))
        elif event_type == "step_error":
            error_message = payload.get("error", "An error occurred.")
            screen._write_to_log(
                f"[bold red]Error:[/bold red] {error_message}", "agent"
            )
        elif event_type == "step_done":
            log.write("")
            screen._write_to_log("Step status: [bold green][DONE][/]", "agent")
            log.write("")

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        """
        Generate system commands for the command palette (Ctrl+P menu).
        """
        for function in [
            self.action_new_chat,
            self.action_stop_agent,
            self.action_toggle_control_mode,
            self.action_change_dataset_name,
        ]:
            command_name = function.__name__.replace("action_", "")
            command_description = function.__doc__ or "No description available."
            yield SystemCommand(
                command_name.replace("_", " ").title(),
                command_description,
                function,
            )

        # Base commands
        yield SystemCommand(
            "Quit the application",
            "Quit the application as soon as possible",
            self.action_quit,
        )
        if screen.query("HelpPanel"):
            yield SystemCommand(
                "Hide keys and help panel",
                "Hide the keys and widget help panel",
                self.action_hide_help_panel,
            )
        else:
            yield SystemCommand(
                "Show keys and help panel",
                "Show help for the focused widget and a summary of available keys",
                self.action_show_help_panel,
            )

    def action_stop_agent(self) -> None:
        """Stop the currently running agent."""
        screen = self._get_main_screen()
        if not screen:
            return

        if self.is_agent_running and self.worker:
            self.worker.cancel()
            screen._write_to_log("Interrupt requested. Stopping agent...", "system")
            # screen.set_running_state(False)
        else:
            screen._write_to_log("No agent is currently running.", "system")

    def action_new_chat(self) -> None:
        """Start a new chat session by clearing the log and stopping any running agent."""
        screen = self._get_main_screen()
        if not screen:
            return

        if self.is_agent_running:
            self.action_stop_agent()
        screen.query_one(RichLog).clear()
        screen._write_welcome_message()

    def action_change_dataset_name(self) -> None:
        """Change the dataset name in which the agent will save its data."""

    def action_toggle_control_mode(self) -> None:
        """Toggle between AI control and keyboard control mode."""
        screen = self._get_main_screen()
        if not screen or not self.current_agent:
            screen._write_to_log(
                "No agent available for control.", "system"
            ) if screen else None
            return

        mode = self.current_agent.toggle_control_mode()
        screen._write_to_log(f"Switched to {mode} control mode", "system")

        # Update UI to reflect new mode
        screen.set_running_state(self.is_agent_running)

    def action_keyboard_forward(self) -> None:
        self._send_command("move_forward")

    def action_keyboard_backward(self) -> None:
        self._send_command("move_backward")

    def action_keyboard_left(self) -> None:
        self._send_command("move_left")

    def action_keyboard_right(self) -> None:
        self._send_command("move_right")

    def action_keyboard_up(self) -> None:
        self._send_command("move_up")

    def action_keyboard_down(self) -> None:
        self._send_command("move_down")

    def action_keyboard_gripper(self) -> None:
        """Toggle gripper between open and closed."""
        if self.gripper_is_open:
            self._send_command("close_gripper")
            self.gripper_is_open = False
        else:
            self._send_command("open_gripper")
            self.gripper_is_open = True

    def _send_command(self, command: str) -> None:
        """Send a command to the current agent."""
        screen = self._get_main_screen()
        if not screen or not self.current_agent:
            return

        self.current_agent.add_action(action=command)
        screen._write_to_log(f"Command: {command}", "system")


if __name__ == "__main__":
    app = AgentApp()
    app.run()

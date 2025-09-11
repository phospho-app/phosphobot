import asyncio
import datetime
import logging
from typing import Any, Iterable, Optional

from rich.text import Text
from textual.app import App, ComposeResult, SystemCommand
from textual.message import Message
from textual.reactive import var
from textual.screen import Screen
from textual.widgets import Footer, Input, RichLog
from textual.worker import Worker
from textual.events import Key

from phosphobot.chat.agent import RoboticAgent
from phosphobot.configs import config
from phosphobot.utils import get_local_ip


class AgentScreen(Screen):
    """The main screen for the agent application."""

    def compose(self) -> ComposeResult:
        """Create the UI layout and widgets."""
        yield RichLog(id="chat-log", wrap=True, highlight=True)
        yield Input(
            placeholder="Click here, type a prompt and press Enter to send",
            id="chat-input",
        )
        yield Footer()

    def on_key(self, event: Key) -> None:
        """Handle key presses at screen level to bypass input focus for manual control."""
        app = self.app
        if not isinstance(app, AgentApp) or not app.current_agent:
            return

        # Manual control keys should work regardless of focus
        if app.current_agent.manual_control:
            # Movement keys
            if event.key == "up":
                app.action_manual_forward()
                event.prevent_default()
            elif event.key == "down":
                app.action_manual_backward()
                event.prevent_default()
            elif event.key == "left":
                app.action_manual_left()
                event.prevent_default()
            elif event.key == "right":
                app.action_manual_right()
                event.prevent_default()
            elif event.key == "d":
                app.action_manual_up()
                event.prevent_default()
            elif event.key == "c":
                app.action_manual_down()
                event.prevent_default()
            # Gripper toggle
            elif event.key == "space":
                app.action_gripper_toggle()
                event.prevent_default()

        # Toggle key always works
        if event.key == "ctrl+t":
            app.action_toggle_control_mode()
            event.prevent_default()

    def on_mount(self) -> None:
        """Focus the input when the screen is mounted."""
        self._write_to_log(
            "🧪 Welcome to phosphobot chat!\n\n"
            + f"Access the dashboard here: http://{get_local_ip()}:{config.PORT}\n"
            + "\nEnter a prompt in the box below or press Ctrl+P for commands.\n"
            + "💡 Tip: Press Ctrl+T to take manual control, Ctrl+S to stop the agent!",
            "system",
        )
        self.query_one(Input).focus()

    def set_running_state(self, running: bool) -> None:
        """Update UI based on agent running state."""
        input_widget = self.query_one(Input)
        app = self.app

        # Check if we're in manual control mode
        manual_mode = (
            isinstance(app, AgentApp)
            and app.current_agent
            and app.current_agent.manual_control
        )

        if manual_mode:
            self.app.sub_title = "Manual Control Active - See command layout below"
            input_widget.disabled = True
            input_widget.placeholder = "Manual control active - keys control robot"
            # Show command layout
            self._show_manual_controls()
            # Remove focus from input so keys work
            self.focus()
        elif running:
            self.app.sub_title = "Agent is running..."
            input_widget.disabled = running
            input_widget.placeholder = "Agent running... (Ctrl+I to stop)"
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

    def _show_manual_controls(self) -> None:
        """Display the manual control layout."""
        controls_text = """
[bold green]🎮 Manual Control Commands:[/bold green]

Movement:
  ↑ ↓ ← →  Move Forward/Back/Left/Right
  D C      Move Up/Down

Gripper:
  Space    Toggle Open/Close

Mode:
  Ctrl+T   Toggle AI/Manual control  
  Ctrl+S   Stop Agent

[dim]Press keys to control the robot immediately[/dim]
        """
        self._write_to_log(controls_text.strip(), "system")


class RichLogHandler(logging.Handler):
    def __init__(self, rich_log: RichLog) -> None:
        super().__init__()
        self.rich_log = rich_log

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self.rich_log.write(f"[DIM]{record.name}[/DIM] - {message}")


class AgentApp(App):
    """A terminal-based chat interface for an agent."""

    TITLE = "Agent Terminal"
    SUB_TITLE = "Ready"

    # REMOVED: The COMMANDS class variable is gone to avoid overwriting defaults.
    # COMMANDS = {AgentCommands}

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
    current_agent: Optional[RoboticAgent] = None
    gripper_is_open: bool = True  # Track gripper state

    class AgentUpdate(Message):
        def __init__(self, event_type: str, payload: dict) -> None:
            self.event_type = event_type
            self.payload = payload
            super().__init__()

    def __init__(self) -> None:
        super().__init__()

    def _get_main_screen(self) -> Optional[AgentScreen]:
        """Safely gets the main screen instance, returning None if not ready."""
        try:
            screen = self.get_screen("main")
            if isinstance(screen, AgentScreen):
                return screen
        except KeyError:
            return None
        return None

    # In AgentApp's on_mount
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.push_screen("main")

    def watch_is_running(self, running: bool) -> None:
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
        agent = RoboticAgent(task_description=prompt)
        self.current_agent = agent  # Store reference for manual control

        if prompt.strip() == "/init":
            screen._write_to_log("Moving robot to initial position", "system")
            asyncio.create_task(agent.phosphobot_client.move_init())
            return None

        self.worker = self.run_worker(self._run_agent(agent), exclusive=True)

    async def _run_agent(self, agent: RoboticAgent) -> None:
        self.is_agent_running = True
        try:
            async for event_type, payload in agent.run():
                self.post_message(self.AgentUpdate(event_type, payload))
        except asyncio.CancelledError:
            self.post_message(self.AgentUpdate("log", {"text": "Agent stopped."}))
        finally:
            self.is_agent_running = False
            self.current_agent = None

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
        yield SystemCommand(
            "New chat",
            "Clear the log output and start a new chat",
            self.action_clear_log,
        )
        yield SystemCommand(
            "Stop Agent", "Stop the currently running agent.", self.action_stop_agent
        )
        yield SystemCommand(
            "Toggle Control Mode",
            "Switch between AI and manual control.",
            self.action_toggle_control_mode,
        )
        yield from super().get_system_commands(screen)

    def action_stop_agent(self) -> None:
        """Stops the agent task. Called by binding or command palette."""
        screen = self._get_main_screen()
        if not screen:
            return

        if self.is_agent_running and self.worker:
            self.worker.cancel()
            screen._write_to_log("Interrupt requested. Stopping agent...", "system")
            # If we were in manual mode, switch back to AI mode and update UI
            if self.current_agent and self.current_agent.manual_control:
                self.current_agent.manual_control = False
                screen._write_to_log(
                    "Manual control disabled - agent stopped.", "system"
                )
                # Update UI to show normal state
                screen.set_running_state(False)
        else:
            screen._write_to_log("No agent is currently running.", "system")

    def action_clear_log(self) -> None:
        """Clears the log. Called by command palette."""
        screen = self._get_main_screen()
        if not screen:
            return

        screen.query_one(RichLog).clear()
        screen._write_to_log("Log cleared.", "system")

    def action_toggle_control_mode(self) -> None:
        """Toggle between AI and manual control mode."""
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

    def action_manual_forward(self) -> None:
        """Send manual forward command."""
        self._send_manual_command("move_forward")

    def action_manual_backward(self) -> None:
        """Send manual backward command."""
        self._send_manual_command("move_backward")

    def action_manual_left(self) -> None:
        """Send manual left command."""
        self._send_manual_command("move_left")

    def action_manual_right(self) -> None:
        """Send manual right command."""
        self._send_manual_command("move_right")

    def action_manual_up(self) -> None:
        """Send manual up command."""
        self._send_manual_command("move_up")

    def action_manual_down(self) -> None:
        """Send manual down command."""
        self._send_manual_command("move_down")

    def action_gripper_toggle(self) -> None:
        """Toggle gripper between open and closed."""
        if self.gripper_is_open:
            self._send_manual_command("close_gripper")
            self.gripper_is_open = False
        else:
            self._send_manual_command("open_gripper")
            self.gripper_is_open = True

    def _send_manual_command(self, command: str) -> None:
        """Send a manual command to the current agent."""
        screen = self._get_main_screen()
        if not screen or not self.current_agent:
            return

        if not self.current_agent.manual_control:
            screen._write_to_log(
                "Manual control not active. Press 'ctrl+T' to toggle.", "system"
            )
            return

        self.current_agent.set_manual_command(command)
        screen._write_to_log(f"Manual command: {command}", "system")


if __name__ == "__main__":
    app = AgentApp()
    app.run()

import asyncio
import datetime
import logging
from typing import Iterable, Optional

from phosphobot.chat.agent import RoboticAgent
from rich.text import Text
from textual.app import App, ComposeResult, SystemCommand
from textual.message import Message
from textual.reactive import var
from textual.screen import Screen
from textual.widgets import Footer, Input, RichLog
from textual.worker import Worker


class AgentScreen(Screen):
    """The main screen for the agent application."""

    def compose(self) -> ComposeResult:
        """Create the UI layout and widgets."""
        yield RichLog(id="chat-log", wrap=True, highlight=True)
        yield Input(placeholder="Type a prompt and press Enter...", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input when the screen is mounted."""
        self._write_to_log(
            "Welcome! Enter a prompt or press Ctrl+P for commands.", "system"
        )
        self.query_one(Input).focus()

    def set_running_state(self, running: bool) -> None:
        """Update UI based on agent running state."""
        input_widget = self.query_one(Input)
        self.app.sub_title = "Agent is running..." if running else "Ready"
        input_widget.disabled = running
        input_widget.placeholder = (
            "Agent running... (Ctrl+I to stop)"
            if running
            else "Type a prompt and press Enter..."
        )
        if not running:
            input_widget.focus()

    def _write_to_log(self, content: str, who: str):
        """Write a formatted message to the RichLog."""
        log = self.query_one(RichLog)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        style, prefix = "", ""
        if who == "user":
            style, prefix = "bold magenta", f"[{timestamp} YOU] "
        elif who == "agent":
            style, prefix = "bold blue", f"[{timestamp} AGENT] "
        elif who == "system":
            style, prefix = "italic dim", f"[{timestamp} SYS] "
        log.write(Text(prefix, style=style) + Text.from_markup(content))


class RichLogHandler(logging.Handler):
    def __init__(self, rich_log: RichLog):
        super().__init__()
        self.rich_log = rich_log

    def emit(self, record):
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
        ("ctrl+i", "stop_agent", "Stop Agent"),
    ]

    CSS = """
    #chat-log {
        height: 1fr;
        border: round $accent;
        margin: 1 2;
    }
    #chat-input {
        dock: bottom;
        height: 3;
        margin: 0 2 1 2;
    }
    """

    is_running: var[bool] = var(False)
    worker: Optional[Worker] = None

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
            return

        screen = self._get_main_screen()
        if not screen:
            return

        screen.query_one(Input).clear()
        self._handle_prompt(prompt, screen)

    def _handle_prompt(self, prompt: str, screen: AgentScreen):
        if self.is_running:
            screen._write_to_log("An agent is already running.", "system")
            return
        screen._write_to_log(prompt, "user")
        agent = RoboticAgent(task_description=prompt)
        if prompt.strip() == "/init":
            screen._write_to_log("Moving robot to initial position", "system")
            asyncio.create_task(agent.phosphobot_client.move_init())
            return

        self.worker = self.run_worker(self._run_agent(agent), exclusive=True)

    async def _run_agent(self, agent: RoboticAgent) -> None:
        self.is_running = True
        try:
            async for event_type, payload in agent.run():
                self.post_message(self.AgentUpdate(event_type, payload))
        except asyncio.CancelledError:
            self.post_message(self.AgentUpdate("log", {"text": "Agent stopped."}))
        finally:
            self.is_running = False
            self.agent_task = None

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
            screen._write_to_log(f"Step status: [bold green][DONE][/]", "agent")
            log.write("")

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        # TODO: Add keyboard control
        yield SystemCommand(
            "New chat",
            "Clear the log output and start a new chat",
            self.action_clear_log,
        )
        yield SystemCommand(
            "Stop Agent", "Stop the currently running agent.", self.action_stop_agent
        )
        yield from super().get_system_commands(screen)

    def action_stop_agent(self) -> None:
        """Stops the agent task. Called by binding or command palette."""
        screen = self._get_main_screen()
        if not screen:
            return

        if self.is_running and self.worker:
            self.worker.cancel()
            screen._write_to_log("Interrupt requested. Stopping agent...", "system")
        else:
            screen._write_to_log("No agent is currently running.", "system")

    def action_clear_log(self) -> None:
        """Clears the log. Called by command palette."""
        screen = self._get_main_screen()
        if not screen:
            return

        screen.query_one(RichLog).clear()
        screen._write_to_log("Log cleared.", "system")


if __name__ == "__main__":
    app = AgentApp()
    app.run()

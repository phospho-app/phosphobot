import asyncio
import datetime
import logging
import re
from typing import Iterable, List, Optional, Tuple

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult, SystemCommand
from textual.events import Key
from textual.message import Message
from textual.reactive import var
from textual.screen import Screen
from textual.widgets import Footer, Input, RichLog, Static
from textual.worker import Worker

from phosphobot.chat.agent import RoboticAgent
from phosphobot.chat.utils import KEYBOARD_CONTROl_TEXT, ascii_test_tube
from phosphobot.configs import config
from phosphobot.utils import get_local_ip

# ---- Command definitions (single source of truth) ----
COMMANDS = [
    {"cmd": "/help", "desc": "Show help", "usage": "/help"},
    {"cmd": "/init", "desc": "Move robot to initial position", "usage": "/init"},
    {"cmd": "/stop", "desc": "Stop the agent", "usage": "/stop"},
    {
        "cmd": "/dataset",
        "desc": "Set dataset name for recording the agent's actions. If no name is provided, returns the current dataset name.",
        "usage": "/dataset <name>",
    },
    {"cmd": "/quit", "desc": "Quit the application", "usage": "/quit"},
    {"cmd": "/new", "desc": "Start a new chat session", "usage": "/new"},
]


class SuggestionBox(Static):
    """
    Simplified suggestion box that displays command suggestions.
    """

    def __init__(
        self, id: Optional[str] = None, max_suggestions: int = 5
    ) -> None:  # Default to 5
        super().__init__("", id=id)
        self.suggestions: List[Tuple[str, str]] = []
        self.selected: int = 0
        self.max_suggestions = max_suggestions  # Store the max suggestions
        self.styles.display = "none"  # Start hidden

    def _build_markup(self) -> str:
        """Return the markup string for current suggestions & selected index."""
        lines: List[str] = []
        for i, (cmd, desc) in enumerate(self.suggestions):
            marker = "→" if i == self.selected else "  "
            if i == self.selected:
                lines.append(f"{marker} [bold]{cmd}[/bold] [dim]{desc}[/dim]")
            else:
                lines.append(f"{marker} {cmd} [dim]{desc}[/dim]")
        return "\n".join(lines)

    def update_suggestions(self, suggestions: List[Tuple[str, str]]) -> None:
        """Update suggestions and show/hide the box accordingly."""
        self.suggestions = suggestions[
            : self.max_suggestions
        ]  # Use the max_suggestions parameter
        self.selected = 0

        if not self.suggestions:
            self.styles.display = "none"
            self.update("")
            return

        self.styles.display = "block"
        markup = self._build_markup()
        self.update(Text.from_markup(markup))

    # ... rest of the class remains the same

    def cycle(self, delta: int) -> None:
        """Move selection up or down."""
        if not self.suggestions:
            return
        self.selected = (self.selected + delta) % len(self.suggestions)
        markup = self._build_markup()
        self.update(Text.from_markup(markup))

    def get_selected(self) -> Optional[Tuple[str, str]]:
        """Get the currently selected suggestion."""
        if not self.suggestions:
            return None
        return self.suggestions[self.selected]


class CustomInput(Input):
    """Custom Input widget that coordinates with suggestion box for tab completion."""

    async def key_tab(self, event: Key) -> None:
        # Get the suggestion box from the screen
        screen = self.app.screen
        if not isinstance(screen, AgentScreen):
            return
        suggestion_box = screen.query_one("#suggest-box")
        if not isinstance(suggestion_box, SuggestionBox):
            return
        if suggestion_box.suggestions:
            # Prevent default tab behavior (focus cycling)
            event.prevent_default()
            # Get the selected suggestion
            selected = suggestion_box.get_selected()
            if selected:
                completion = selected[0]
                self.value = completion + " "
                # Move cursor to the end
                self.cursor_position = len(self.value)
            return
        # Default behavior if no suggestions
        # await self.action_focus_next()

    async def key_ctrl_c(self, event: Key) -> None:
        # Let the Ctrl+C event bubble up to the app level
        # Don't prevent default or stop propagation
        pass


class AgentScreen(Screen):
    """
    The main screen for the phosphobot chat interface.
    This screen handles user input, displays chat logs, and manages agent interactions.
    """

    def compose(self) -> ComposeResult:
        """Create the UI layout and widgets."""
        yield RichLog(id="chat-log", wrap=True, highlight=True)
        # Suggestion box: sits above the input
        yield SuggestionBox(id="suggest-box")
        yield CustomInput(
            placeholder="Click here, type a prompt and press Enter to send",
            id="chat-input",
        )
        yield Footer()

    def on_key(self, event: Key) -> None:
        """Handle key presses at screen level for both robot keyboard control and suggestion cycling."""
        app = self.app
        if not isinstance(app, AgentApp):
            return

        # Keyboard control keys should work regardless of focus
        if app.current_agent.control_mode == "keyboard":
            # Movement keys
            if event.key == "up":
                app.action_keyboard_forward()
                event.prevent_default()
                return
            elif event.key == "down":
                app.action_keyboard_backward()
                event.prevent_default()
                return
            elif event.key == "left":
                app.action_keyboard_left()
                event.prevent_default()
                return
            elif event.key == "right":
                app.action_keyboard_right()
                event.prevent_default()
                return
            elif event.key == "d":
                app.action_keyboard_up()
                event.prevent_default()
                return
            elif event.key == "c":
                app.action_keyboard_down()
                event.prevent_default()
                return
            # Gripper toggle
            elif event.key == "space":
                app.action_keyboard_gripper()
                event.prevent_default()
                return

        # Toggle key always works
        if event.key == "ctrl+t":
            app.action_toggle_control_mode()
            event.prevent_default()
            return

        # Handle suggestion navigation and tab completion when input is focused
        input_widget = self.query_one(Input)
        focused = self.app.focused is input_widget
        suggestion_box = self.query_one(SuggestionBox)

        if focused:
            # Up/Down: cycle suggestions if present
            if event.key == "up":
                if suggestion_box.suggestions:
                    suggestion_box.cycle(-1)
                    event.prevent_default()
                    return
            elif event.key == "down":
                if suggestion_box.suggestions:
                    suggestion_box.cycle(1)
                    event.prevent_default()
                    return
            elif event.key == "tab" and suggestion_box.suggestions:
                # Let the CustomInput handle the tab completion
                event.prevent_default()
                event.stop()

    def _write_welcome_message(self) -> None:
        self._write_to_log(
            f"""🧪 Welcome to phosphobot chat!

{ascii_test_tube()}

[grey46]Access the phosphobot dashboard here: http://{get_local_ip()}:{config.PORT}

💡 Tip: Type /help for commands. When the agent is running, press Ctrl+T for keyboard control and Ctrl+S to stop the agent.[/grey46]
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

    # Input changed -> update suggestion box
    def on_input_changed(self, event: Input.Changed) -> None:
        """
        Called every time the input changes. We'll compute the top 3 suggestions
        based on the first token (the command prefix).
        """
        text = event.value or ""
        suggestion_box = self.query_one(SuggestionBox)

        # If the first token starts with "/", attempt to suggest commands
        first_token = text.split(" ", 1)[0]
        if first_token.startswith("/"):
            q = first_token
            # delegate ranking to App helper
            app = self.app
            if isinstance(app, AgentApp):
                suggestions = app.find_command_suggestions(q)
                suggestion_box.update_suggestions(suggestions)
            else:
                suggestion_box.update_suggestions([])
        else:
            # not a command prefix => clear suggestions
            suggestion_box.update_suggestions([])


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
    SuggestionBox {
        height: auto;
        max-height: 12;
    }
    #chat-input {
        dock: bottom;
        height: 5;
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

        self.current_agent = RoboticAgent(write_to_log=self._write_to_log)

    def _write_to_log(self, message: str, who: str = "agent") -> None:
        """Internal logging helper to write messages to the main screen's log."""
        screen = self._get_main_screen()
        if screen:
            screen._write_to_log(message, who)

    def _handle_prompt(self, prompt: str, screen: AgentScreen) -> None:
        """
        Handles submitted input. Supports direct command forms like:
          /dataset my_dataset
          /init
          /stop
          etc.
        If the prompt isn't a recognized command, it will be treated as a
        task description for the agent.
        """
        if self.is_agent_running:
            screen._write_to_log("An agent is already running.", "system")
            return None

        screen._write_to_log(prompt, "user")

        # Simple command parsing
        # /dataset <name>
        m = re.match(r"^/dataset(?:\s+(.+))?$", prompt, flags=re.IGNORECASE)
        if m:
            name = m.group(1)
            if name:
                self.current_agent.dataset_name = name.strip()
                screen._write_to_log(
                    f"Dataset name set to: [bold green]{self.current_agent.dataset_name}[/bold green]",
                    "system",
                )
            else:
                # no name provided: returns the current dataset name and explain usage
                screen._write_to_log(
                    f"Current dataset name: [bold green]{self.current_agent.dataset_name}[/bold green]. "
                    "To change it, use: /dataset <name> (e.g. /dataset my_dataset)",
                    "system",
                )
            return None

        # /init command
        if prompt.strip().lower() == "/init":
            screen._write_to_log("Moving robot to initial position", "system")
            asyncio.create_task(self.current_agent.phosphobot_client.move_init())
            return None

        # /stop command
        if prompt.strip().lower() == "/stop":
            self.action_stop_agent()
            return None

        # /help command
        if prompt.strip().lower() == "/help":
            table = Table(show_header=True, header_style="dim")
            table.add_column("Command", style="green", no_wrap=True)
            table.add_column("Description", style="green")
            table.add_column("Usage", style="dim")

            for cmd in COMMANDS:
                table.add_row(cmd["cmd"], cmd["desc"], cmd["usage"])

            screen._write_to_log(
                "[bold green]Available Commands:[/bold green]", "system"
            )
            log = screen.query_one(RichLog)
            log.write(table)
            return None

        # /quit command
        if prompt.strip().lower() == "/quit":
            screen._write_to_log("Quitting the application...", "system")
            self.exit()
            return None

        # /new command
        if prompt.strip().lower() == "/new":
            # Start a new chat session
            self.action_new_chat()
            return None

        # If it starts with a slash but isn't a recognized command, log an error
        if prompt.startswith("/"):
            screen._write_to_log(
                f"Unknown command: {prompt}. Type /help for available commands.",
                "system",
            )
            return None

        # Otherwise, treat input as a task for the agent
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
        # clear suggestions when submitting
        screen.query_one(SuggestionBox).update_suggestions([])
        self._handle_prompt(prompt, screen)

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
                f"[bold red]Error:[/bold red] [red]{error_message}[/red]", "system"
            )
        elif event_type == "step_done":
            log.write("")
            screen._write_to_log("Step status: [bold green][DONE][/]", "agent")
            log.write("")

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        """
        Generate system commands for the command palette (Ctrl+P menu).
        Note: we still provide the palette; inline autocomplete is the primary flow.
        """
        for function in [
            self.action_new_chat,
            self.action_stop_agent,
            self.action_toggle_control_mode,
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

    def action_stop_agent(self) -> None:
        """Stop the currently running agent."""
        screen = self._get_main_screen()
        if not screen:
            return

        if self.is_agent_running and self.worker:
            self.worker.cancel()
            screen._write_to_log("Interrupt requested. Stopping agent...", "system")
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

    def find_command_suggestions(self, prefix: str) -> List[Tuple[str, str]]:
        """
        Return suggestions as (cmd, desc) given a prefix like '/data'.
        """
        prefix_l = prefix.lower()
        starts = []
        contains = []
        for c in COMMANDS:
            cmd_l = c["cmd"].lower()
            if cmd_l.startswith(prefix_l):
                starts.append((c["cmd"], c["desc"]))
            elif prefix_l in cmd_l:
                contains.append((c["cmd"], c["desc"]))

        # Combine and return all matches (the SuggestionBox will handle limiting)
        results = starts + contains

        return results


if __name__ == "__main__":
    app = AgentApp()
    app.run()

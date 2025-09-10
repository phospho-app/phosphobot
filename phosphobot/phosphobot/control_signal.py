import threading
from typing import Literal
from uuid import uuid4

# Class to control the auto control thread
# Learn more about threading in Python:
# https://medium.com/@yashwanthnandam/understanding-thread-lock-and-thread-release-and-rlock-b95e1ceb4a17#:~:text=A%20Lock%20object%20in%20Python's,Lock%20until%20it%20is%20released.


class ControlSignal:
    def __init__(self) -> None:
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def is_in_loop(self) -> bool:
        with self._lock:
            return self._running


class AIControlSignal(ControlSignal):
    _status: Literal["stopped", "running", "paused", "waiting"]

    def __init__(self) -> None:
        super().__init__()
        self._status = "stopped"
        self.id = str(uuid4())
        self._is_in_loop = False

    def new_id(self) -> None:
        self.id = str(uuid4())

    def start(self) -> None:
        with self._lock:
            self._is_in_loop = True
            self._status = "waiting"

    def set_running(self) -> None:
        with self._lock:
            self._is_in_loop = True
            self._status = "running"

    def stop(self) -> None:
        with self._lock:
            self._is_in_loop = False
            self._status = "stopped"

    def is_in_loop(self) -> bool:
        with self._lock:
            return self._is_in_loop

    @property
    def status(self) -> Literal["stopped", "running", "paused", "waiting"]:
        return self._status

    @status.setter
    def status(self, value: Literal["stopped", "running", "paused", "waiting"]) -> None:
        if value == "stopped":
            self.stop()
        elif value == "running":
            self._status = value
            with self._lock:
                self._is_in_loop = True
        elif value == "paused":
            self._status = value
        elif value == "waiting":
            self._status = value
            with self._lock:
                self._is_in_loop = True

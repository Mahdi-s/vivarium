from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Protocol

from aam.types import ActionRequest


class Channel(Protocol):
    def submit(self, req: ActionRequest) -> None: ...

    def take_all(self) -> List[ActionRequest]: ...


@dataclass
class InMemoryChannel:
    _q: Deque[ActionRequest]

    def __init__(self) -> None:
        self._q = deque()

    def submit(self, req: ActionRequest) -> None:
        self._q.append(req)

    def take_all(self) -> List[ActionRequest]:
        out: List[ActionRequest] = list(self._q)
        self._q.clear()
        return out



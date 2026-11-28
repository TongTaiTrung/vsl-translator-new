"""Per-frame voting: a label must win N consecutive frames to be added to the word sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class VoteConfig:
    min_consecutive: int = 5
    max_words: int = 64
    early_votes: int | None = None

    def __post_init__(self) -> None:
        if self.min_consecutive < 1 or self.max_words < 1:
            raise ValueError("min_consecutive and max_words must be positive")
        if self.early_votes is not None and not 1 <= self.early_votes <= self.min_consecutive:
            raise ValueError("early_votes must be in [1, min_consecutive]")


class WordVoter:

    def __init__(self, config: VoteConfig | None = None) -> None:
        self.config = config or VoteConfig()
        self.sequence: list[str] = []
        self.candidate: str | None = None
        self.count = 0
        self.blocked: str | None = None
        self.strong = False

    def vote(self, label: str | None, *, strong: bool = False,
             confirm: Callable[[str, bool], bool] | None = None) -> str | None:
        if label is None:
            self.interrupt()
            return None
        if label == self.candidate:
            self.count += 1
            self.strong = self.strong and strong
        else:
            self.candidate, self.count, self.strong = label, 1, strong
        early = self.config.early_votes is not None and self.strong and self.count >= self.config.early_votes
        if label == self.blocked or (self.count < self.config.min_consecutive and not early):
            return None
        if confirm is not None and not confirm(label, self.count < self.config.min_consecutive):
            self.interrupt()
            return None
        self.sequence.append(label)
        del self.sequence[:-self.config.max_words]
        self.blocked = label
        self.interrupt()
        return label

    def interrupt(self) -> None:
        self.candidate, self.count, self.strong = None, 0, False

    def release(self) -> None:
        self.interrupt()
        self.blocked = None

    def undo(self) -> str | None:
        self.release()
        return self.sequence.pop() if self.sequence else None

    def clear(self) -> None:
        self.release()
        self.sequence.clear()

    def state(self) -> dict:
        return {
            "candidate": self.candidate,
            "count": self.count,
            "required": self.config.min_consecutive,
            "blocked": self.blocked,
            "strong": self.strong,
            "early_required": self.config.early_votes,
        }

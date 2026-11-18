"""Bỏ phiếu theo frame: một nhãn phải thắng N frame liên tiếp mới được thêm vào chuỗi từ."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoteConfig:
    min_consecutive: int = 5      # số frame dự đoán liên tiếp cùng một nhãn để chốt từ
    max_words: int = 64           # giới hạn độ dài chuỗi (bỏ từ cũ nhất khi vượt)

    def __post_init__(self) -> None:
        if self.min_consecutive < 1 or self.max_words < 1:
            raise ValueError("min_consecutive and max_words must be positive")


class WordVoter:
    """Đếm chuỗi dự đoán liên tiếp và chốt từ vào ``sequence``.

    - ``vote(label)``: một frame có dự đoán mới; ``None`` = bị từ chối, làm đứt chuỗi đếm.
    - ``interrupt()``: frame không có dự đoán (đang gom cửa sổ); đứt chuỗi đếm.
    - ``release()``: tay đã rời khung hình; đứt chuỗi đếm và cho phép lặp lại từ vừa chốt.

    Từ vừa chốt bị chặn cho tới khi tay rời khung hoặc một từ khác được chốt, nên giữ
    nguyên một ký hiệu sẽ không thêm cùng từ nhiều lần.
    """

    def __init__(self, config: VoteConfig | None = None) -> None:
        self.config = config or VoteConfig()
        self.sequence: list[str] = []
        self.candidate: str | None = None
        self.count = 0
        self.blocked: str | None = None

    def vote(self, label: str | None) -> str | None:
        """Ghi nhận một dự đoán; trả về nhãn nếu nó vừa được thêm vào chuỗi."""
        if label is None:
            self.interrupt()
            return None
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1
        if label == self.blocked or self.count < self.config.min_consecutive:
            return None
        self.sequence.append(label)
        del self.sequence[:-self.config.max_words]
        self.blocked = label
        self.candidate, self.count = None, 0
        return label

    def interrupt(self) -> None:
        self.candidate, self.count = None, 0

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
        }

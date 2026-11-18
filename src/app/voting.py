"""Bỏ phiếu theo frame: một nhãn phải thắng N frame liên tiếp mới được thêm vào chuỗi từ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class VoteConfig:
    min_consecutive: int = 5      # số frame dự đoán liên tiếp cùng một nhãn để chốt từ
    max_words: int = 64           # giới hạn độ dài chuỗi (bỏ từ cũ nhất khi vượt)
    early_votes: int | None = None   # chốt sớm khi cả chuỗi đều là phiếu "mạnh"; None = tắt

    def __post_init__(self) -> None:
        if self.min_consecutive < 1 or self.max_words < 1:
            raise ValueError("min_consecutive and max_words must be positive")
        if self.early_votes is not None and not 1 <= self.early_votes <= self.min_consecutive:
            raise ValueError("early_votes must be in [1, min_consecutive]")


class WordVoter:
    """Đếm chuỗi dự đoán liên tiếp và chốt từ vào ``sequence``.

    - ``vote(label)``: một frame có dự đoán mới; ``None`` = bị từ chối, làm đứt chuỗi đếm.
    - ``interrupt()``: frame không có dự đoán (đang gom cửa sổ); đứt chuỗi đếm.
    - ``release()``: tay đã rời khung hình; đứt chuỗi đếm và cho phép lặp lại từ vừa chốt.

    Từ vừa chốt bị chặn cho tới khi tay rời khung hoặc một từ khác được chốt, nên giữ
    nguyên một ký hiệu sẽ không thêm cùng từ nhiều lần.

    Nhánh nhanh: ``vote(label, strong=True)`` đánh dấu phiếu rất chắc chắn. Nếu cả chuỗi
    đều mạnh, chỉ cần ``early_votes`` phiếu. ``confirm(label, early)`` (nếu có) được gọi ngay
    trước khi chốt; trả về False thì chuỗi bị huỷ và từ không được thêm.
    """

    def __init__(self, config: VoteConfig | None = None) -> None:
        self.config = config or VoteConfig()
        self.sequence: list[str] = []
        self.candidate: str | None = None
        self.count = 0
        self.blocked: str | None = None
        self.strong = False       # mọi phiếu trong chuỗi hiện tại đều mạnh

    def vote(self, label: str | None, *, strong: bool = False,
             confirm: Callable[[str, bool], bool] | None = None) -> str | None:
        """Ghi nhận một dự đoán; trả về nhãn nếu nó vừa được thêm vào chuỗi."""
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

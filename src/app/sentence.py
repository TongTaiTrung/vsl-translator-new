"""Ghép chuỗi nhãn thô thành câu tiếng Việt bằng LLM trên Groq (mặc định Qwen 3.8 27B)."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Nghĩa của các nhãn thô không tự đọc được. Nhãn khác: "_" được đổi thành khoảng trắng.
DEFAULT_GLOSSARY: dict[str, str] = {
    "toi": "tôi",
    "ten_la": "tên là",
    "age": "tuổi",
}

SYSTEM_PROMPT_TEMPLATE = """\
Bạn là bộ hậu xử lý của hệ thống dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) sang tiếng Việt viết.

ĐẦU VÀO là chuỗi NHÃN THÔ do mô hình nhận dạng ký hiệu đưa ra, đúng thứ tự người dùng ký, ngăn cách bởi " | ".
Đặc điểm của nhãn thô:
- Có thể viết không dấu hoặc dùng "_" thay khoảng trắng. Bảng chú giải bên dưới cho biết nghĩa của các nhãn đã biết.
- Mục dạng "[đánh vần: T-R-U-N-G]" là các chữ cái đánh vần bằng ngón tay liên tiếp. Ghép chúng thành MỘT từ, \
thường là tên riêng hoặc từ chưa có ký hiệu riêng (ví dụ T-R-U-N-G → "Trung"). Viết hoa chữ đầu nếu là tên riêng. \
Chỉ thêm dấu thanh khi chắc chắn đó là tên/từ tiếng Việt phổ biến; nếu không, giữ nguyên các chữ cái.
- Ngữ pháp ký hiệu khác tiếng Việt nói: thường lược hư từ, trợ từ và từ "là", có thể đưa chủ đề lên trước, \
từ để hỏi thường đứng cuối.
- Mô hình nhận dạng đôi khi lặp nhãn; bỏ những lần lặp không có nghĩa.

NHIỆM VỤ: ghép chuỗi nhãn thành câu tiếng Việt tự nhiên, đúng chính tả, đủ dấu thanh và dấu câu.
QUY TẮC:
1. Giữ đúng ý và trình tự ý của người ký. Không bịa thêm thông tin, tên, số hay chi tiết không có trong nhãn.
2. Được thêm hư từ, trợ từ, từ nối cần thiết (là, của, ở, thì, ạ, và...) và sắp xếp lại cho đúng ngữ pháp.
3. Nếu chuỗi chưa thành câu trọn vẹn (người dùng đang ký dở), trả về cụm từ tự nhiên nhất, không tự thêm ý.
4. Chỉ trả về đúng kết quả trên một dòng: không giải thích, không ngoặc kép, không tiền tố kiểu "Câu:".

BẢNG CHÚ GIẢI NHÃN:
{glossary}

VÍ DỤ:
Nhãn: xin chào | toi | ten_la | [đánh vần: T-R-U-N-G]
Kết quả: Xin chào, tôi tên là Trung.
Nhãn: toi | địa chỉ
Kết quả: Địa chỉ của tôi
Nhãn: cảm ơn | cảm ơn
Kết quả: Cảm ơn.
"""

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)


def _is_letter(label: str) -> bool:
    return len(label) == 1 and label.isalpha()


def label_text(label: str, glossary: Mapping[str, str] = DEFAULT_GLOSSARY) -> str:
    return glossary.get(label, label.replace("_", " "))


def build_system_prompt(glossary: Mapping[str, str] = DEFAULT_GLOSSARY) -> str:
    lines = "\n".join(f'- "{raw}" = "{text}"' for raw, text in glossary.items()) or "- (không có)"
    return SYSTEM_PROMPT_TEMPLATE.format(glossary=lines)


def group_labels(words: Sequence[str]) -> list[str | list[str]]:
    """Gom các chữ cái đơn liên tiếp (đánh vần) thành một nhóm."""
    groups: list[str | list[str]] = []
    for word in words:
        if _is_letter(word):
            if groups and isinstance(groups[-1], list):
                groups[-1].append(word)
            else:
                groups.append([word])
        else:
            groups.append(word)
    return groups


def format_labels(words: Sequence[str]) -> str:
    return " | ".join(
        f"[đánh vần: {'-'.join(letter.upper() for letter in item)}]" if isinstance(item, list) else item
        for item in group_labels(words)
    )


def naive_sentence(words: Sequence[str], glossary: Mapping[str, str] = DEFAULT_GLOSSARY) -> str:
    """Câu dự phòng khi không gọi được LLM: dịch nhãn theo chú giải và ghép chữ cái."""
    parts = [
        "".join(item).capitalize() if isinstance(item, list) else label_text(item, glossary)
        for item in group_labels(words)
    ]
    text = " ".join(parts).strip()
    return text[:1].upper() + text[1:] if text else ""


def clean_completion(text: str) -> str:
    text = _THINK_RE.sub("", text or "").strip()
    text = re.sub(r"^(kết quả|câu)\s*:\s*", "", text, flags=re.IGNORECASE)
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return line.strip('"“”\'')


def load_api_key(env_var: str = "GROQ_API_KEY", env_file: str | Path | None = None) -> str | None:
    """Đọc key từ biến môi trường, sau đó từ file ``.env`` ở gốc dự án (không commit)."""
    key = os.environ.get(env_var)
    if key:
        return key.strip()
    path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip().removeprefix("export ").strip() == env_var:
                return value.strip().strip('"\'') or None
    return None


class GroqClient:
    """Client tối giản cho endpoint chat completions (tương thích OpenAI) của Groq."""

    def __init__(self, api_key: str | None = None, *, model: str = DEFAULT_MODEL, url: str = GROQ_URL,
                 timeout: float = 15.0, session: requests.Session | None = None) -> None:
        self.api_key = api_key or load_api_key()
        if not self.api_key:
            raise ValueError("Thiếu GROQ_API_KEY: đặt biến môi trường hoặc ghi vào file .env ở gốc dự án")
        self.model, self.url, self.timeout = model, url, timeout
        self.session = session or requests.Session()

    def chat(self, messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 200) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if "qwen3" in self.model:
            payload["reasoning_effort"] = "none"   # tắt chế độ suy nghĩ: nhanh hơn, không trả <think>
        response = self.session.post(
            self.url, json=payload, timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Groq API lỗi {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"] or ""


class SentenceComposer:
    """Gọi LLM trong luồng nền để vòng lặp camera không bị chặn.

    ``submit(words)`` chỉ giữ yêu cầu mới nhất; các chuỗi cũ chưa kịp gửi bị bỏ qua và
    kết quả của yêu cầu cũ không ghi đè kết quả mới hơn.
    """

    def __init__(self, client: GroqClient | None = None, *, glossary: Mapping[str, str] = DEFAULT_GLOSSARY,
                 on_update: Callable[[dict], None] | None = None, background: bool = True) -> None:
        self.client = client
        self.glossary = dict(glossary)
        self.system_prompt = build_system_prompt(self.glossary)
        self.on_update = on_update
        self.sentence = ""
        self.words: tuple[str, ...] = ()
        self.status = "idle"           # idle | pending | ok | fallback | error
        self.error: str | None = None
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._pending: tuple[int, tuple[str, ...]] | None = None
        self._version = 0
        self._shown = 0
        self._closed = False
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._run, name="sentence-composer", daemon=True)
            self._thread.start()

    def messages(self, words: Sequence[str]) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Nhãn: {format_labels(words)}\nKết quả:"},
        ]

    def compose(self, words: Iterable[str]) -> tuple[str, str, str | None]:
        """Ghép đồng bộ; trả về ``(câu, status, lỗi)``. Dùng câu dự phòng nếu LLM lỗi."""
        words = tuple(words)
        if not words:
            return "", "idle", None
        fallback = naive_sentence(words, self.glossary)
        if self.client is None:
            return fallback, "fallback", None
        try:
            sentence = clean_completion(self.client.chat(self.messages(words)))
        except Exception as exc:   # mạng/API lỗi: vẫn hiển thị câu thô
            return fallback, "error", str(exc)
        return (sentence, "ok", None) if sentence else (fallback, "fallback", None)

    def submit(self, words: Iterable[str]) -> None:
        words = tuple(words)
        with self._lock:
            self._version += 1
            version = self._version
            if not words:            # chuỗi rỗng: không cần gọi API
                self._pending = None
                self._publish(version, words, "", "idle", None)
                return
            self._pending = (version, words)
            self.status = "pending"
        if self._thread is None:
            self._process_pending()
        else:
            self._wake.set()

    def state(self) -> dict:
        with self._lock:
            return {"sentence": self.sentence, "words": list(self.words), "status": self.status, "error": self.error}

    def close(self) -> None:
        self._closed = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _publish(self, version: int, words, sentence: str, status: str, error: str | None) -> None:
        # gọi khi đang giữ _lock
        if version < self._shown:
            return
        self._shown = version
        self.sentence, self.words, self.error = sentence, tuple(words), error
        self.status = status if version == self._version else "pending"
        if self.on_update is not None:
            self.on_update({"sentence": sentence, "words": list(words), "status": status, "error": error})

    def _process_pending(self) -> None:
        with self._lock:
            job, self._pending = self._pending, None
        if job is None:
            return
        version, words = job
        sentence, status, error = self.compose(words)
        with self._lock:
            self._publish(version, words, sentence, status, error)

    def _run(self) -> None:
        while not self._closed:
            self._wake.wait()
            self._wake.clear()
            if self._closed:
                break
            self._process_pending()

"""Compose a Vietnamese sentence from raw sign labels with an LLM on Groq."""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
- Mục dạng "[đánh vần: ...]" là các chữ cái đánh vần bằng ngón tay liên tiếp, dùng cho tên riêng hoặc từ \
chưa có ký hiệu riêng (ví dụ T-R-U-N-G → "Trung").
- Trong mục đánh vần có thể có DẤU THANH dạng "(dấu sắc)", "(dấu huyền)", "(dấu hỏi)", "(dấu ngã)", "(dấu nặng)". \
Dấu thanh được ký NGAY SAU các chữ cái của âm tiết nó thuộc về: đặt dấu đó lên âm tiết đứng liền trước, đúng vị trí \
theo chính tả tiếng Việt (B-I-(dấu nặng) → "bị", T-O-A-N-(dấu sắc) → "toán"). Dấu thanh cũng là dấu hiệu kết thúc \
một âm tiết.
- Trong mục đánh vần còn có DẤU CHỮ dạng "(dấu â,ô,ê)" (dấu mũ), "(dấu ă)" (dấu trăng) và "(dấu ơ,ư)" (dấu móc). Dấu chữ biến nguyên âm \
đứng gần nhất phía trước nó thành chữ có dấu: "(dấu â,ô,ê)" đổi a→â, o→ô, e→ê; "(dấu ă)" đổi a→ă; "(dấu ơ,ư)" đổi o→ơ, u→ư \
(T-H-O-(dấu ơ,ư) → "Thơ", C-O-(dấu â,ô,ê)-N-G → "công", H-O-(dấu ơ,ư)-N → "hơn", A-(dấu ă)-(dấu hỏi) → "ẳ"). Dấu chữ là một phần của chữ cái, \
KHÔNG phải một từ và KHÔNG kết thúc âm tiết.
- MỘT mục đánh vần có thể chứa NHIỀU âm tiết/từ viết liền, không có ký hiệu ngăn cách. Hãy tách chuỗi chữ cái thành \
các âm tiết tiếng Việt hợp lệ sao cho ghép lại có nghĩa nhất (ví dụ M-I-N-H-B-I-(dấu nặng)-N-G-U → "Minh bị ngu"). \
Viết hoa chữ đầu của tên riêng; các từ thường viết thường.
- BẮT BUỘC giữ mọi dấu thanh đã được ký, kể cả khi bỏ dấu cũng ra một tên/từ có thật (H-O-A-(dấu huyền) là "Hòa", \
không phải "Hoa"). Âm tiết không có dấu thanh đi kèm thì giữ thanh ngang, trừ khi chắc chắn đó là tên/từ tiếng Việt \
phổ biến bị thiếu dấu.
- TUYỆT ĐỐI không viết nguyên văn tên dấu ("dấu ơ,ư", "dấu â,ô,ê", "dấu ă", "dấu hỏi", "dấu nặng"...) hay chữ cái gốc kèm tên dấu (KHÔNG BAO GIỜ ra "A dấu ă dấu hỏi") vào kết quả, kể cả khi nhãn dấu \
đứng riêng ngoài mục đánh vần: luôn đặt dấu đó lên chữ cái liền trước.
- Ngữ pháp ký hiệu khác tiếng Việt nói: thường lược hư từ, trợ từ và từ "là", có thể đưa chủ đề lên trước, \
từ để hỏi thường đứng cuối.
- Mô hình nhận dạng đôi khi lặp nhãn; bỏ những lần lặp không có nghĩa.

NHIỆM VỤ: ghép chuỗi nhãn thành câu tiếng Việt tự nhiên, đúng chính tả, đủ dấu thanh và dấu câu.
QUY TẮC:
1. Giữ đúng ý và trình tự ý của người ký. Không bịa thêm thông tin, tên, số hay chi tiết không có trong nhãn.
2. Được thêm hư từ, trợ từ, từ nối cần thiết (là, của, ở, thì, ạ, và...) và sắp xếp lại cho đúng ngữ pháp.
3. Nếu chuỗi chưa thành câu trọn vẹn (người dùng đang ký dở), trả về cụm từ tự nhiên nhất, không tự thêm ý.
4. Chỉ trả về đúng kết quả trên một dòng: không giải thích, không ngoặc kép, không tiền tố kiểu "Câu:".
5. Kết quả LUÔN kết thúc bằng dấu câu: dấu chấm "." (hoặc "?" nếu là câu hỏi, "!" nếu là câu cảm thán), kể cả khi chỉ là một cụm từ.

BẢNG CHÚ GIẢI NHÃN:
{glossary}

VÍ DỤ:
Nhãn: xin chào | toi | ten_la | [đánh vần: T-R-U-N-G]
Kết quả: Xin chào, tôi tên là Trung.
Nhãn: toi | địa chỉ
Kết quả: Địa chỉ của tôi.
Nhãn: cảm ơn | cảm ơn
Kết quả: Cảm ơn.
Nhãn: [đánh vần: M-I-N-H-B-I-(dấu nặng)-N-G-U]
Kết quả: Minh bị ngu.
Nhãn: xin chào | toi | ten_la | [đánh vần: T-H-O-(dấu ơ,ư)]
Kết quả: Xin chào, tôi tên là Thơ.
Nhãn: [đánh vần: A-(dấu ă)-(dấu hỏi)]
Kết quả: Ẳ.
Nhãn: [đánh vần: M-A-(dấu ă)-N-G-(dấu sắc)]
Kết quả: Mắng.
"""

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)


TONE_MARKS: dict[str, str] = {
    "dấu sắc": "\u0301",
    "dấu huyền": "\u0300",
    "dấu hỏi": "\u0309",
    "dấu ngã": "\u0303",
    "dấu nặng": "\u0323",
}
_VOWELS = set("aăâeêioôơuưy")

VOWEL_MARKS: dict[str, dict[str, str]] = {
    "dấu â,ô,ê": {"a": "â", "o": "ô", "e": "ê"},
    "dấu ă": {"a": "ă"},
    "dấu ơ,ư": {"o": "ơ", "u": "ư"},
}
_UO_RE = re.compile("u\u031bo(?!\u031b)")


MARK_ALIASES: dict[str, str] = {
    "dấu sắt": "dấu sắc",
}


def _mark_key(label: str) -> str:
    key = unicodedata.normalize("NFC", label).strip().lower()
    return MARK_ALIASES.get(key, key)


def _is_vowel_mark(label: str) -> bool:
    return _mark_key(label) in VOWEL_MARKS


def _apply_vowel_mark(text: str, label: str) -> str:
    mapping = {base: unicodedata.normalize("NFD", marked) for base, marked in VOWEL_MARKS[_mark_key(label)].items()}
    chars = unicodedata.normalize("NFD", text)
    for position in range(len(chars) - 1, -1, -1):
        if chars[position] in mapping and not (position + 1 < len(chars) and chars[position + 1] in "\u0302\u031b\u0306"):
            chars = chars[:position] + mapping[chars[position]] + chars[position + 1:]
            break
    return unicodedata.normalize("NFC", chars)


def _fix_uo(run: str) -> str:
    return unicodedata.normalize("NFC", _UO_RE.sub("u\u031bo\u031b", unicodedata.normalize("NFD", run)))


def _is_letter(label: str) -> bool:
    return len(label) == 1 and label.isalpha()


def _is_tone(label: str) -> bool:
    return _mark_key(label) in TONE_MARKS


def _is_mark(label: str) -> bool:
    return _is_tone(label) or _is_vowel_mark(label)


def _spell_token(label: str) -> str:
    if _is_mark(label):
        return f"({_mark_key(label)})"
    return label.upper()


_SYLLABLE_RE = re.compile(
    r"^(ngh|ng|gh|gi|kh|nh|ph|th|tr|ch|qu|[bcdđghklmnprstvx])?([aăâeêioôơuưy]{1,3})(ng|nh|ch|[cmnpt])?$")
_MAX_SYLLABLE = 7


def _split_syllables(run: str) -> list[str]:
    best: list[list[str] | None] = [[]] + [None] * len(run)
    for end in range(1, len(run) + 1):
        for start in range(end - 1, max(0, end - _MAX_SYLLABLE) - 1, -1):
            prev = best[start]
            if prev is not None and _SYLLABLE_RE.match(run[start:end]):
                if best[end] is None or len(prev) + 1 < len(best[end]):
                    best[end] = prev + [run[start:end]]
    return best[-1] or [run]


def _apply_tone(syllable: str, mark: str) -> str:
    match = _SYLLABLE_RE.match(syllable)
    if not match:
        idx = max((position for position, char in enumerate(syllable) if char in _VOWELS), default=-1)
        return unicodedata.normalize("NFC", syllable[:idx + 1] + mark + syllable[idx + 1:]) if idx >= 0 else syllable
    onset, vowels, coda = (group_text or "" for group_text in match.groups())
    special = [position for position, char in enumerate(vowels) if char in "ăâêôơư"]
    if special:
        pos = special[-1]
    elif coda:
        pos = len(vowels) - 1
    else:
        pos = 1 if len(vowels) == 3 else 0
    vowels = unicodedata.normalize("NFC", vowels[:pos + 1] + mark + vowels[pos + 1:])
    return onset + vowels + coda


def _spelled_text(items: Sequence[str]) -> str:
    words: list[str] = []
    run = ""
    for item in items:
        if _is_vowel_mark(item):
            if run:
                run = _apply_vowel_mark(run, item)
            elif words:
                words[-1] = _apply_vowel_mark(words[-1], item)
            continue
        if not _is_tone(item):
            run += item.lower()
            continue
        if run:
            syllables = _split_syllables(_fix_uo(run))
            syllables[-1] = _apply_tone(syllables[-1], TONE_MARKS[_mark_key(item)])
            words += syllables
            run = ""
    if run:
        words += _split_syllables(_fix_uo(run))
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def label_text(label: str, glossary: Mapping[str, str] = DEFAULT_GLOSSARY) -> str:
    return glossary.get(label, label.replace("_", " "))


def build_system_prompt(glossary: Mapping[str, str] = DEFAULT_GLOSSARY) -> str:
    lines = "\n".join(f'- "{raw}" = "{text}"' for raw, text in glossary.items()) or "- (không có)"
    return SYSTEM_PROMPT_TEMPLATE.format(glossary=lines)


def group_labels(words: Sequence[str]) -> list[str | list[str]]:
    groups: list[str | list[str]] = []
    for word in words:
        if _is_letter(word) or (_is_mark(word) and groups and isinstance(groups[-1], list)):
            if groups and isinstance(groups[-1], list):
                groups[-1].append(word)
            else:
                groups.append([word])
        else:
            groups.append(word)
    return groups


def format_labels(words: Sequence[str]) -> str:
    return " | ".join(
        f"[đánh vần: {'-'.join(_spell_token(letter) for letter in item)}]" if isinstance(item, list) else item
        for item in group_labels(words)
    )


def naive_sentence(words: Sequence[str], glossary: Mapping[str, str] = DEFAULT_GLOSSARY) -> str:
    parts = [
        _spelled_text(item) if isinstance(item, list) else label_text(item, glossary)
        for item in group_labels(words)
    ]
    text = " ".join(parts).strip()
    return text[:1].upper() + text[1:] if text else ""


def clean_completion(text: str) -> str:
    text = _THINK_RE.sub("", text or "").strip()
    text = re.sub(r"^(kết quả|câu)\s*:\s*", "", text, flags=re.IGNORECASE)
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    line = line.strip('"“”\'')
    return line + "." if line and line[-1] not in ".?!…" else line


def load_api_key(env_var: str = "GROQ_API_KEY", env_file: str | Path | None = None) -> str | None:
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
            payload["reasoning_effort"] = "none"
        response = self.session.post(
            self.url, json=payload, timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Groq API lỗi {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"] or ""


class SentenceComposer:

    def __init__(self, client: GroqClient | None = None, *, glossary: Mapping[str, str] = DEFAULT_GLOSSARY,
                 on_update: Callable[[dict], None] | None = None, background: bool = True) -> None:
        self.client = client
        self.glossary = dict(glossary)
        self.system_prompt = build_system_prompt(self.glossary)
        self.on_update = on_update
        self.sentence = ""
        self.words: tuple[str, ...] = ()
        self.status = "idle"
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
        words = tuple(words)
        if not words:
            return "", "idle", None
        fallback = naive_sentence(words, self.glossary)
        if self.client is None:
            return fallback, "fallback", None
        try:
            sentence = clean_completion(self.client.chat(self.messages(words)))
        except Exception as exc:
            return fallback, "error", str(exc)
        return (sentence, "ok", None) if sentence else (fallback, "fallback", None)

    def submit(self, words: Iterable[str]) -> None:
        words = tuple(words)
        with self._lock:
            self._version += 1
            version = self._version
            if not words:
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

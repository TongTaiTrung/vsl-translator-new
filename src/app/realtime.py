"""Dịch VSL liên tục từ webcam: dự đoán theo frame → bỏ phiếu → chuỗi từ → câu (LLM).

Chạy: ``.venv/bin/python -m src.app [--checkpoint runs/.../ensemble.pt] [--no-llm] [--fast]``

``--fast``: nhánh nhanh (``src/app/fast.py``) — sàng lọc bằng một model, chốt sớm khi rất chắc
chắn và xác nhận bằng cả ensemble trước khi thêm từ.
"""

from __future__ import annotations

import argparse
import dataclasses
import textwrap
from pathlib import Path

import cv2
import numpy as np

from src.app.sentence import DEFAULT_MODEL, GroqClient, SentenceComposer
from src.app.voting import VoteConfig, WordVoter
from src.features.camera import open_camera, parse_resolution
from src.model.predict import StreamConfig, VSLPredictor


class RealtimeTranslator:
    """Ghép ``VSLStreamSession`` + ``WordVoter`` + ``SentenceComposer`` cho một camera/client.

    Mỗi frame có dự đoán mới được tính là một phiếu. Khi một nhãn thắng
    ``VoteConfig.min_consecutive`` phiếu liên tiếp, nó được thêm vào chuỗi từ; chuỗi đổi
    thì composer được gọi (trong luồng nền) để viết lại câu.
    """

    def __init__(self, predictor: VSLPredictor, *, stream_config: StreamConfig | None = None,
                 vote_config: VoteConfig | None = None, composer: SentenceComposer | None = None,
                 reset_after_commit: bool = True, session=None) -> None:
        stream_config = stream_config or StreamConfig(predict_every=1)
        self.session = session if session is not None else predictor.open_stream(stream_config)
        self.voter = WordVoter(vote_config)
        self.composer = composer if composer is not None else SentenceComposer(None)
        self.reset_after_commit = reset_after_commit
        self._last_state: dict | None = None
        self._last_prediction: dict | None = None
        self._snapshot: dict | None = None

    @property
    def words(self) -> list[str]:
        return list(self.voter.sequence)

    def process(self, frame_bgr: np.ndarray, *, now: float | None = None) -> dict:
        state = self.session.update_frame(frame_bgr, now=now)
        if state is self._last_state and self._snapshot is not None:   # frame bị bỏ vì target_fps
            return {**self._snapshot, "committed": None, "sentence": self.composer.state()}
        self._last_state = state

        committed = None
        prediction = self.session.prediction
        if state["status"] == "idle":
            self.voter.release()
        elif state["status"] == "collecting":
            self.voter.interrupt()
        elif prediction is not None and prediction is not self._last_prediction:
            committed = self._vote(prediction)
        self._last_prediction = prediction

        if committed is not None:
            self.composer.submit(self.voter.sequence)
            if self.reset_after_commit:   # cửa sổ mới cho ký hiệu kế tiếp
                self.session.reset()
                self._last_prediction = None
        self._snapshot = {
            **state,
            "vote": self.voter.state(),
            "words": self.words,
            "committed": committed,
        }
        return {**self._snapshot, "sentence": self.composer.state()}

    def _vote(self, prediction: dict) -> str | None:
        """Một dự đoán mới = một phiếu; trả về nhãn nếu vừa được chốt."""
        return self.voter.vote(prediction["label"])

    def undo(self) -> str | None:
        removed = self.voter.undo()
        if removed is not None:
            self.composer.submit(self.voter.sequence)
        return removed

    def clear(self) -> None:
        self.voter.clear()
        self.session.reset()
        self._last_prediction = None
        self.composer.submit(())

    def close(self) -> None:
        self.composer.close()
        self.session.close()

    def __enter__(self) -> RealtimeTranslator:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------- hiển thị OpenCV

FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/SFNS.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
)
STATUS_TEXT = {"idle": "Chưa thấy tay", "collecting": "Đang ghi nhận", "predicting": "Đang dự đoán",
               "ready": "Đang dự đoán"}
SENTENCE_STATUS = {"idle": "", "pending": " (đang ghép...)", "ok": "", "fallback": " (thô)",
                   "error": " (lỗi API, câu thô)"}


class Renderer:
    def __init__(self, size: int = 640, mirror_preview: bool = True) -> None:
        from PIL import ImageFont
        self.size, self.mirror_preview = size, mirror_preview
        font_path = next((p for p in FONT_CANDIDATES if p.is_file()), None)
        load = (lambda px: ImageFont.truetype(str(font_path), px)) if font_path else \
            (lambda _px: ImageFont.load_default())
        self.font, self.small, self.big = load(22), load(18), load(26)

    def __call__(self, session, result: dict) -> np.ndarray:
        from PIL import Image, ImageDraw
        from src.visualization.landmarks import draw

        size = self.size
        frame = session.last_frame if session.last_frame is not None else np.zeros((size, size, 3), np.uint8)
        preview = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        if session.last_pose_result is not None:
            draw(preview, session.last_pose_result, point_color=(80, 190, 255), line_color=(110, 170, 200))
        if session.last_hand_result is not None:
            draw(preview, session.last_hand_result, point_color=(60, 230, 120), line_color=(200, 255, 220),
                 secondary_color=(255, 190, 80))
        if self.mirror_preview:
            preview = cv2.flip(preview, 1)

        panel = 230
        canvas = np.full((size + panel, size, 3), (32, 32, 32), dtype=np.uint8)
        canvas[:size] = preview
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        painter = ImageDraw.Draw(image)

        prediction, vote = result["prediction"], result["vote"]
        if prediction and prediction["label"]:
            title = f'{prediction["label"]} {prediction["confidence"]:.0%}'
            if prediction.get("strong"):
                title += " ⚡"
        elif prediction:
            title = f'? ({prediction["rejection_reason"]})'
        else:
            title = "—"
        required = vote["early_required"] if vote.get("strong") and vote.get("early_required") \
            else vote["required"]
        bar = "■" * vote["count"] + "□" * max(required - vote["count"], 0)
        sentence = result["sentence"]
        lines = [
            (f'Dự đoán: {title}   {bar}', self.font, (110, 230, 160) if vote["count"] else (240, 240, 240)),
            (f'{STATUS_TEXT[result["status"]]}  |  {result["duration"]:.1f}s  |  '
             f'R: xóa  ⌫: bỏ từ cuối  Q: thoát', self.small, (145, 205, 250)),
            (f'Từ: {" · ".join(result["words"][-10:]) or "—"}', self.font, (250, 220, 145)),
        ]
        y = size + 10
        for text, font, color in lines:
            painter.text((16, y), self._fit(painter, text, font), font=font, fill=color)
            y += 36
        text = (sentence["sentence"] or "—") + SENTENCE_STATUS.get(sentence["status"], "")
        for line in textwrap.wrap(text, width=48)[:2]:
            painter.text((16, y), self._fit(painter, line, self.big), font=self.big, fill=(255, 255, 255))
            y += 36
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _fit(self, painter, text: str, font) -> str:
        width = self.size - 32
        if painter.textlength(text, font=font) <= width:
            return text
        while text and painter.textlength(text + "…", font=font) > width:
            text = text[:-1]
        return text + "…"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dịch VSL realtime: vote theo frame + ghép câu bằng LLM (Groq)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="mặc định: runs/*/ensemble.pt mới nhất")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--resolution", type=parse_resolution, default=None,
                        help="RỘNGxCAO cố định (vd. 1920x1080); mặc định: độ phân giải 16:9 lớn nhất camera hỗ trợ")
    parser.add_argument("--mirror-data", action="store_true", help="phải giống MIRROR_DATA lúc thu dữ liệu")
    parser.add_argument("--no-mirror-preview", action="store_true")
    parser.add_argument("--min-consecutive", type=int, default=5, help="số frame liên tiếp để chốt một từ")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--min-gap", type=float, default=0.08)
    parser.add_argument("--keep-window", action="store_true", help="không reset cửa sổ sau khi chốt từ")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model Groq")
    parser.add_argument("--no-llm", action="store_true", help="không gọi Groq, chỉ ghép nhãn thô")
    parser.add_argument("--display-size", type=int, default=640)
    fast = parser.add_argument_group("nhánh nhanh (--fast)")
    fast.add_argument("--fast", action="store_true",
                      help="sàng lọc bằng 1 model, chốt sớm khi chắc chắn, ensemble xác nhận trước khi chốt")
    fast.add_argument("--fast-model", type=int, default=-1, help="chỉ số model sàng lọc (-1: model toàn bộ dữ liệu)")
    fast.add_argument("--early-window", type=float, default=1.0, help="giây; bắt đầu dự đoán sớm")
    fast.add_argument("--early-votes", type=int, default=3, help="số phiếu mạnh liên tiếp để chốt sớm")
    fast.add_argument("--strong-confidence", type=float, default=0.9)
    fast.add_argument("--strong-gap", type=float, default=0.5)
    fast.add_argument("--escalate", action="store_true", help="chạy cả ensemble mỗi khi model nhanh từ chối")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    predictor = VSLPredictor(args.checkpoint, min_confidence=args.min_confidence,
                             min_probability_gap=args.min_gap)
    print(f"Model: {predictor.checkpoint_path} | {len(predictor.classes)} lớp: {', '.join(predictor.classes)}")

    client = None
    if not args.no_llm:
        try:
            client = GroqClient(model=args.model)
            print(f"LLM: Groq {args.model}")
        except ValueError as exc:
            print(f"{exc} → chạy không có LLM")
    composer = SentenceComposer(
        client, on_update=lambda s: print(f'Câu [{s["status"]}]: {s["sentence"]}  ←  {" | ".join(s["words"])}'),
    )
    stream_config = StreamConfig(predict_every=1, mirror_data=args.mirror_data)
    renderer = Renderer(args.display_size, mirror_preview=not args.no_mirror_preview)

    try:
        cap = open_camera(args.camera, args.resolution)
    except RuntimeError:
        composer.close()
        raise
    window = "VSL realtime translator"
    translator = None
    try:
        vote_config = VoteConfig(min_consecutive=args.min_consecutive)
        if args.fast:
            from src.app.fast import FastConfig, FastRealtimeTranslator   # fast.py import realtime.py
            fast_config = FastConfig(
                fast_model=args.fast_model, early_window_seconds=args.early_window,
                early_votes=min(args.early_votes, args.min_consecutive),
                strong_confidence=args.strong_confidence, strong_gap=args.strong_gap, escalate=args.escalate,
            )
            translator = FastRealtimeTranslator(
                predictor, stream_config=stream_config, fast_config=fast_config, composer=composer,
                vote_config=vote_config, reset_after_commit=not args.keep_window,
            )
            print(f"Nhánh nhanh: {fast_config}")
        else:
            translator = RealtimeTranslator(
                predictor, stream_config=stream_config, composer=composer,
                vote_config=vote_config, reset_after_commit=not args.keep_window,
            )
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera đã ngừng cung cấp hình ảnh.")
                break
            result = translator.process(frame)
            if result["committed"]:
                print(f'+ {result["committed"]}')
            cv2.imshow(window, renderer(translator.session, result))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                translator.clear()
            elif key in (8, 127):
                translator.undo()
    finally:
        cap.release()
        if translator is not None:
            final = translator.composer.state()
            translator.close()
            print("Kết quả:", final["sentence"] or "(trống)")
        else:
            composer.close()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()

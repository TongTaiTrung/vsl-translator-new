"""Nhánh dự đoán nhanh: sàng lọc bằng một model, xác nhận bằng cả ensemble trước khi chốt từ.

Nhánh thường chạy cả ensemble (5 fold + 1 full) ở MỌI frame và chỉ bắt đầu dự đoán khi cửa
sổ đủ ``min_window_seconds`` (1.5 s). Nhánh nhanh:

1. **Cascade**: mỗi frame chỉ chạy một model (mặc định model toàn bộ dữ liệu, ~1/6 chi phí);
   cả ensemble chỉ chạy ở bước xác nhận. ``escalate=True`` chạy thêm ensemble mỗi khi model
   nhanh từ chối (sau ``min_window_seconds``); mô phỏng held-out cho thấy không cải thiện
   WER mà tốn gấp ~3 lần số lần gọi ensemble, nên mặc định tắt.
2. **Thoát sớm có điều kiện**: từ ``early_window_seconds`` (1.0 s) đã dự đoán, nhưng trước
   ``min_window_seconds`` chỉ những phiếu "mạnh" (confidence >= ``strong_confidence`` và
   gap >= ``strong_gap``) mới được tính. Chuỗi toàn phiếu mạnh chốt sau ``early_votes`` phiếu
   thay vì ``min_consecutive``.
3. **Xác nhận**: ngay trước khi chốt, cả ensemble chạy lại trên đúng cửa sổ đó và phải đồng ý
   nhãn. Chốt sớm (ít phiếu hoặc cửa sổ ngắn) đòi ensemble cũng đạt ngưỡng chặt
   (``strict_confidence`` / ``strict_gap``).

Mọi dự đoán thêm ``source`` (fast | ensemble), ``strong`` và ``early`` vào dict kết quả.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.app.realtime import RealtimeTranslator
from src.app.sentence import SentenceComposer
from src.app.voting import VoteConfig
from src.features.extract import process_sequence
from src.model import training
from src.model.predict import StreamConfig, VSLPredictor, VSLStreamSession


@dataclass(frozen=True)
class FastConfig:
    fast_model: int = -1                 # model sàng lọc trong ensemble; -1 = model toàn bộ dữ liệu (cuối cùng)
    early_window_seconds: float = 1.0    # bắt đầu dự đoán sớm (chỉ phiếu mạnh được tính)
    strong_confidence: float = 0.9       # phiếu mạnh của model nhanh
    strong_gap: float = 0.5
    early_votes: int = 3                 # số phiếu mạnh liên tiếp để chốt sớm
    escalate: bool = False               # model nhanh từ chối (sau min_window) -> chạy cả ensemble
    confirm: bool = True                 # ensemble phải đồng ý trước khi chốt từ
    strict_confidence: float = 0.85      # ngưỡng ensemble khi xác nhận một lần chốt sớm
    strict_gap: float = 0.4

    def __post_init__(self) -> None:
        if self.early_window_seconds <= 0:
            raise ValueError("early_window_seconds must be positive")
        for name in ("strong_confidence", "strong_gap", "strict_confidence", "strict_gap"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.early_votes < 1:
            raise ValueError("early_votes must be positive")


def make_fast_ensemble(predictor: VSLPredictor, index: int = -1) -> training.VSLEnsemble:
    """Ensemble con gồm đúng một model của checkpoint (dùng chung trọng số, không copy)."""
    ensemble = predictor.ensemble
    n = len(ensemble.models)
    if not -n <= index < n:
        raise IndexError(f"fast_model={index} ngoài phạm vi ensemble {n} model")
    index %= n
    device = ensemble.means.device
    fast = training.VSLEnsemble(
        [ensemble.models[index]],
        ensemble.means[index:index + 1].cpu().numpy(), ensemble.stds[index:index + 1].cpu().numpy(),
        float(ensemble.temperature), presence_aware=ensemble.presence_aware,
    )
    return fast.to(device).eval()


class FastStreamSession(VSLStreamSession):
    """``VSLStreamSession`` dùng cascade + thoát sớm. Một session cho mỗi camera/client.

    ``stats`` đếm số lần chạy model nhanh / ensemble và số lần xác nhận bị từ chối.
    """

    def __init__(self, predictor: VSLPredictor, config: StreamConfig | None = None,
                 fast_config: FastConfig | None = None, landmarkers=None,
                 fast_ensemble: training.VSLEnsemble | None = None) -> None:
        self.fast_config = fast_config or FastConfig()
        config = config or StreamConfig(predict_every=1)
        if self.fast_config.early_window_seconds > config.min_window_seconds:
            raise ValueError("early_window_seconds must be <= StreamConfig.min_window_seconds")
        self.fast_ensemble = fast_ensemble if fast_ensemble is not None \
            else make_fast_ensemble(predictor, self.fast_config.fast_model)
        self.stats = {"fast": 0, "ensemble": 0, "confirm": 0, "confirm_rejected": 0}
        super().__init__(predictor, config, landmarkers)

    def reset(self) -> None:
        super().reset()
        self.sequence: np.ndarray | None = None     # cửa sổ đã xử lý của dự đoán gần nhất
        self.confirmation: dict | None = None

    def _min_ready_seconds(self) -> float:
        return self.fast_config.early_window_seconds

    def _predict(self, raw: np.ndarray, duration: float) -> dict:
        cfg, top_k = self.fast_config, self.config.top_k
        self.sequence = process_sequence(raw, target_len=self.predictor.target_len)
        result = self.predictor.predict_processed(self.sequence, top_k=top_k, ensemble=self.fast_ensemble)
        self.stats["fast"] += 1
        source = "fast"
        strong = bool(result["accepted"] and result["confidence"] >= cfg.strong_confidence
                      and result["probability_gap"] >= cfg.strong_gap)
        early = duration < self.config.min_window_seconds
        if early and not strong:          # cửa sổ còn ngắn: chỉ tin phiếu mạnh
            result = {**result, "label": None, "accepted": False, "rejection_reason": "early_uncertain"}
        elif not result["accepted"] and cfg.escalate:
            result = self.predictor.predict_processed(self.sequence, top_k=top_k)
            self.stats["ensemble"] += 1
            source = "ensemble"
        return {**result, "source": source, "strong": strong, "early": early}

    def confirm(self, label: str, early: bool = False) -> bool:
        """Chạy cả ensemble trên cửa sổ của dự đoán gần nhất; True nếu nó đồng ý ``label``.

        early: lần chốt dùng ít phiếu hơn ``min_consecutive`` -> đòi ngưỡng chặt.
        """
        if self.sequence is None:
            return False
        cfg = self.fast_config
        result = self.predictor.predict_processed(self.sequence, top_k=self.config.top_k)
        self.stats["ensemble"] += 1
        self.stats["confirm"] += 1
        strict = early or bool(self.prediction and self.prediction.get("early"))
        ok = result["label"] == label and (not strict or (
            result["confidence"] >= cfg.strict_confidence and result["probability_gap"] >= cfg.strict_gap))
        self.confirmation = {**result, "confirmed": ok, "strict": strict}
        if not ok:
            self.stats["confirm_rejected"] += 1
        return ok


class FastRealtimeTranslator(RealtimeTranslator):
    """``RealtimeTranslator`` dùng ``FastStreamSession``: phiếu mạnh chốt sớm, ensemble xác nhận."""

    def __init__(self, predictor: VSLPredictor, *, stream_config: StreamConfig | None = None,
                 fast_config: FastConfig | None = None, vote_config: VoteConfig | None = None,
                 composer: SentenceComposer | None = None, reset_after_commit: bool = True,
                 session=None) -> None:
        self.fast_config = fast_config or FastConfig()
        vote_config = vote_config or VoteConfig()
        if vote_config.early_votes is None:
            vote_config = dataclasses.replace(
                vote_config, early_votes=min(self.fast_config.early_votes, vote_config.min_consecutive))
        if session is None:
            session = FastStreamSession(predictor, stream_config or StreamConfig(predict_every=1),
                                        self.fast_config)
        super().__init__(predictor, vote_config=vote_config, composer=composer,
                         reset_after_commit=reset_after_commit, session=session)

    def _vote(self, prediction: dict) -> str | None:
        confirm = self.session.confirm if self.fast_config.confirm else None
        return self.voter.vote(prediction["label"], strong=bool(prediction.get("strong")), confirm=confirm)

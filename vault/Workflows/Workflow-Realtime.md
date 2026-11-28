# Dịch realtime bằng webcam

```mermaid
sequenceDiagram
    participant C as Camera
    participant R as RealtimeTranslator
    participant S as VSLStreamSession
    participant V as WordVoter
    participant L as SentenceComposer
    C->>R: frame BGR
    R->>S: update_frame
    S-->>R: dự đoán (accepted / rejected)
    R->>V: vote(label | None)
    V-->>R: từ chốt
    R->>L: chuỗi từ
    L-->>R: câu tiếng Việt
```

Lệnh: `.venv/bin/python -m src.app [--checkpoint runs/.../ensemble.pt] [--no-llm] [--fast]` ([[Module-app]]).

- Dự đoán chỉ bắt đầu khi cửa sổ đủ `min_window_seconds`; phiếu bị từ chối làm đứt chuỗi đếm.
- `--fast`: sàng lọc bằng model toàn bộ dữ liệu, chốt sớm khi rất chắc và xác nhận bằng ensemble.
- Cần `GROQ_API_KEY` (môi trường hoặc `.env`) để ghép câu; thiếu thì dùng `--no-llm`/ghép thô.
- Notebook tương đương: `realtime_predict.ipynb`.

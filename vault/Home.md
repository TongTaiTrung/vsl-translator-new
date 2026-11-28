# VSL Translator

Dịch Ngôn ngữ ký hiệu Việt Nam (VSL) từ video/webcam sang nhãn từ, rồi ghép thành câu tiếng Việt.

Luồng chính: video BGR → pad 1280×1280 → MediaPipe pose + hand → **383 đặc trưng/frame** → `process_sequence` → chuỗi **(30, 434)** → ensemble phân loại có từ chối → bỏ phiếu → câu (LLM Groq, tuỳ chọn).

## Bản đồ ghi chú

| Chủ đề | Ghi chú |
|---|---|
| Kiến trúc tổng thể | [[Architecture]] |
| Định dạng đặc trưng | [[Feature-Contract]] |
| Dữ liệu trên đĩa, chống rò rỉ | [[Data-Layout]] |
| Cây thư mục | [[Directory-Structure]] |
| Cài đặt, lệnh, test | [[Setup-and-Development]] |
| Thư viện và cấu hình | [[Dependencies-and-Config]] |
| Công cụ đánh giá/thí nghiệm | [[Experiment-Tools]] |

### Module
[[Module-features]] · [[Module-data]] · [[Module-model]] · [[Module-app]] · [[Module-mediapipe-builtin]] · [[Module-visualization]] · [[Module-web]]

### Quy trình
[[Workflow-Data-to-Model]] · [[Workflow-Realtime]] · [[Workflow-Web-API]]

## Quy ước
- Comment/docstring/thông báo CLI viết bằng tiếng Việt.
- Số liệu trong vault lấy từ code (`src/features/extract.py`); đổi extractor thì cập nhật [[Feature-Contract]].
- Không có `# comment` trong code: lý do thiết kế nằm ở docstring và ở các ghi chú này.

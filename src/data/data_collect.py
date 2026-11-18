"""Record square webcam clips and their processed VSL feature sequences.

Run from the project root with ``python -m src.data.data_collect -c xin_chao``.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.builtin.mediapipe_builtin import LandmarkerLoader, VisionRunningMode
from src.features.extract import extract, process_sequence
from src.features.frame import frame_padding


def get_next_id(raw_dir: Path, class_name: str, extracted_dir: Path | None = None) -> int:
    """Find the next unused ID across raw videos and extracted sequences."""
    existing_ids = []
    directories = [(raw_dir / class_name, "*.mp4")]
    if extracted_dir is not None:
        directories.append((extracted_dir, "*.npz"))
    for directory, pattern in directories:
        if directory.exists():
            existing_ids.extend(int(path.stem) for path in directory.glob(pattern) if path.stem.isdigit())
    return max(existing_ids, default=-1) + 1


def save_sample(frames, sample_id, raw_dir, extracted_dir, fps, landmarkers):
    if not frames:
        return None

    frames = [frame_padding(frame) for frame in frames]
    raw_features = []
    for frame in frames:
        pose_result, hand_result, _ = landmarkers.detect(frame)
        raw_features.append(extract(pose_result, hand_result))
    sequence = process_sequence(raw_features, target_len=30)

    height, width = frames[0].shape[:2]
    video_path = raw_dir / f"{sample_id}.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot open video writer: {video_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    try:
        np.savez_compressed(extracted_dir / f"{sample_id}.npz", X=sequence)
    except BaseException:
        video_path.unlink(missing_ok=True)
        raise
    return sequence.shape


def main():
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu VSL")
    parser.add_argument("-c", "--class_name", required=True, help="Tên từ vựng")
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    parser.add_argument("-g", "--group", type=int, default=0)
    parser.add_argument("--root", default="root_project", help="Thư mục gốc lưu dữ liệu")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.num_samples < 1 or args.fps < 1:
        parser.error("num_samples and fps must be positive")

    root = Path(args.root)
    raw_dir = root / "raw" / args.class_name
    extracted_dir = root / "extracted" / args.class_name / str(args.group)
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    landmarkers = None
    try:
        landmarkers = LandmarkerLoader(running_mode=VisionRunningMode.IMAGE)
        next_id = get_next_id(root / "raw", args.class_name, extracted_dir)
        collected = 0
        recording = False
        frames = []

        print(f"Thu thập {args.class_name}: SPACE bắt đầu/kết thúc, Q hoặc ESC thoát")
        while collected < args.num_samples:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot read frame from camera")
            frame = frame_padding(cv2.flip(frame, 1))
            display = frame.copy()

            if recording:
                frames.append(frame.copy())
                cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display, f"REC: {len(frames)} frames", (50, 37),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, f"{args.class_name} | group {args.group}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"{collected}/{args.num_samples} | SPACE to record", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Data Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:
                if not recording:
                    recording, frames = True, []
                    print(f"Đang quay mẫu {next_id}...")
                else:
                    recording = False
                    shape = save_sample(frames, next_id, raw_dir, extracted_dir, args.fps, landmarkers)
                    if shape is None:
                        print("Mẫu rỗng, đã bỏ qua")
                    else:
                        print(f"Đã lưu mẫu {next_id}: {len(frames)} frames, tensor {shape}")
                        collected += 1
                        next_id += 1
                    frames = []
    finally:
        cap.release()
        if landmarkers is not None:
            landmarkers.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

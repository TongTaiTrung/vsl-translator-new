import numpy as np
import mediapipe as mp
import cv2
from .defs import *
from .stream_process import StreamProcess
from src.features.frame import frame_padding
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, perf_counter_ns as _clock, sleep

class LandmarkerLoader:
    class _TimeStampInitializer:
        __slots__ = ("timestamp", "_offset_ns")
        def __init__(self):
            self._offset_ns = _clock() - 500_000
            self.timestamp = -1
        def get2update(self):
            current = (_clock() - self._offset_ns) // 1_000_000
            ts = self.timestamp
            self.timestamp = current if current > ts else ts + 1
            return self.timestamp
        def reset(self):
            self._offset_ns = _clock() - 500_000
            self.timestamp = -1

    def __init__(
        self,
        delegate=Delegate.CPU,
        running_mode=VisionRunningMode.LIVE_STREAM,
        pose_model_path=PoseLandmarkerPath,
        hand_model_path=HandLandmarkerPath,
        num_hands=2,
        detection_confidence=0.5,
        presence_confidence=0.5,
        tracking_confidence=0.5,
        parallel=False,
    ):
        self._delegate=delegate
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="pose") if parallel else None
        self._running_mode = running_mode
        self._time_stamper = LandmarkerLoader._TimeStampInitializer()
        self._hand_processor = StreamProcess()
        self._pose_processor = StreamProcess()

        self.pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(pose_model_path),
                delegate=delegate,
            ),
            running_mode=running_mode,
            result_callback=self._pose_processor.update if running_mode == VisionRunningMode.LIVE_STREAM else None,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(hand_model_path),
                delegate=delegate,
            ),
            running_mode=running_mode,
            result_callback=self._hand_processor.update if running_mode == VisionRunningMode.LIVE_STREAM else None,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            num_hands=num_hands,
        )

        self._pose_landmarker = PoseLandmarker.create_from_options(self.pose_options)
        try:
            self._hand_landmarker = HandLandmarker.create_from_options(self.hand_options)
        except BaseException:
            self._pose_landmarker.close()
            if self._executor is not None:
                self._executor.shutdown()
            raise

    def detect(
        self,
        frame: np.ndarray,
        wait_for_result: bool = False,
        timeout_s: float = 5.0,
        timestamp_ms: int | None = None,
    ) -> tuple[
        PoseLandmarkerResult | None,
        HandLandmarkerResult | None,
        int | None,
    ]:
        if self._running_mode == VisionRunningMode.LIVE_STREAM and wait_for_result and timeout_s <= 0:
            raise ValueError("timeout_s must be positive.")

        frame = frame_padding(frame)
        if self._delegate == Delegate.CPU:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        else:
            rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGBA, data=np.ascontiguousarray(rgba_frame))
        if self._running_mode == VisionRunningMode.IMAGE:
            if self._executor is not None:
                pose = self._executor.submit(self._pose_landmarker.detect, mp_img)
                hand = self._hand_landmarker.detect(mp_img)
                return pose.result(), hand, None
            return self._pose_landmarker.detect(mp_img), self._hand_landmarker.detect(mp_img), None

        if timestamp_ms is None:
            timestamp_ms = self._time_stamper.get2update()
        elif timestamp_ms <= self._time_stamper.timestamp:
            raise ValueError("timestamp_ms must increase on every frame.")
        else:
            self._time_stamper.timestamp = timestamp_ms

        if self._running_mode == VisionRunningMode.VIDEO:
            if self._executor is not None:
                pose = self._executor.submit(self._pose_landmarker.detect_for_video, mp_img, timestamp_ms)
                hand = self._hand_landmarker.detect_for_video(mp_img, timestamp_ms)
                return pose.result(), hand, timestamp_ms
            return (
                self._pose_landmarker.detect_for_video(mp_img, timestamp_ms),
                self._hand_landmarker.detect_for_video(mp_img, timestamp_ms),
                timestamp_ms,
            )

        self._pose_landmarker.detect_async(mp_img, timestamp_ms)
        self._hand_landmarker.detect_async(mp_img, timestamp_ms)

        if wait_for_result:
            deadline = monotonic() + timeout_s
            while True:
                pose_timestamp = self._pose_processor.latest_timestamp
                hand_timestamp = self._hand_processor.latest_timestamp

                if pose_timestamp == timestamp_ms and hand_timestamp == timestamp_ms:
                    return (
                        self._pose_processor.get(),
                        self._hand_processor.get(),
                        timestamp_ms,
                    )

                if (
                    pose_timestamp is not None
                    and pose_timestamp > timestamp_ms
                    or hand_timestamp is not None
                    and hand_timestamp > timestamp_ms
                ):
                    raise RuntimeError(
                        f"Landmarker callback skipped timestamp {timestamp_ms}."
                    )

                if monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for Pose and Hand results at "
                        f"timestamp {timestamp_ms}."
                    )
                sleep(0.001)

        pose_timestamp = self._pose_processor.latest_timestamp
        hand_timestamp = self._hand_processor.latest_timestamp

        return (
            self._pose_processor.get(),
            self._hand_processor.get(),
            pose_timestamp,
        )

    def close(self) -> None:
        """Release the underlying MediaPipe landmarker resources."""
        if self._executor is not None:
            self._executor.shutdown()
        self._pose_landmarker.close()
        self._hand_landmarker.close()

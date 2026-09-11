"""
Task 1 — Vehicle counting with a pretrained YOLO model.

This script demonstrates how to use the Ultralytics YOLOv8 object detection
network to solve an applied computer vision task: counting vehicles passing
through a region in a traffic video. The provided example expects a video such
as `cars.mp4` from the workspace root, but any road scene can be analysed.

Key steps:
1. Load a pretrained YOLOv8 model (e.g. `yolov8n.pt`) that was trained on the
   COCO dataset and already knows how to detect classes such as cars, buses,
   and trucks.
2. Track detections across frames with ByteTrack to obtain stable IDs.
3. Mark a counting line inside the frame and increment a counter whenever a
   tracked vehicle crosses it.

YOLOv8 (You Only Look Once, version 8) is a single-stage detector. An input
image passes through a CSPDarknet-inspired backbone with C2f blocks that
progressively reduce spatial resolution while enriching semantic features. A
feature pyramid neck (PAN/FPN hybrid) fuses multi-scale features, and a
decoupled head performs classification and bounding-box regression directly on
three detection scales. The model is optimised end-to-end with losses on
objectness, class predictions, and bounding boxes, enabling real-time
performance on GPU and competitive accuracy on CPU.

Usage:
    python task1_yolo_car_count.py --video cars.mp4 --weights yolov8n.pt

Dependencies:
    pip install ultralytics opencv-python
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import cv2
import numpy as np
from ultralytics import YOLO


# COCO vehicle class IDs recognised by YOLO models.
# COCO class IDs -> текстовая подпись для транспорта, который считаем.
CLASS_LABELS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class VehicleCounterConfig:
    video_path: Path
    output_path: Optional[Path]
    weights_path: str
    tracker_config: Optional[str]
    line_position_ratio: float = 0.5  # где рисовать линию по высоте (0..1)
    count_direction: str = "down"  # в какую сторону считать пересечение
    font: int = cv2.FONT_HERSHEY_SIMPLEX

    @property
    def line_color(self) -> tuple[int, int, int]:
        return 0, 255, 0

    @property
    def text_color(self) -> tuple[int, int, int]:
        return 255, 255, 255

    @property
    def vehicle_box_color(self) -> tuple[int, int, int]:
        return 0, 140, 255


class VehicleCounter:
    def __init__(self, config: VehicleCounterConfig):
        self.config = config
        self.model = YOLO(config.weights_path)  # загружаем предобученную YOLO
        self.total_count = 0
        self.counted_ids: Set[int] = set()  # ID объектов, которые уже посчитаны
        self.last_positions: Dict[int, float] = {}  # предыдущие позиции по Y
        self.writer: Optional[cv2.VideoWriter] = None  # для сохранения видео
        self.frame_size: Optional[tuple[int, int]] = None

    def _setup_writer(self, frame: np.ndarray, fps: float) -> None:
        if self.config.output_path is None:
            return
        height, width = frame.shape[:2]
        self.frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.config.output_path), fourcc, fps, self.frame_size
        )

    def _should_count(self, track_id: int, center_y: float, line_y: float) -> bool:
        previous_y = self.last_positions.get(track_id)
        self.last_positions[track_id] = center_y

        if previous_y is None:
            return False
        if self.config.count_direction == "down":
            crossed = previous_y < line_y <= center_y
        elif self.config.count_direction == "up":
            crossed = previous_y > line_y >= center_y
        else:  # both directions
            crossed = (previous_y < line_y <= center_y) or (
                previous_y > line_y >= center_y
            )
        not_counted_before = track_id not in self.counted_ids
        return crossed and not_counted_before

    def _draw_overlays(
        self,
        frame: np.ndarray,
        line_y: int,
        detections: Iterable[tuple[int, float, float]],
    ) -> None:
        cv2.line(
            frame,
            (0, line_y),
            (frame.shape[1], line_y),
            self.config.line_color,
            2,
        )
        for track_id, center_x, center_y in detections:
            cv2.circle(frame, (int(center_x), int(center_y)), 4, self.config.line_color, -1)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (int(center_x) - 10, int(center_y) - 10),
                self.config.font,
                0.5,
                self.config.vehicle_box_color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"Vehicles: {self.total_count}",
            (10, 30),
            self.config.font,
            1.0,
            self.config.text_color,
            2,
            cv2.LINE_AA,
        )

    def process(self) -> int:
        cap = cv2.VideoCapture(str(self.config.video_path))
        if not cap.isOpened():
            msg = f"Cannot open video: {self.config.video_path}"
            raise FileNotFoundError(msg)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ret, sample_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame from the video.")
        cap.release()

        self._setup_writer(sample_frame, fps)
        frame_height = sample_frame.shape[0]
        line_y = int(frame_height * self.config.line_position_ratio)  # позиция линии

        results_stream = self.model.track(
            source=str(self.config.video_path),
            stream=True,
            tracker=self.config.tracker_config,
            verbose=False,
        )

        for result in results_stream:
            frame = result.orig_img  # исходный кадр
            if frame is None:
                continue

            tracked_detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])  # индекс класса COCO
                    if cls not in CLASS_LABELS:
                        continue
                    class_name = CLASS_LABELS[cls]

                    if box.id is None:
                        continue
                    track_id = int(box.id[0])  # уникальный ID трека
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    if self._should_count(track_id, center_y, line_y):
                        self.total_count += 1
                        self.counted_ids.add(track_id)

                    tracked_detections.append((track_id, center_x, center_y))
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        self.config.vehicle_box_color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        class_name,
                        (int(x1), int(y1) - 10),
                        self.config.font,
                        0.6,
                        self.config.text_color,
                        2,
                        cv2.LINE_AA,
                    )

            self._draw_overlays(frame, line_y, tracked_detections)

            if self.writer is not None:
                self.writer.write(frame)

            cv2.imshow("YOLO Vehicle Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        return self.total_count


def parse_args() -> VehicleCounterConfig:
    parser = argparse.ArgumentParser(
        description="Count vehicles in a video using a pretrained YOLO model."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("cars.mp4"),
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path or alias to YOLOv8 weights.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker YAML config. Use '-' to disable tracking (inference only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/vehicle_count.mp4"),
        help="Optional path to save an annotated video. Use '-' to disable saving.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=("down", "up", "both"),
        default="down",
        help="Which direction counts as crossing the line.",
    )

    args = parser.parse_args()

    output_path: Optional[Path]
    if args.output == Path("-"):
        output_path = None
    else:
        output_path = args.output
        if output_path is not None and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker_config: Optional[str]
    if args.tracker == "-":
        tracker_config = None
    else:
        tracker_config = args.tracker

    return VehicleCounterConfig(
        video_path=args.video,
        output_path=output_path,
        weights_path=args.weights,
        tracker_config=tracker_config,
        count_direction=args.direction,
    )


def main() -> None:
    config = parse_args()
    counter = VehicleCounter(config)
    total = counter.process()
    print(f"Total vehicles counted: {total}")


if __name__ == "__main__":
    main()


"""
Traffic Sign Recognition & Lane Departure Detection (YOLOv8).

Runs a trained YOLOv8 model on a video, drawing:
  - traffic-sign boxes with class label + confidence
  - lane-marking boxes (the `LANE` class)
  - a live "Departure" / "No Departure" warning

Example:
    python datn.py --model best.pt --source 3.mp4 --output out.avi
    python datn.py --model best.pt --source 0 --device cuda   # webcam on GPU
"""

import argparse

import cv2
from ultralytics import YOLO

LANE_CLASS_NAME = "LANE"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic sign & lane departure detection with YOLOv8."
    )
    parser.add_argument("--model", default="best.pt", help="Path to YOLOv8 weights.")
    parser.add_argument(
        "--source",
        default="3.mp4",
        help="Video file path, or a camera index like 0.",
    )
    parser.add_argument(
        "--output",
        default="output_lane_detection.avi",
        help="Path to the annotated output video.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Inference device: 'cpu' or 'cuda'."
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold."
    )
    parser.add_argument(
        "--lane-region",
        type=float,
        default=0.55,
        help="Lane boxes are kept only if their bottom edge is below this "
        "fraction of the frame height (0-1).",
    )
    parser.add_argument(
        "--departure-threshold",
        type=float,
        default=125.0,
        help="Pixel distance from the frame center below which a lane "
        "marking triggers a departure warning.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live preview window (useful on headless devices).",
    )
    return parser.parse_args()


def open_source(source):
    """Accept either a file path or an integer camera index."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def annotate_frame(frame, results, model, region_y, vehicle_x, departure_threshold):
    """Draw lanes + signs on a frame and return the departure alert string."""
    alert = "No Departure"

    for result in results:
        # --- Lane markings: keep only boxes near the bottom of the frame ---
        lane_boxes = [
            box.tolist()
            for box in result.boxes.xyxy.cpu().numpy()
            if box[3] > region_y  # y2 below the lane region line
        ]

        for x1, y1, x2, y2 in lane_boxes:
            color = (0, 255, 0)  # green by default
            # A lane marking that sits close to the center line => departing.
            if x2 > vehicle_x and (x2 - vehicle_x) <= departure_threshold:
                alert, color = "Departure", (0, 0, 255)
            elif x1 < vehicle_x and (vehicle_x - x1) <= departure_threshold:
                alert, color = "Departure", (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # --- Traffic signs: everything that is not a lane marking ---
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            label = model.names[int(cls)]
            if label == LANE_CLASS_NAME:
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

    cv2.putText(
        frame, alert, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )
    return alert


def main():
    args = parse_args()

    model = YOLO(args.model)
    model.to(args.device)

    cap = open_source(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    region_y = height * args.lane_region
    vehicle_x = width / 2  # assume the vehicle is centered in the frame

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=args.conf, verbose=False)
            annotate_frame(
                frame, results, model, region_y, vehicle_x, args.departure_threshold
            )

            writer.write(frame)

            if not args.no_display:
                cv2.imshow("Lane Detection with Traffic Signs", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved annotated video to: {args.output}")


if __name__ == "__main__":
    main()

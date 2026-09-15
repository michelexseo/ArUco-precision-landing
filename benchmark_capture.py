"""
Capture a pose-estimation benchmark dataset.

Hold the camera still, enter tape-measure ground truth, then press SPACE to
save a burst of raw frames. Overlay is preview-only; saved images are untouched.

Usage:
    python benchmark_capture.py
    python benchmark_capture.py -t DICT_4X4_1000 -n 30

Keys:
    g     enter / change ground truth (click the terminal)
    SPACE capture burst for the current condition
    q/ESC quit
"""

import argparse
import csv
import os
import time

import cv2

from utils import ARUCO_DICT

CSV_FIELDS = [
    "condition_id",
    "distance_m",
    "yaw_deg",
    "image_region",
    "frame_index",
    "frame_path",
]
REGIONS = ("center", "left", "right")


def pose_label(yaw_deg):
    if abs(yaw_deg) < 0.5:
        return "front"
    sign = "p" if yaw_deg > 0 else "m"
    return f"yaw_{sign}{int(round(abs(yaw_deg)))}"


def make_condition_id(distance_m, yaw_deg, region):
    dist_cm = int(round(distance_m * 100))
    return f"d{dist_cm:03d}_{pose_label(yaw_deg)}_{region}"


def prompt_ground_truth(previous=None):
    print("\n--- Ground truth (click this terminal) ---")
    print("Measure tape to the center of the black square, perpendicular to the marker.")
    if previous is not None:
        print(
            "Press Enter to keep a previous value "
            f"(now {previous['distance_m']} m, {previous['yaw_deg']} deg, {previous['region']})."
        )

    distance_raw = input("Distance in meters (e.g. 0.80): ").strip()
    yaw_raw = input("Yaw in degrees (0 if facing the camera, e.g. 30 or -30): ").strip()
    region_raw = input("Image region [center/left/right]: ").strip().lower()

    if distance_raw:
        distance_m = float(distance_raw)
    elif previous is not None:
        distance_m = previous["distance_m"]
    else:
        raise ValueError("Distance is required.")

    if yaw_raw:
        yaw_deg = float(yaw_raw)
    elif previous is not None:
        yaw_deg = previous["yaw_deg"]
    else:
        raise ValueError("Yaw is required (use 0 if the marker faces the camera).")

    if region_raw:
        region = region_raw
    elif previous is not None:
        region = previous["region"]
    else:
        raise ValueError("Region is required (center, left, or right).")

    if distance_m <= 0:
        raise ValueError("Distance must be positive.")
    if region not in REGIONS:
        raise ValueError(f"Region must be one of {REGIONS}.")

    return {
        "distance_m": distance_m,
        "yaw_deg": yaw_deg,
        "region": region,
        "condition_id": make_condition_id(distance_m, yaw_deg, region),
    }


def ensure_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def replace_condition_rows(csv_path, condition_id, rows):
    ensure_csv(csv_path)
    with open(csv_path, "r", newline="") as f:
        existing = [
            row
            for row in csv.DictReader(f)
            if row.get("condition_id") != condition_id
        ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)


def draw_hud(frame, gt, marker_seen, n_frames):
    display = frame.copy()
    if marker_seen:
        status = "marker OK"
        color = (0, 200, 0)
    else:
        status = "NO marker"
        color = (0, 0, 255)

    if gt is None:
        gt_line = "GT not set  (press g)"
    else:
        gt_line = (
            f"{gt['condition_id']}   "
            f"d={gt['distance_m']:.3f}m  yaw={gt['yaw_deg']:.1f}  {gt['region']}"
        )

    cv2.putText(display, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display, gt_line, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(
        display,
        f"g=GT   SPACE=capture {n_frames} frames   q=quit",
        (16, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        2,
    )
    return display


def countdown(cap, detector, seconds):
    deadline = time.time() + seconds
    while True:
        remaining = int(deadline - time.time()) + 1
        if remaining <= 0:
            break
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        display = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
        cv2.putText(
            display,
            str(max(remaining, 1)),
            (frame.shape[1] // 2 - 40, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 255, 255),
            6,
        )
        cv2.putText(
            display,
            "Hold still",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Benchmark Capture", display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False
    return True


def capture_burst(cap, out_dir, n_frames):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera frame dropped during burst.")
            break
        rel_path = os.path.join(out_dir, f"{i:03d}.png")
        if not cv2.imwrite(rel_path, frame):
            print(f"[ERROR] Failed to write {rel_path}")
            break
        saved.append((i, rel_path.replace("\\", "/")))
        preview = frame.copy()
        cv2.putText(
            preview,
            f"Saving {i + 1}/{n_frames}",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Benchmark Capture", preview)
        cv2.waitKey(1)
    return saved


def main():
    ap = argparse.ArgumentParser(description="Capture pose-estimation benchmark frames.")
    ap.add_argument(
        "-t",
        "--type",
        type=str,
        default="DICT_4X4_1000",
        help="ArUCo dictionary used only for the live preview",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="benchmark",
        help="root directory for frames and captures.csv",
    )
    ap.add_argument(
        "-n",
        "--n-frames",
        type=int,
        default=30,
        help="frames to save per condition",
    )
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    if args.type not in ARUCO_DICT:
        print(f"[ERROR] ArUCo tag type '{args.type}' is not supported.")
        return
    if args.n_frames < 1:
        print("[ERROR] --n-frames must be >= 1")
        return

    frames_root = os.path.join(args.output_dir, "frames")
    csv_path = os.path.join(args.output_dir, "captures.csv")
    os.makedirs(frames_root, exist_ok=True)
    ensure_csv(csv_path)

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args.type]),
        cv2.aruco.DetectorParameters(),
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    time.sleep(2.0)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print("--- Pose benchmark capture ---")
    print(f"Saving raw frames under '{frames_root}/<condition_id>/'")
    print(f"Metadata: '{csv_path}'")
    print("Press 'g' to enter ground truth, SPACE to capture, q to quit.")

    gt = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        marker_seen = ids is not None and len(ids) > 0
        display = frame.copy()
        if marker_seen:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
        display = draw_hud(display, gt, marker_seen, args.n_frames)
        cv2.imshow("Benchmark Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        if key == ord("g"):
            try:
                gt = prompt_ground_truth(gt)
                print(f"[OK] Condition {gt['condition_id']}. Click the video window, then SPACE.")
            except (ValueError, KeyError) as exc:
                print(f"[ERROR] {exc}")
            continue

        if key != 32:  # SPACE
            continue

        if gt is None:
            print("[WARN] Set ground truth first (press g).")
            continue

        condition_dir = os.path.join(frames_root, gt["condition_id"])
        if os.path.isdir(condition_dir) and os.listdir(condition_dir):
            print(f"[WARN] '{condition_dir}' already has files.")
            overwrite = input("Overwrite this condition? [y/N]: ").strip().lower()
            if overwrite != "y":
                print("Capture skipped.")
                continue
            for name in os.listdir(condition_dir):
                os.remove(os.path.join(condition_dir, name))

        print("Hold still...")
        if not countdown(cap, detector, seconds=3):
            print("Capture cancelled.")
            break

        saved = capture_burst(cap, condition_dir, args.n_frames)
        if len(saved) != args.n_frames:
            print(f"[WARN] Saved {len(saved)}/{args.n_frames} frames.")

        rows = [
            {
                "condition_id": gt["condition_id"],
                "distance_m": f"{gt['distance_m']:.4f}",
                "yaw_deg": f"{gt['yaw_deg']:.1f}",
                "image_region": gt["region"],
                "frame_index": idx,
                "frame_path": path,
            }
            for idx, path in saved
        ]
        replace_condition_rows(csv_path, gt["condition_id"], rows)
        print(f"[OK] {len(rows)} frames → {condition_dir}")
        print("Press g for the next condition, or SPACE to recapture this one.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

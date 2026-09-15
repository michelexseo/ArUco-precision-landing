"""
Evaluate pose estimation on a captured benchmark dataset.

Reads benchmark/captures.csv, runs the same PnP pipeline as pose_estimation.py
on each saved frame, then writes per-frame and per-condition metrics.

Usage:
    python benchmark_eval.py
    python benchmark_eval.py -cp camera_calibration.npz -s 0.1 -t DICT_4X4_1000
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

from utils import ARUCO_DICT

FRAME_FIELDS = [
    "condition_id",
    "frame_index",
    "frame_path",
    "distance_m",
    "yaw_deg",
    "image_region",
    "detected",
    "marker_id",
    "tvec_x",
    "tvec_y",
    "tvec_z",
    "distance",
    "yaw_est_deg",
    "reproj_rmse",
]

SUMMARY_FIELDS = [
    "condition_id",
    "image_region",
    "n_frames",
    "n_detected",
    "det_pct",
    "d_gt",
    "mean_d",
    "std_d",
    "std_d_mm",
    "bias",
    "bias_cm",
    "rel_pct",
    "mae",
    "rmse",
    "scale",
    "mean_z",
    "bias_z",
    "yaw_gt",
    "mean_yaw",
    "std_yaw",
    "yaw_err",
    "mean_reproj",
]


def marker_object_points(side_length):
    half = side_length / 2.0
    return np.array(
        [
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0],
        ],
        dtype=np.float32,
    )


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def marker_yaw_deg(rvec):
    """Yaw of the marker normal in the camera XZ plane. 0 = facing the camera."""
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    return float(np.degrees(np.arctan2(normal[0], -normal[2])))


def refine_corners(gray, image_points):
    refined = cv2.cornerSubPix(
        gray,
        image_points.reshape(-1, 1, 2),
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
    )
    return refined.reshape(4, 2)


def estimate_pose(gray, detector, camera_matrix, dist_coeffs, object_points, refine=False, marker_id=None):
    corners, ids, _ = detector.detectMarkers(gray)
    empty = {
        "detected": False,
        "marker_id": "",
        "tvec": None,
        "distance": np.nan,
        "yaw_est_deg": np.nan,
        "reproj_rmse": np.nan,
    }
    if ids is None or len(corners) == 0:
        return empty

    ids = ids.flatten()
    index = 0
    if marker_id is not None:
        matches = np.where(ids == marker_id)[0]
        if len(matches) == 0:
            return empty
        index = int(matches[0])

    image_points = corners[index].reshape(4, 2).astype(np.float32)
    if refine:
        image_points = refine_corners(gray, image_points)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return empty

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    residual = projected.reshape(4, 2) - image_points
    reproj_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    tvec = tvec.reshape(3)

    return {
        "detected": True,
        "marker_id": int(ids[index]),
        "tvec": tvec,
        "distance": float(np.linalg.norm(tvec)),
        "yaw_est_deg": marker_yaw_deg(rvec),
        "reproj_rmse": reproj_rmse,
    }


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "nan"
    return f"{value:.{digits}f}"


def print_table(summaries):
    headers = [
        ("condition", 28),
        ("d_gt", 7),
        ("mean_d", 8),
        ("bias_cm", 8),
        ("rel_%", 7),
        ("std_mm", 7),
        ("yaw_err", 8),
        ("reproj", 7),
        ("det%", 6),
    ]
    header = "  ".join(name.ljust(width) for name, width in headers)
    print("\n" + header)
    print("-" * len(header))
    for row in summaries:
        values = [
            str(row["condition_id"]).ljust(28),
            fmt(row["d_gt"], 3).rjust(7),
            fmt(row["mean_d"], 3).rjust(8),
            fmt(row["bias_cm"], 2).rjust(8),
            fmt(row["rel_pct"], 2).rjust(7),
            fmt(row["std_d_mm"], 1).rjust(7),
            fmt(row["yaw_err"], 1).rjust(8),
            fmt(row["mean_reproj"], 2).rjust(7),
            fmt(row["det_pct"], 1).rjust(6),
        ]
        print("  ".join(values))


def summarize_group(condition_id, rows):
    n_frames = len(rows)
    detected = [row for row in rows if row["detected"]]
    n_detected = len(detected)
    unique_d_gt = {float(row["distance_m"]) for row in rows}
    unique_yaw_gt = {float(row["yaw_deg"]) for row in rows}
    unique_region = {row["image_region"] for row in rows}
    d_gt = next(iter(unique_d_gt)) if len(unique_d_gt) == 1 else np.nan
    yaw_gt = next(iter(unique_yaw_gt)) if len(unique_yaw_gt) == 1 else np.nan
    region = next(iter(unique_region)) if len(unique_region) == 1 else "all"

    distances = np.array([row["distance"] for row in detected], dtype=float)
    d_gts = np.array([row["distance_m"] for row in detected], dtype=float)
    zs = np.array([row["tvec_z"] for row in detected], dtype=float)
    yaws = np.array([row["yaw_est_deg"] for row in detected], dtype=float)
    yaw_gts = np.array([row["yaw_deg"] for row in detected], dtype=float)
    reprojs = np.array([row["reproj_rmse"] for row in detected], dtype=float)

    if n_detected == 0:
        mean_d = std_d = bias = mae = rmse = scale = rel_pct = np.nan
        mean_z = bias_z = mean_yaw = std_yaw = yaw_err = mean_reproj = np.nan
    else:
        errors = distances - d_gts
        mean_d = float(np.mean(distances))
        # std is repeatability only when every frame shares the same tape distance
        if len(unique_d_gt) == 1 and n_detected > 1:
            std_d = float(np.std(distances, ddof=1))
        elif len(unique_d_gt) == 1:
            std_d = 0.0
        else:
            std_d = np.nan
        bias = float(np.mean(errors))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        scale = float(np.mean(distances / d_gts))
        rel_pct = float(np.mean(100.0 * errors / d_gts))
        mean_z = float(np.mean(zs))
        bias_z = float(np.mean(zs - d_gts))
        mean_yaw = float(np.mean(yaws))
        yaw_errs = np.array([wrap_deg(est - gt) for est, gt in zip(yaws, yaw_gts)])
        yaw_err = float(np.mean(yaw_errs))
        if len(unique_yaw_gt) == 1 and n_detected > 1:
            std_yaw = float(np.std(yaws, ddof=1))
        elif len(unique_yaw_gt) == 1:
            std_yaw = 0.0
        else:
            std_yaw = np.nan
        mean_reproj = float(np.mean(reprojs))

    return {
        "condition_id": condition_id,
        "image_region": region,
        "n_frames": n_frames,
        "n_detected": n_detected,
        "det_pct": 100.0 * n_detected / n_frames if n_frames else 0.0,
        "d_gt": d_gt,
        "mean_d": mean_d,
        "std_d": std_d,
        "std_d_mm": std_d * 1000.0 if np.isfinite(std_d) else np.nan,
        "bias": bias,
        "bias_cm": bias * 100.0 if np.isfinite(bias) else np.nan,
        "rel_pct": rel_pct,
        "mae": mae,
        "rmse": rmse,
        "scale": scale,
        "mean_z": mean_z,
        "bias_z": bias_z,
        "yaw_gt": yaw_gt,
        "mean_yaw": mean_yaw,
        "std_yaw": std_yaw,
        "yaw_err": yaw_err,
        "mean_reproj": mean_reproj,
    }


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row[key]
                if isinstance(value, bool):
                    out[key] = int(value)
                elif isinstance(value, float):
                    out[key] = "" if not np.isfinite(value) else f"{value:.6f}"
                else:
                    out[key] = value
            writer.writerow(out)


def load_captures(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No capture file at '{csv_path}'. Run benchmark_capture.py first."
        )
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"'{csv_path}' has no data rows.")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate pose estimation on captured frames.")
    ap.add_argument(
        "-t",
        "--type",
        type=str,
        default="DICT_4X4_1000",
        help="ArUCo dictionary type",
    )
    ap.add_argument(
        "-cp",
        "--calibration_params",
        type=str,
        default="camera_calibration.npz",
        help="path to calibration .npz",
    )
    ap.add_argument(
        "-s",
        "--side_length",
        type=float,
        default=0.1,
        help="black-square side length in meters",
    )
    ap.add_argument(
        "-i",
        "--captures",
        type=str,
        default=os.path.join("benchmark", "captures.csv"),
        help="CSV written by benchmark_capture.py",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="benchmark",
        help="directory for results.csv and summary.csv",
    )
    ap.add_argument(
        "--marker-id",
        type=int,
        default=None,
        help="use this marker ID if several are in the frame (default: first)",
    )
    ap.add_argument(
        "--refine",
        action="store_true",
        help="refine corners with cornerSubPix before solvePnP",
    )
    args = ap.parse_args()

    if args.type not in ARUCO_DICT:
        print(f"[ERROR] ArUCo tag type '{args.type}' is not supported.")
        return
    if args.side_length <= 0:
        print("[ERROR] --side_length must be positive.")
        return
    if not os.path.exists(args.calibration_params):
        print(f"[ERROR] Calibration file '{args.calibration_params}' does not exist.")
        return

    captures = load_captures(args.captures)
    calib = np.load(args.calibration_params)
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]
    object_points = marker_object_points(args.side_length)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args.type]),
        cv2.aruco.DetectorParameters(),
    )

    print(f"Loaded {len(captures)} frames from '{args.captures}'")
    print(f"Calibration: {args.calibration_params}")
    print(f"Marker side: {args.side_length} m   refine={args.refine}")

    frame_rows = []
    grouped = defaultdict(list)

    for capture in captures:
        frame_path = capture["frame_path"]
        image = cv2.imread(frame_path)
        result = {
            "condition_id": capture["condition_id"],
            "frame_index": capture["frame_index"],
            "frame_path": frame_path,
            "distance_m": float(capture["distance_m"]),
            "yaw_deg": float(capture["yaw_deg"]),
            "image_region": capture["image_region"],
            "detected": False,
            "marker_id": "",
            "tvec_x": np.nan,
            "tvec_y": np.nan,
            "tvec_z": np.nan,
            "distance": np.nan,
            "yaw_est_deg": np.nan,
            "reproj_rmse": np.nan,
        }

        if image is None:
            print(f"[WARN] Could not read {frame_path}")
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pose = estimate_pose(
                gray,
                detector,
                camera_matrix,
                dist_coeffs,
                object_points,
                refine=args.refine,
                marker_id=args.marker_id,
            )
            result["detected"] = pose["detected"]
            result["marker_id"] = pose["marker_id"]
            result["distance"] = pose["distance"]
            result["yaw_est_deg"] = pose["yaw_est_deg"]
            result["reproj_rmse"] = pose["reproj_rmse"]
            if pose["tvec"] is not None:
                result["tvec_x"], result["tvec_y"], result["tvec_z"] = pose["tvec"]

        frame_rows.append(result)
        grouped[capture["condition_id"]].append(result)

    summaries = [
        summarize_group(condition_id, rows)
        for condition_id, rows in grouped.items()
    ]
    overall = summarize_group("ALL", frame_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.csv")
    summary_path = os.path.join(args.output_dir, "summary.csv")
    write_csv(results_path, FRAME_FIELDS, frame_rows)
    write_csv(summary_path, SUMMARY_FIELDS, summaries + [overall])

    print_table(summaries + [overall])
    print(f"\n[OK] Per-frame results: {results_path}")
    print(f"[OK] Per-condition summary: {summary_path}")
    if np.isfinite(overall["scale"]):
        print(
            f"Overall scale mean(||t||)/d_gt = {overall['scale']:.4f}  "
            f"(1.00 is ideal; 0.94 was the old calibration bias)"
        )


if __name__ == "__main__":
    main()

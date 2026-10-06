import csv
import cv2
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

CSV_COLUMNS = [
    "timestamp", "loop_ms", "detected",
    "tvec_x", "tvec_y", "tvec_z", "dist_m",
    "body_x", "body_y", "body_z",
    "veh_x", "veh_y", "veh_z", "roll", "pitch", "yaw",
    "armed", "flight_mode",
    "pad_x", "pad_y", "pad_z", "diff_x", "diff_y", "diff_z",
    "sent", "skip_reason",
]


def aruco_display(corners, ids, rejected, image):
    if ids is None or len(corners) == 0:
        return image

    ids = ids.flatten()

    for marker_corner, marker_id in zip(corners, ids):
        points = marker_corner.reshape((4, 2)).astype(int)
        top_left, top_right, bottom_right, bottom_left = points

        cv2.polylines(
            image,
            [points.reshape((-1, 1, 2))],
            isClosed=True,
            color=(0, 255, 0),
            thickness=5,
            lineType=cv2.LINE_AA,
        )

        center_x = int((top_left[0] + bottom_right[0]) / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2)

        cv2.circle(
            image,
            (center_x, center_y),
            8,
            (0, 0, 255),
            -1,
        )

        text = f"ID: {marker_id}"
        text_x = max(10, top_left[0])
        text_y = max(45, top_left[1] - 15)

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            4,
            cv2.LINE_AA,
        )

    return image


def _fmt_float(value, decimals=4):
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}"


class FlightLogger:
    def __init__(
        self,
        log_dir="logs",
        summary_interval=5.0,
        run_id=None,
        save_frames=False,
        frame_interval=30,
    ):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_interval = summary_interval
        self.last_summary = time.time()
        self.save_frames = save_frames
        self.frame_interval = frame_interval
        self.frames_dir = self.log_dir / f"frames_{self.run_id}"
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / f"data_{self.run_id}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(CSV_COLUMNS)

        self.loops = 0
        self.detections = 0
        self.sent = 0
        self.pos_ok = 0
        self.att_ok = 0
        self.no_frame = 0
        self.no_marker = 0
        self.no_pos_att = 0
        self.frames_saved = 0

    def log_frame(
        self,
        *,
        loop_ms,
        detected=False,
        tvec=None,
        target_body=None,
        veh_ned=None,
        attitude=None,
        armed=None,
        flight_mode=None,
        pad_ned=None,
        diff_ned=None,
        sent=False,
        skip_reason="",
        frame=None,
    ):
        self.loops += 1
        if detected:
            self.detections += 1
            if (
                self.save_frames
                and frame is not None
                and self.frame_interval > 0
                and self.loops % self.frame_interval == 0
            ):
                self._save_frame(frame)
        if veh_ned is not None:
            self.pos_ok += 1
        if attitude is not None:
            self.att_ok += 1
        if sent:
            self.sent += 1
        if skip_reason == "no_frame":
            self.no_frame += 1
        elif skip_reason == "no_marker":
            self.no_marker += 1
        elif skip_reason == "no_pos_att":
            self.no_pos_att += 1

        tvec_vals = [None, None, None]
        dist_m = None
        if tvec is not None:
            flat = tvec.reshape(3)
            tvec_vals = [float(flat[0]), float(flat[1]), float(flat[2])]
            dist_m = float((flat[0] ** 2 + flat[1] ** 2 + flat[2] ** 2) ** 0.5)

        body_vals = list(target_body) if target_body is not None else [None, None, None]
        veh_vals = list(veh_ned) if veh_ned is not None else [None, None, None]
        att_vals = list(attitude) if attitude is not None else [None, None, None]
        pad_vals = list(pad_ned) if pad_ned is not None else [None, None, None]
        diff_vals = list(diff_ned) if diff_ned is not None else [None, None, None]

        row = [
            datetime.now(timezone.utc).isoformat(),
            _fmt_float(loop_ms, 1),
            int(detected),
            *[_fmt_float(v) for v in tvec_vals],
            _fmt_float(dist_m),
            *[_fmt_float(v) for v in body_vals],
            *[_fmt_float(v) for v in veh_vals],
            *[_fmt_float(v) for v in att_vals],
            "" if armed is None else int(armed),
            "" if flight_mode is None else int(flight_mode),
            *[_fmt_float(v, 2) for v in pad_vals],
            *[_fmt_float(v, 2) for v in diff_vals],
            int(sent),
            skip_reason,
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        now = time.time()
        if now - self.last_summary >= self.summary_interval:
            self.write_summary()
            self.last_summary = now

        if sent and pad_ned is not None and veh_ned is not None and tvec is not None:
            logging.info(
                "sent pad NED=(%.1f, %.1f, %.1f) veh NED=(%.1f, %.1f, %.1f) "
                "tvec=(%.2f, %.2f, %.2f) diff=(%.2f, %.2f, %.2f)",
                pad_ned[0], pad_ned[1], pad_ned[2],
                veh_ned[0], veh_ned[1], veh_ned[2],
                tvec_vals[0], tvec_vals[1], tvec_vals[2],
                diff_vals[0] or 0, diff_vals[1] or 0, diff_vals[2] or 0,
            )

    def _save_frame(self, frame):
        path = self.frames_dir / f"frame_{self.run_id}_{self.loops:06d}.jpg"
        if cv2.imwrite(str(path), frame):
            self.frames_saved += 1
            logging.info("Saved debug frame %s", path.name)
        else:
            logging.warning("Failed to save debug frame %s", path)

    def write_summary(self):
        rate = 100.0 * self.detections / max(self.loops, 1)
        pos_rate = 100.0 * self.pos_ok / max(self.loops, 1)
        att_rate = 100.0 * self.att_ok / max(self.loops, 1)
        logging.info(
            "SUMMARY loops=%d detections=%d rate=%.1f%% sent=%d "
            "pos_ok=%d (%.1f%%) att_ok=%d (%.1f%%) "
            "no_frame=%d no_marker=%d no_pos_att=%d frames_saved=%d",
            self.loops, self.detections, rate, self.sent,
            self.pos_ok, pos_rate, self.att_ok, att_rate,
            self.no_frame, self.no_marker, self.no_pos_att,
            self.frames_saved,
        )

    def close(self):
        self.write_summary()
        self.csv_file.close()
        logging.info("CSV log saved to %s", self.csv_path)
        if self.save_frames:
            logging.info("Debug frames saved to %s", self.frames_dir)
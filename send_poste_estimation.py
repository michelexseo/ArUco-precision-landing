import os, time
os.environ["MAVLINK20"] = "1"

import cv2
import numpy as np
from pymavlink import mavutil
from utils import ARUCO_DICT, FlightLogger
import math
import yaml
from pathlib import Path
import logging
from datetime import datetime, timezone

def setup_logging(log_dir="logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/run_{run_id}.log"),
            logging.StreamHandler(),
        ],
    )
    return run_id


# Downward camera: OpenCV (x right, y down, z optical) -> FRD
R_CAM_TO_BODY = np.array([
    [ 0, -1, 0],  # x_frd = -y_cam
    [ 1,  0, 0],  # y_frd =  x_cam
    [ 0,  0, 1],  # z_frd =  z_cam
], dtype=float)

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def mavlink_message_types(config):
    types = list(config.get("LOG_MAVLINK_MESSAGES") or [])
    for required in ("ATTITUDE", "LOCAL_POSITION_NED"):
        if required not in types:
            types.append(required)
    return types

def heartbeat_fields(hb):
    if hb is None:
        return None, None
    armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return armed, hb.custom_mode

def rotation_body_to_ned(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ]

def body_to_ned_offset(target_body, roll, pitch, yaw):
    """Rotate a body-frame offset vector into NED."""
    R = rotation_body_to_ned(roll, pitch, yaw)
    x_b, y_b, z_b = target_body
    x_n = R[0][0] * x_b + R[0][1] * y_b + R[0][2] * z_b
    y_n = R[1][0] * x_b + R[1][1] * y_b + R[1][2] * z_b
    z_n = R[2][0] * x_b + R[2][1] * y_b + R[2][2] * z_b
    return x_n, y_n, z_n

def camera_to_frd(tvec):
    return R_CAM_TO_BODY @ tvec.reshape(3)

def estimate_marker(frame, detector, K, dist, side, expected_marker_id):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or ids[0] != expected_marker_id:
        return None
    marker_pts = np.array([
        [-side/2,  side/2, 0],
        [ side/2,  side/2, 0],
        [ side/2, -side/2, 0],
        [-side/2, -side/2, 0],
    ], np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        marker_pts, corners[0].reshape(4, 2).astype(np.float32),
        K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return tvec if ok else None

def mavlink_connection(mavlink_connection_string):
    master = mavutil.mavlink_connection(
        mavlink_connection_string,
        source_system=255,
        source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER,
    )

    logging.info("Sending initial heartbeat to register with router...")
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
    )

    logging.info("Waiting for heartbeat from PX4...")
    master.wait_heartbeat()
    logging.info(
        "Connected to PX4 (sys=%s comp=%s)",
        master.target_system, master.target_component,
    )

    def request_message(msg_id, hz):
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0,
        )

    logging.info("Requesting ATTITUDE and LOCAL_POSITION_NED at 10 Hz...")
    request_message(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 10)
    request_message(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10)

    return master, master.mav

def video_capture(config):
    cap = cv2.VideoCapture(config["CAMERA_INDEX"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["CAMERA_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["CAMERA_HEIGHT"])
    return cap

def calibration_parameters(config):
    calib = np.load(config["CALIBRATION_PARAMS"])
    K, dist = calib["camera_matrix"], calib["dist_coeffs"]
    return K, dist

def aruco_detector(config):
    aruco_dict_type = ARUCO_DICT[config["ARUCO_DICT_TYPE"]]
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(aruco_dict_type),
        cv2.aruco.DetectorParameters(),
    )
    return detector

def send_landing_target(mav, x_ned, y_ned, z_ned):
    mav.landing_target_send(
        int(time.time() * 1_000_000),
        0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0.0, 0.0, 0.0, 0.0, 0.0,
        x=x_ned, y=y_ned, z=z_ned,
        q=[1.0, 0.0, 0.0, 0.0],
        position_valid=1,
    )

def should_log_frame(log_every_frame, detected, sent, skip_reason):
    if log_every_frame:
        return True
    return detected or sent or skip_reason == "no_pos_att"

def record_frame(logger, config, loop_start, **fields):
    if not should_log_frame(
        config.get("LOG_EVERY_FRAME", True),
        fields.get("detected", False),
        fields.get("sent", False),
        fields.get("skip_reason", ""),
    ):
        return
    loop_ms = (time.time() - loop_start) * 1000
    logger.log_frame(loop_ms=loop_ms, **fields)
    

def main_loop(config, master, mav, cache, cap, detector, K, dist, logger):
    last_hb = time.time()
    mavlink_types = mavlink_message_types(config)

    while True:
        loop_start = time.time()

        now = time.time()
        if now - last_hb > 1.0:
            mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
            )
            last_hb = now

        while True:
            msg = master.recv_match(type=mavlink_types, blocking=False)
            if not msg:
                break
            cache[msg.get_type()] = msg

        pos = cache.get("LOCAL_POSITION_NED")
        att = cache.get("ATTITUDE")
        hb = cache.get("HEARTBEAT")
        armed, flight_mode = heartbeat_fields(hb)

        veh_ned = (pos.x, pos.y, pos.z) if pos is not None else None
        attitude = (att.roll, att.pitch, att.yaw) if att is not None else None

        ok, frame = cap.read()
        if not ok:
            record_frame(
                logger, config, loop_start,
                veh_ned=veh_ned, attitude=attitude,
                armed=armed, flight_mode=flight_mode,
                skip_reason="no_frame",
            )
            time.sleep(0.05)
            continue

        tvec = estimate_marker(
            frame, detector, K, dist,
            side=config["SIDE_LENGTH"],
            expected_marker_id=config["EXPECTED_MARKER_ID"],
        )

        if tvec is None:
            record_frame(
                logger, config, loop_start,
                veh_ned=veh_ned, attitude=attitude,
                armed=armed, flight_mode=flight_mode,
                skip_reason="no_marker",
            )
            continue

        if pos is None or att is None:
            record_frame(
                logger, config, loop_start,
                detected=True, tvec=tvec, frame=frame,
                veh_ned=veh_ned, attitude=attitude,
                armed=armed, flight_mode=flight_mode,
                skip_reason="no_pos_att",
            )
            continue

        target_body = tuple(camera_to_frd(tvec))
        dx, dy, dz = body_to_ned_offset(target_body, att.roll, att.pitch, att.yaw)
        x_ned = pos.x + dx
        y_ned = pos.y + dy
        z_ned = pos.z + dz
        diff_ned = (pos.x - x_ned, pos.y - y_ned, pos.z - z_ned)

        
        send_landing_target(mav, x_ned, y_ned, z_ned)

        record_frame(
            logger, config, loop_start,
            detected=True, tvec=tvec, target_body=target_body, frame=frame,
            veh_ned=veh_ned, attitude=attitude,
            armed=armed, flight_mode=flight_mode,
            pad_ned=(x_ned, y_ned, z_ned),
            diff_ned=diff_ned,
            sent=True,
        )

def main():
    config = load_config()

    log_dir = config.get("LOG_DIR", "logs")
    summary_interval = config.get("LOG_SUMMARY_INTERVAL_SEC", 5)
    run_id = setup_logging(log_dir)
    logger = FlightLogger(
        log_dir,
        summary_interval,
        run_id=run_id,
        save_frames=config.get("LOG_SAVE_FRAMES", False),
        frame_interval=config.get("LOG_FRAME_INTERVAL", 30),
    )
    logging.info("Starting precision landing (run_id=%s)", run_id)

    try:
        master, mav = mavlink_connection(config["MAVLINK_CONNECTION_STRING"])
        cache = {}
        cap = video_capture(config)
        K, dist = calibration_parameters(config)
        detector = aruco_detector(config)
        main_loop(config, master, mav, cache, cap, detector, K, dist, logger)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

import os, time
os.environ["MAVLINK20"] = "1"

import cv2
import numpy as np
from pymavlink import mavutil
from utils import ARUCO_DICT
import math
import argparse

# Downward camera: OpenCV (x right, y down, z optical) -> FRD
R_CAM_TO_BODY = np.array([
    [ 0, -1, 0],  # x_frd = -y_cam
    [ 1,  0, 0],  # y_frd =  x_cam
    [ 0,  0, 1],  # z_frd =  z_cam
], dtype=float)


def latest_message(master, msgtype, cache):
    msg = master.recv_match(type=msgtype, blocking=False)
    if msg is not None:
        cache[msgtype] = msg
    return cache.get(msgtype)

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

def estimate_marker(frame, detector, K, dist, side):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    marker_pts = np.array([
        [-side/2,  side/2, 0],
        [ side/2,  side/2, 0],
        [ side/2, -side/2, 0],
        [-side/2, -side/2, 0],
    ], np.float32)
    # pick first / expected ID
    ok, rvec, tvec = cv2.solvePnP(
        marker_pts, corners[0].reshape(4, 2).astype(np.float32),
        K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return tvec if ok else None


if __name__ == "__main__":
    ap  = argparse.ArgumentParser()
    ap.add_argument(
        "-t",
        "--type",
        type=str,
        default="DICT_4X4_1000",
        help="type of ArUCo tag to detect",
    )

    ap.add_argument(
        "-cp",
        "--calibration_params",
        type=str,
        default="camera_calibration.npz",
        help="path to the calibration parameters file"
    )

    ap.add_argument(
        "-s",
        "--side_length",
        type=float,
        default=0.1,
        help="side length of the ArUCo marker in meters"
    )

    args = vars(ap.parse_args())

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_params = args["calibration_params"]
    side_length = args["side_length"]
    calib = np.load(calibration_params)
    K, dist = calib["camera_matrix"], calib["dist_coeffs"]
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(aruco_dict_type),
        cv2.aruco.DetectorParameters(),
    )

    mavutil.set_dialect("common")
    master = mavutil.mavlink_connection("udpin:0.0.0.0:14540")
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Connected to PX4 — sending LANDING_TARGET from camera")
    mav = master.mav
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    calib = np.load("camera_calibration.npz")
    K, dist = calib["camera_matrix"], calib["dist_coeffs"]
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_1000"]),
        cv2.aruco.DetectorParameters(),
    )
    print(f"Using ArUCo tag type: {aruco_dict_type}")
    print(f"Using calibration parameters from: {calibration_params}")
    print(f"Using side length: {side_length} meters")
    print(f"Using camera matrix: {K}")
    print(f"Using distortion coefficients: {dist}")
    cache = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No camera frame")
            time.sleep(0.05)
            continue
        tvec = estimate_marker(frame, detector, K, dist, side=0.1)
        pos = latest_message(master, "LOCAL_POSITION_NED", cache)
        att = latest_message(master, "ATTITUDE", cache)
        # Optional PC debug: see the tag. Remove on the Pi / systemd.
        cv2.imshow("pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if tvec is None:
            print("No ArUco detection — not sending")
            time.sleep(0.1)
            continue
        if pos is None or att is None:
            print("PX4 pos/att not received yet — not sending")
            time.sleep(0.01)
            continue
        target_body = tuple(camera_to_frd(tvec))
        dx, dy, dz = body_to_ned_offset(target_body, att.roll, att.pitch, att.yaw)
        x_ned = pos.x + dx
        y_ned = pos.y + dy
        z_ned = pos.z + dz
        mav.landing_target_send(
            int(time.time() * 1_000_000),
            0,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0.0, 0.0, 0.0, 0.0, 0.0,
            x=x_ned, y=y_ned, z=z_ned,
            q=[1.0, 0.0, 0.0, 0.0],
            # type=mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL,
            position_valid=1,
        )
        print(
            f"vehicle NED=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})  "
            f"cam tvec=({tvec[0,0]:.2f}, {tvec[1,0]:.2f}, {tvec[2,0]:.2f})  "
            f"detected body=({target_body[0]:.2f}, {target_body[1]:.2f}, {target_body[2]:.2f})  "
            f"sent pad NED=({x_ned:.1f}, {y_ned:.1f}, {z_ned:.1f})"
            f"print difference between detected and vehicle positon {pos.x - x_ned: .2f} {pos.y - y_ned: .2f} {pos.z - z_ned: .2f}"
        )
        time.sleep(0.1)
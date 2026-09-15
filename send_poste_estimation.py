import os, time
os.environ["MAVLINK20"] = "1"

import cv2
import numpy as np
from pymavlink import mavutil
from utils import ARUCO_DICT
import math
import argparse
import yaml
from pathlib import Path

# Downward camera: OpenCV (x right, y down, z optical) -> FRD
R_CAM_TO_BODY = np.array([
    [ 0, -1, 0],  # x_frd = -y_cam
    [ 1,  0, 0],  # y_frd =  x_cam
    [ 0,  0, 1],  # z_frd =  z_cam
], dtype=float)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

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

def mavlink_connection(mavlink_connection_string):
    master = mavutil.mavlink_connection(mavlink_connection_string)
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Connected to PX4 — sending LANDING_TARGET from camera")
    mav = master.mav
    return master, mav

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



def main_loop(config, master, mav, cache, cap, detector, K, dist, pos, att):
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No camera frame")
            time.sleep(0.05)
            continue
        # estimate the marker position
        tvec = estimate_marker(frame, detector, K, dist, side=config["SIDE_LENGTH"])
        # if no marker is detected, sleep for 0.1 seconds and continue
        if tvec is None:
            print("No ArUco detection — not sending")
            time.sleep(0.1)
            continue
        pos = latest_message(master, "LOCAL_POSITION_NED", cache)
        att = latest_message(master, "ATTITUDE", cache)
        # if no marker is detected, sleep for 0.1 seconds and continue
        if tvec is None:
            print("No ArUco detection — not sending")
            time.sleep(0.1)
            continue
        if pos is None or att is None:
            print("PX4 pos/att not received yet — not sending")
            time.sleep(0.01)
            continue

        # convert the marker position to body frame
        target_body = tuple(camera_to_frd(tvec))

        # convert the marker position to NED frame
        dx, dy, dz = body_to_ned_offset(target_body, att.roll, att.pitch, att.yaw)

        # get the target position in NED frame
        x_ned = pos.x + dx
        y_ned = pos.y + dy
        z_ned = pos.z + dz

        # send the target position to the vehicle
        send_landing_target(mav, x_ned, y_ned, z_ned)

        # print the target position
        print(
            f"vehicle NED=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})  "
            f"cam tvec=({tvec[0,0]:.2f}, {tvec[1,0]:.2f}, {tvec[2,0]:.2f})  "
            f"detected body=({target_body[0]:.2f}, {target_body[1]:.2f}, {target_body[2]:.2f})  "
            f"sent pad NED=({x_ned:.1f}, {y_ned:.1f}, {z_ned:.1f})"
            f"print difference between detected and vehicle positon {pos.x - x_ned: .2f} {pos.y - y_ned: .2f} {pos.z - z_ned: .2f}"
        )

        # sleep for 0.1 seconds
        time.sleep(0.1)

def main():
    # load configuration variables from config.yaml
    config = load_config()

    # connect to the mavlink
    master, mav = mavlink_connection(config["MAVLINK_CONNECTION_STRING"])
    cache = {}
    
    # get the latest message from the vehicle locacl position and attitude topic
    pos = latest_message(master, "LOCAL_POSITION_NED", cache)
    att = latest_message(master, "ATTITUDE", cache)

    # set up the video capture
    cap = video_capture(config)
   
    # load the calibration parameters
    K, dist = calibration_parameters(config)

    # set up the aruco detector
    detector = aruco_detector(config)

    # start the main loop
    main_loop(config, master, mav, cache, cap, detector, K, dist, pos, att)


if __name__ == "__main__":
    main()


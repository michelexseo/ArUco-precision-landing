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
    # pick first / expected ID
    ok, rvec, tvec = cv2.solvePnP(
        marker_pts, corners[0].reshape(4, 2).astype(np.float32),
        K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return tvec if ok else None

def mavlink_connection(mavlink_connection_string):
    # FIX 1: Set component ID so PX4 recognizes the companion computer
    master = mavutil.mavlink_connection(
        mavlink_connection_string,
        source_system=255,
        source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
    )
    
    # FIX 2: Send initial heartbeat to wake up mavlink-router over UDP
    print("Sending initial heartbeat to register with router...")
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
    )

    print("Waiting for heartbeat from PX4...")
    master.wait_heartbeat()
    print("Connected to PX4!")
    
    # FIX 3: Request the specific topics you need at 10 Hz
    def request_message(msg_id, hz):
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0
        )
    print("Requesting ATTITUDE and LOCAL_POSITION_NED streams...")
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

def main_loop(config, master, mav, cache, cap, detector, K, dist):
    last_hb = time.time()
    
    while True:
        now = time.time()
        
        # FIX 4: Send companion heartbeat at 1 Hz
        if now - last_hb > 1.0:
            mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
            )
            last_hb = now

        # FIX 5: Drain all pending MAVLink messages into the cache to avoid lag
        while True:
            msg = master.recv_match(type=['ATTITUDE', 'LOCAL_POSITION_NED'], blocking=False)
            if not msg:
                break
            cache[msg.get_type()] = msg
            
        pos = cache.get("LOCAL_POSITION_NED")
        att = cache.get("ATTITUDE")

        # Read camera frame (this naturally blocks and dictates loop speed)
        ok, frame = cap.read()
        if not ok:
            print("No camera frame")
            time.sleep(0.05)
            continue
            
        # Estimate the marker position
        tvec = estimate_marker(frame, detector, K, dist, side=config["SIDE_LENGTH"], expected_marker_id=config["EXPECTED_MARKER_ID"]    )
        
        if tvec is None:
            print("No marker detected")
            continue

        if pos is None or att is None:
            print("PX4 pos/att not received yet — not sending")
            continue

        # Convert the marker position to body frame
        target_body = tuple(camera_to_frd(tvec))

        # Convert the marker position to NED frame
        dx, dy, dz = body_to_ned_offset(target_body, att.roll, att.pitch, att.yaw)

        # Get the target position in NED frame
        x_ned = pos.x + dx
        y_ned = pos.y + dy
        z_ned = pos.z + dz

        # Send the target position to the vehicle
        send_landing_target(mav, x_ned, y_ned, z_ned)

        print(
            f"vehicle NED=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})  "
            f"cam tvec=({tvec[0,0]:.2f}, {tvec[1,0]:.2f}, {tvec[2,0]:.2f})  "
            f"detected body=({target_body[0]:.2f}, {target_body[1]:.2f}, {target_body[2]:.2f})  "
            f"sent pad NED=({x_ned:.1f}, {y_ned:.1f}, {z_ned:.1f})  "
            f"diff=({pos.x - x_ned:.2f}, {pos.y - y_ned:.2f}, {pos.z - z_ned:.2f})"
        )

def main():
    # load configuration variables from config.yaml
    config = load_config()

    # connect to the mavlink
    master, mav = mavlink_connection(config["MAVLINK_CONNECTION_STRING"])
    cache = {}

    # set up the video capture
    cap = video_capture(config)
   
    # load the calibration parameters
    K, dist = calibration_parameters(config)

    # set up the aruco detector
    detector = aruco_detector(config)

    # start the main loop
    main_loop(config, master, mav, cache, cap, detector, K, dist)


if __name__ == "__main__":
    main()
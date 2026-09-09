"""
Run this script to perform pose estimation using ArUCo markers. The script detects ArUCo markers in the video stream, estimates their pose, and displays the results in real-time.
Usage:
    python detect_aruco.py --type <ARUCO_TAG_TYPE> --calibration_params <CALIBRATION_PARAMS_FILE>
Arguments:
    --type: Type of ArUCo tag to detect (default: DICT_4X4_1000). Supported types are defined in utils.py.
    --calibration_params: Path to the calibration parameters file (default: calibration_params.npz).
"""

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import os


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, side_length=0.05):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arucodict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucodict, arucoParams)


    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    frame = aruco_display(
        corners,
        ids,
        rejected_img_points,
        frame,
    )
    
    # If markers are detected
    if ids is not None and len(corners) > 0:
        marker_points = np.array([
            [-side_length / 2,  side_length / 2, 0],
            [ side_length / 2,  side_length / 2, 0],
            [ side_length / 2, -side_length / 2, 0],
            [-side_length / 2, -side_length / 2, 0],
        ], dtype=np.float32)

        for i, marker_corners in enumerate(corners):
            image_points = marker_corners.reshape(4, 2).astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                image_points,
                matrix_coefficients,
                distortion_coefficients
            )

            if not success:
                continue

            # cv2.aruco.drawDetectedMarkers(
            #     frame,
            #     [marker_corners],
            #     ids[i:i + 1],
            # )

            cv2.drawFrameAxes(
                frame,
                matrix_coefficients,
                distortion_coefficients,
                rvec,
                tvec,
                side_length * 0.5,
                4,
            )
            origin, _ = cv2.projectPoints(np.zeros((1, 3), np.float32), rvec, tvec, k, d)
            print(origin.ravel(), image_points.mean(axis=0))

            print(f"Marker ID: {ids[i]}")
            print(f"Position x, y, z: {tvec.ravel()}")
            print(f"Rotation vector: {rvec.ravel()}")
            distance = np.linalg.norm(tvec)
            print(f"Marker distance: {distance:.3f} m")

    return frame

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

    args = vars(ap.parse_args())

    if sys.argv[1:] == []:
        print("[ERROR] No arguments provided. Please specify the ArUCo tag type and calibration parameters file.")
        sys.exit(0)

    if args["type"] not in ARUCO_DICT.keys():
        print(f"[ERROR] ArUCo tag type '{args['type']}' is not supported.")
        sys.exit(0)
    if not os.path.exists(args["calibration_params"]):
        print(f"[ERROR] Calibration parameters file '{args['calibration_params']}' does not exist.")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_params_path = args["calibration_params"]

    calibration_params = np.load(calibration_params_path)
    # in the calibration_params.npz file, the camera matrix is stored under the key "camera_matrix" and the distortion coefficients under "dist_coeffs"
    k = calibration_params["camera_matrix"]
    d = calibration_params["dist_coeffs"]
    print(k)
    print(d)

    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        side = 0.1 # side length of the ArUco marker in meters
        output = pose_esitmation(frame, aruco_dict_type, k, d, side_length=side)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


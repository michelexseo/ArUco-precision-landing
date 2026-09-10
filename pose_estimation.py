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


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, side_length=0.05):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    dists - List of distances from the camera to each detected marker
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arucodict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucodict, arucoParams)

    dists = []

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
            # Reshape the marker corners to a 4x2 array and convert to float32 [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
            # The marker corners are in the order: top-left, top-right, bottom-right, bottom-left
            image_points = marker_corners.reshape(4, 2).astype(np.float32)

            # Solve for the rotation and translation vectors using solvePnP
            # rvec - Rotation vector (Rodrigues) that describes the rotation of the marker with respect to the camera
            # tvec - Translation vector that describes the position of the marker with respect to the camera
            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                image_points,
                matrix_coefficients,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
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
            # Calculate the origin of the marker in the camera coordinate system
            origin, _ = cv2.projectPoints(np.zeros((1, 3), np.float32), rvec, tvec, k, d)
            print(origin.ravel(), image_points.mean(axis=0))

            print(f"Marker ID: {ids[i]}")
            print(f"Position x, y, z: {tvec.ravel()}")
            print(f"Rotation vector: {rvec.ravel()}")
            distance = np.linalg.norm(tvec)
            print(f"Marker distance: {distance:.3f} m")
            dists.append(distance)

           

    return frame, dists

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

    # if sys.argv[1:] == []:
    #     print("[ERROR] No arguments provided. Please specify the ArUCo tag type and calibration parameters file.")
    #     sys.exit(0)

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
    # print(k)
    # print(d)

    side_length = args["side_length"]

    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
            
        output, dists = pose_estimation(frame, aruco_dict_type, k, d, side_length)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # to save the frame to disk when the spacebar is pressed
        if key == 32:  # SPACE
            # save the frame to disk
            filename = "pose_estimation" + str(int(time.time())) + ".png"
            cv2.imwrite(filename, output)
            print(f"[INFO] saved {filename} to disk.")
    mean = np.mean(dists) if dists else 0
    print(f"Mean distance to detected markers: {mean:.3f} m")
    

    video.release()
    cv2.destroyAllWindows()


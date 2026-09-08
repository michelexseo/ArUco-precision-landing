import cv2
import os
import numpy as np
import argparse
import time
import sys


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

def detect(args):
    # verify that the supplied ArUCo tag exists and is supported by
    # OpenCV
    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            args["type"]))
        sys.exit(0)

    # load the ArUCo dictionary and grab the ArUCo parameters
    print(f"[INFO] detecting '{args['type']}' tags...")
    arucodict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucodict, arucoParams)

    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    counter = 0

    while True:
        ret, frame = vs.read()
        if not ret or frame is None:
            break
        frame = vs.read()[1]
        print("[DEBG]: detecting ArUCo markers in the frame...")
        # detect ArUCo markers in the input frame
        # corners is a list of corners  for each detected marker, ids is the list of ids for each detected marker, and rejected is a list of rejected candidates
        (corners, ids, rejected) = detector.detectMarkers(frame)

        if (ids is not None and len(corners) > 0):
            print("[DEBUG]: detected {} ArUCo markers in the frame.".format(len(ids)))
            print("[DEBUG]: ArUco maerker corners: {}".format(corners))
            # flatten the ArUCo IDs list and draw the bounding box of the ArUCo detections
            ids = ids.flatten()

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if key == ord("s"):
            # save the frame to disk
            filename = "aruco_detected" + str(counter) + ".png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] saved {filename} to disk.")
            counter += 1
      
            
    vs.release()
    cv2.destroyAllWindows()
    # do a bit of cleanup
    cv2.destroyAllWindows()  

                
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str,
        default="DICT_ARUCO_ORIGINAL",
        help="type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    detect(args)

if __name__ == "__main__":
    main()
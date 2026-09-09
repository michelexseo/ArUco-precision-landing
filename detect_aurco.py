import cv2
import os
import numpy as np
import argparse
import time
import sys
from utils import ARUCO_DICT
from utils import aruco_display

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
        # print("[DEBG]: detecting ArUCo markers in the frame...")
        # detect ArUCo markers in the input frame
        # corners is a list of corners  for each detected marker, ids is the list of ids for each detected marker, and rejected is a list of rejected candidates
        (corners, ids, rejected) = detector.detectMarkers(frame)

        if (ids is not None and len(corners) > 0):
            # print("[DEBUG]: detected {} ArUCo markers in the frame.".format(len(ids)))
            # print("[DEBUG]: ArUco maerker corners: {}".format(corners))
            # flatten the ArUCo IDs list and draw the bounding box of the ArUCo detections
            ids = ids.flatten()

            # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            frame = aruco_display(corners, ids, rejected, frame)

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
        default="DICT_4X4_1000",
        help="type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    detect(args)

if __name__ == "__main__":
    main()
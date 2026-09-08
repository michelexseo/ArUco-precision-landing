import cv2

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

def aruco_display(corners, ids, rejected, image):
    if ids is None or len(corners) == 0:
        return image

    ids = ids.flatten()

    for marker_corner, marker_id in zip(corners, ids):
        points = marker_corner.reshape((4, 2)).astype(int)
        top_left, top_right, bottom_right, bottom_left = points

        # Thick bounding box
        cv2.polylines(
            image,
            [points.reshape((-1, 1, 2))],
            isClosed=True,
            color=(0, 255, 0),
            thickness=5,
            lineType=cv2.LINE_AA,
        )

        # Large center point
        center_x = int((top_left[0] + bottom_right[0]) / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2)

        cv2.circle(
            image,
            (center_x, center_y),
            8,
            (0, 0, 255),
            -1,
        )

        # Large ID text
        text = f"ID: {marker_id}"
        text_x = max(10, top_left[0])
        text_y = max(45, top_left[1] - 15)

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,              # font size
            (0, 255, 255),    # yellow
            4,                # text thickness
            cv2.LINE_AA,
        )

        print(f"[Inference] ArUCo marker ID: {marker_id}")

    return image
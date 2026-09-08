import cv2 as cv
import glob
import os
import shutil
import sys
import numpy as np


def capture_images():
    output_dir = "calibration_images"
    if os.path.exists(output_dir):
        ans = input(f"Do you want to delete content of '{output_dir}'? (y/n): ")
        if ans.lower() == "y":
            shutil.rmtree(output_dir)
        else:
            print("Keeping existing images and exiting capture mode.")
            return

    os.makedirs(output_dir, exist_ok=True)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    captured_count = 0
    max_images = 30

    print("--- Camera Calibration Capture ---")
    print("Press 'SPACE' to capture an image.")
    print("Press 'q' or 'ESC' to quit early.")

    while captured_count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        display_frame = frame.copy()
        cv.putText(
            display_frame,
            f"Captured: {captured_count}/{max_images}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv.imshow("Calibration Capture - Press SPACE to take photo", display_frame)

        key = cv.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            file_path = os.path.join(output_dir, f"calib_img_{captured_count:02d}.jpg")
            cv.imwrite(file_path, frame)
            print(f"[{captured_count + 1}/{max_images}] Saved {file_path}")
            captured_count += 1
        elif key == ord("q") or key == 27:  # ESC or q
            print("Capture aborted by user.")
            break

    cap.release()
    cv.destroyAllWindows()


def run_calibration(objpoints, imgpoints, image_size):
    """Calibrate camera using accumulated 3D and 2D points."""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("\n--- Calibration Results ---")
    print(f"RMS Re-projection Error: {ret:.4f}")
    print("Camera Matrix (Intrinsic Parameters):\n", camera_matrix)
    print("Distortion Coefficients [k1, k2, p1, p2, k3]:\n", dist_coeffs.ravel())

    # Optional: Save calibration values to disk for later use
    np.savez(
        "camera_calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    print("Saved calibration parameters to 'camera_calibration.npz'")

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def calibrate(should_capture=False):
    pattern_size = (9, 6)  # (columns, rows) of internal corners
    square_size = 0.025       # Adjust if you want metric scale (e.g. 0.025 for 25mm)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D points in real-world grid: (0,0,0), (1,0,0), ..., (6,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    image_size = None

    if should_capture:
        capture_images()

    image_paths = sorted(glob.glob("calibration_images/*.jpg"))
    if not image_paths:
        print("Error: No images found in 'calibration_images/'")
        return

    print(f"\nProcessing {len(image_paths)} images for corners...")

    for fname in image_paths:
        img = cv.imread(fname)
        if img is None:
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if image_size is None:
            # (width, height) required by cv.calibrateCamera
            image_size = gray.shape[::-1]

        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Visualize detections
            cv.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv.imshow("Chessboard Detection", img)
            cv.waitKey(5000)
        else:
            print(f"Skipping {fname}: chessboard corners not found.")

    cv.destroyAllWindows()

    # Step final: Run calibration only if we have collected enough valid frames
    if len(objpoints) < 10:
        print(f"Warning: Only {len(objpoints)} valid images found. For accurate results, at least 15-20 are recommended.")

    if len(objpoints) > 0:
        run_calibration(objpoints, imgpoints, image_size)
    else:
        print("Error: No chessboard patterns were detected in any image.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibration.py <CAPTURE_IMAGES: true/false>")
        sys.exit(1)

    should_capture = sys.argv[1].lower() in ["true", "1", "yes", "y"]
    calibrate(should_capture)


if __name__ == "__main__":
    main()

"""
Script to calibrate a camera using chessboard images. It captures images from a webcam, detects chessboard corners, computes camera intrinsics and distortion coefficients, evaluates reprojection errors, and provides a live undistortion preview.
Usage:
    python calibration.py <CAPTURE_IMAGES: true/false>
Arguments:
    <CAPTURE_IMAGES: true/false> - Whether to capture new calibration images from the

"""

import cv2 as cv
import glob
import os
import shutil
import sys
import numpy as np


def capture_images(output_dir="calibration_images", max_images=30):
    existing_count = 0

    if os.path.exists(output_dir):
        ans = input(f"Do you want to delete content of '{output_dir}'? (y/n): ")
        if ans.lower() == "y":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            existing_count = 0
        else:
            existing_files = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
            existing_count = len(existing_files)
            print(f"Found {existing_count} existing calibration image(s) in '{output_dir}'.")

            if existing_count >= max_images:
                print(f"Enough images already exist ({max_images}); reusing them and skipping capture.")
                return

            print(f"Capturing {max_images - existing_count} additional image(s) to reach {max_images}.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    print("--- Camera Calibration Capture ---")
    print("Press 'SPACE' to capture an image.")
    print("Press 'q' or 'ESC' to quit early.")

    captured_count = existing_count
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
        elif key in (ord("q"), 27):
            print("Capture stopped by user.")
            break

    cap.release()
    cv.destroyAllWindows()

def evaluate_per_image_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs, valid_fnames):
    """Computes and displays re-projection error for each image individually."""
    errors = []
    print("\n--- Per-Image Re-projection Errors ---")
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        
        # Flatten both to shape (N, 2) to ensure matching float32 arrays
        pts_detected = imgpoints[i].reshape(-1, 2)
        pts_projected = imgpoints2.reshape(-1, 2)
        
        # Calculate root-mean-square Euclidean distance using NumPy
        error = np.sqrt(np.mean(np.sum((pts_detected - pts_projected) ** 2, axis=1)))
        
        file_basename = os.path.basename(valid_fnames[i])
        errors.append((error, file_basename))
        print(f"  {file_basename}: {error:.4f} px")

    # Sort descending to isolate the worst outliers
    errors.sort(key=lambda item: item[0], reverse=True)
    print("\nTop Outliers (Worst Candidates to remove if error is high):")
    for err, name in errors[:5]:
        print(f"  {name}: {err:.4f} px")

    return errors

def live_undistortion_preview(camera_matrix, dist_coeffs):
    """Streams live video side-by-side: original vs undistorted."""
    print("\n--- Starting Live Undistortion Verification ---")
    print("Point the camera at straight edges (e.g. door frames, monitors).")
    print("Press 'q' or 'ESC' to exit preview.")

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv.undistort(frame, camera_matrix, dist_coeffs)

        # Label frames for comparison
        cv.putText(frame, "Original (Distorted)", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(undistorted, "Calibrated (Undistorted)", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.hstack((frame, undistorted))
        cv.imshow("Side-by-Side Verification", combined)

        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv.destroyAllWindows()


def run_calibration(objpoints, imgpoints, image_size, valid_fnames, output_file="camera_calibration.npz"):
    """Calibrate camera, evaluate reprojection errors, and save parameters."""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("\n================ Calibration Results ================")
    print(f"Global RMS Re-projection Error: {ret:.4f} px")
    print("\nCamera Matrix (Intrinsics):\n", camera_matrix)
    print("\nDistortion Coefficients [k1, k2, p1, p2, k3]:\n", dist_coeffs.ravel())

    evaluate_per_image_error(
        objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs, valid_fnames
    )

    output_path = os.path.abspath(output_file)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez(
        output_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    print(f"\n[OK] Saved calibration parameters to '{output_path}'")

    return camera_matrix, dist_coeffs


def calibrate(should_capture=False, output_file="camera_calibration.npz"):
    pattern_size = (9, 6)  # Inner corners (cols, rows)
    square_size = 0.025    # Real size of one chessboard square in meters (e.g., 25mm)
    debug_dir = "calibration_debug"

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare 3D object points in target coordinate frame: (0,0,0), (1,0,0), ...
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    if should_capture:
        capture_images()

    image_paths = sorted(glob.glob("calibration_images/*.jpg"))[:30]
    if not image_paths:
        print("Error: No images found in 'calibration_images/'")
        return

    os.makedirs(debug_dir, exist_ok=True)

    objpoints = []
    imgpoints = []
    valid_fnames = []
    image_size = None

    print(f"\nProcessing {len(image_paths)} images for corner detection...")

    for fname in image_paths:
        img = cv.imread(fname)
        if img is None:
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_fnames.append(fname)

            # Draw and save annotated frame to calibration_debug/
            debug_img = img.copy()
            cv.drawChessboardCorners(debug_img, pattern_size, corners2, ret)
            cv.imwrite(os.path.join(debug_dir, f"detected_{os.path.basename(fname)}"), debug_img)

            # Quick display during iteration
            cv.imshow("Chessboard Detection", debug_img)
            cv.waitKey(100)
        else:
            print(f"Skipping {os.path.basename(fname)}: chessboard corners not found.")

    cv.destroyAllWindows()
    print(f"\nSaved {len(valid_fnames)} detection overlays to './{debug_dir}/'")

    if len(objpoints) < 10:
        print(f"Warning: Only {len(objpoints)} valid images found. Consider collecting at least 15-20.")

    if len(objpoints) == 0:
        print("Error: No valid corners found in any image.")
        return

    # Run solver and inspect individual image errors
    camera_matrix, dist_coeffs = run_calibration(
        objpoints,
        imgpoints,
        image_size,
        valid_fnames,
        output_file=output_file,
    )

    # Launch live video check
    live_undistortion_preview(camera_matrix, dist_coeffs)


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibration.py <CAPTURE_IMAGES: true/false>")
        sys.exit(1)

    should_capture = sys.argv[1].lower() in ["true", "1", "yes", "y"]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "camera_calibration.npz"
    calibrate(should_capture, output_file=output_file)


if __name__ == "__main__":
    main()
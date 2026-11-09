# ArUco Precision Landing

Camera calibration and ArUco marker pose estimation for precision-landing
experiments. The project can also send marker estimates to a flight controller
over MAVLink using `LANDING_TARGET` messages. It does not start a landing
mission or configure the flight controller's precision-landing behavior.

## Features

- Calibrate camera intrinsics and distortion using chessboard images.
- Detect ArUco markers and estimate their pose relative to the camera.
- Optionally stream pose estimates and telemetry logs over MAVLink.
- Capture and evaluate repeatable pose-estimation benchmark data.

## Requirements

- Python
- A camera for live capture
- A printed chessboard for calibration and a printed ArUco marker for tracking
- MAVLink Router and a configured flight-controller connection for MAVLink use

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate it, then install the project dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## Quick start

### 1. Calibrate the camera

Capture chessboard images with a connected camera:

```bash
python calibration.py true
```

Or calibrate using images already in `calibration_images/`:

```bash
python calibration.py false
```

Calibration parameters are saved to `camera_calibration.npz` by default. Pass
an optional output path as the second argument to choose a different location.
The calibration script uses a 9-by-6 inner-corner chessboard; the printed
pattern's square dimensions must match the values used by the script.

### 2. Detect a marker

```bash
python detect_aruco.py --type DICT_4X4_1000
```

The detection script opens the camera feed. Press `q` to exit.

### 3. Estimate marker pose

```bash
python pose_estimation.py --type DICT_4X4_1000 --calibration_params camera_calibration.npz --side_length 0.1
```

`--side_length` is the black marker's side length in meters; measure the
printed marker rather than including its white margin. Press `q` to exit.
Press the space bar to save a pose-estimation frame in the project directory.

## Images and marker

The sample images and printable marker are in [`imgs/`](imgs/):

![Chessboard calibration pattern](imgs/pattern_chessboard.png)

![ArUco marker, dictionary 4x4_1000, ID 0](imgs/aruco_4x4_1000-0.svg)

Example marker-detection output:

![Detected ArUco marker](imgs/aruco_detected0.png)

Example pose-estimation output:

![Pose estimation example](imgs/pose_estimation1788951384.png)
![Pose estimation example](imgs/pose_estimation1788951391.png)

Calibration input and debug examples are in `calibration_images/` and
`calibration_debug/`.

## Send estimates over MAVLink

`send_pose_estimation.py` reads camera, marker, and connection settings from
[`config.yaml`](config.yaml). Before running it:

1. Set `CALIBRATION_PARAMS` to the camera's calibration file.
2. Set `SIDE_LENGTH` to the printed marker's measured black-square width, in
   meters, and `EXPECTED_MARKER_ID` to the marker's ID. The bundled marker is
   ID 0; the sample configuration defaults to ID 1.
3. Configure `MAVLINK_CONNECTION_STRING` to match the MAVLink Router endpoint.
   The sample value is `udpin:127.0.0.1:14540`.
4. Configure the router to forward flight-controller traffic to that endpoint.

Install MAVLink Router on the Linux companion computer. On Debian- or
Ubuntu-based systems:

```bash
sudo apt update
sudo apt install mavlink-router
```

Configure its input and output endpoints in
`/etc/mavlink-router/main.conf`, following the
[MAVLink Router documentation](https://github.com/mavlink-router/mavlink-router).
Start the router, then run:

```bash
python send_pose_estimation.py
```

Verify that the flight controller receives valid estimates and configure its
precision-landing parameters and mission separately, using documentation for
the flight-controller version. Test in a safe environment and follow the
vehicle manufacturer's safety procedures.

## Benchmarking

Capture benchmark frames with a connected camera:

```bash
python benchmark_capture.py
```

Evaluate the captured data using a calibration file:

```bash
python benchmark_eval.py --calibration_params camera_calibration.npz
```

See each script's `--help` output for additional options. Benchmark results and
runtime logs are written to `benchmark/` and `logs/`, respectively, and are
excluded from version control.

## Logging

For details on MAVLink runtime logs, CSV fields, and saved debug frames, see
[`LOGGING.md`](LOGGING.md).

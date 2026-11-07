# Precision Landing

## Camera Calibration

In order to get the accurate distance and projection of a 3D point using a camera we need first to calibrate the camera. The process returns a Reprojection Error which is a measure of how well the calibration process has worked. A lower reprojection error indicates a better calibration. A value of less than 0.5 is generally considered good, but this can vary depending on the specific application and requirements. The calibration process also returns the camera matrix and distortion coefficients which are used in the pose estimation process.
After calibration, the camera matrix and distortion coefficients are saved to a file (camera_calibration.npz) which is then used in the pose estimation process. The calibration process can be run by executing the calibration.py script with the argument true or false to indicate whether to capture new calibration images or not. The script will capture images of a chessboard pattern and use them to calibrate the camera.

```bash 
python calibration.py true "output_file.npz"  # Capture new images and save calibration parameters to output_file.npz
```
```bash 
python calibration.py false "output_file.npz" # Use existing images to calibrate and save calibration parameters to output_file.npz
```

***Note***: If no output file is specified, the calibration parameters will be saved to camera_calibration.npz by default.

***Note***: The calibration process requires a chessboard pattern to be printed and used for capturing images. The chessboard pattern should have a known number of inner corners (e.g., 9x6) and the size of each square should be known (e.g., 0.025 meters). 

The process should generate a set of calibration images like the following ones:
<div style="display: flex; gap: 10px;">
  <img src="calibration_images/calib_img_00.jpg" alt="Example of calibration images" style="width: 50%; height: auto;" />
  <img src="calibration_debug/detected_calib_img_00.jpg" alt="Example of calibration images" style="width: 50%; height: auto;" />
</div>


# Marker Detection

our goal is to detect ArUco markers in the video stream and estimate their pose relative to the camera. The pose estimation process uses the camera matrix and distortion coefficients obtained from the calibration process to accurately determine the position and orientation of the detected markers in 3D space. The estimated pose can be used for various applications, such as augmented reality, robotics, and precision landing.

First we need a marker. We can genrate one online at [https://chev.me/arucogen/](https://chev.me/arucogen/) or we can use the one provided in the repository. The marker is a 4x4 ArUco marker with ID 0. The marker should be printed and placed (on a flat surface) in the environment where the camera will be used for detection.

***Note***: Always keep track of the size of the marker. The size of the marker is used in the pose estimation process to accurately determine the distance from the camera to the marker.if need check with a ruler the size of the marker measuring the length of the black square's side.

To try the marker detection run the detect_aruco.py script. The script will open a video stream from the camera and detect the ArUco marker in real-time. The detected marker will be highlighted with a bounding box and its ID will be displayed on the screen.

```bash
python detect_aruco.py [<AURCO_DICT_TYPE>] # Optional argument to specify the ArUco dictionary type (default: DICT_4X4_100)
```

in utils.py you can find a list of all the available ArUco dictionary types. The default dictionary type is DICT_4X4_100 which is the one used to generate the marker provided in the repository. 
The file also contains a function to display the detected markers in the video stream in a more accessible and readble way.

Example of the output of the detect_aruco.py script:
![Example of the output of the detect_aruco.py script](aruco_detected0.png)


## Pose Estimation

To estimate the pose of the detected ArUco marker, we can use the pose_estimation.py script. The script will open a video stream from the camera and detect the ArUco marker in real-time. The estimated pose of the marker will be displayed on the screen along with the distance from the camera to the marker.

```bash
python pose_estimation.py -t [<AURCO_DICT_TYPE>] -c [<CALIBRATION_FILE>] -s [<MARKER_SIZE>]
```

where:
- `<AURCO_DICT_TYPE>`: Optional argument to specify the ArUco dictionary type (default: DICT_4X4_100)
- `<CALIBRATION_FILE>`: Optional argument to specify the calibration file (default: camera_calibration.npz)
- `<MARKER_SIZE>`: Optional argument to specify the size of the marker in meters (default: 0.1)

You should obtain something like this:
![Example of the output of the pose_estimation.py script](./pose_estimation1788951384.png)
![Example of the output of the pose_estimation.py script](./pose_estimation1788951391.png)


## Running with MAVLink Router

MAVLink Router can forward telemetry between the flight controller and the
companion running this application. The application uses the camera to estimate
the marker's position and sends it to PX4 in `LANDING_TARGET` MAVLink messages;
it also receives vehicle position, attitude, and heartbeat messages from PX4.
The router provides the communication link, but does not start a landing
mission or configure PX4 to use precision-landing data. Configure the relevant
PX4 precision-landing settings and mission or flight mode separately, following
the documentation for your PX4 version.

The basic setup is:

1. Connect the flight controller to the companion running this application,
   using a supported serial or network link.
2. Configure MAVLink Router to receive MAVLink from the flight controller and
   forward traffic to the application's UDP endpoint.
3. Set `MAVLINK_CONNECTION_STRING` in `config.yaml` to match that endpoint,
   then run `send_pose_estimation.py`.
4. Verify that PX4 is receiving valid marker estimates and that its precision
   landing configuration is correct before attempting a landing.

Test the complete setup in a safe environment and follow the PX4 and vehicle
manufacturer's safety procedures.

Install MAVLink Router on the Linux system that will run the pose estimation
script. On Debian- or Ubuntu-based systems, install the distribution package:

```bash
sudo apt update
sudo apt install mavlink-router
```

If the package is not available for your distribution, follow the installation
instructions in the [MAVLink Router project](https://github.com/mavlink-router/mavlink-router).

Configure the router using its system configuration file, usually
`/etc/mavlink-router/main.conf`:

```bash
sudoedit /etc/mavlink-router/main.conf
```

Set up an input endpoint for the flight controller (for example, its serial
device or UDP connection) and an output endpoint for this application. The
output must match `MAVLINK_CONNECTION_STRING` in `config.yaml`; its default is
`udpin:127.0.0.1:14540`. Adjust the application setting or router endpoint if
your network setup uses a different address or port. Refer to the MAVLink
Router documentation for endpoint options and configuration syntax.

Start MAVLink Router. If your installation provides a systemd service, start
it and check its status:

```bash
sudo systemctl enable --now mavlink-router
sudo systemctl status mavlink-router
```

Otherwise, run the router directly using its system configuration:

```bash
sudo mavlink-routerd -c /etc/mavlink-router/main.conf
```

Once the router is running, activate the Python environment with the project
dependencies and start pose estimation:

```bash
source ~/venv/bin/activate
python send_pose_estimation.py
```

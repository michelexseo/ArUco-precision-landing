# Precision Landing — Logging Guide

This document describes the logging system used by `send_poste_estimation.py` during bench tests and real drone flights. The goal is to record enough data to answer, after a run:

- Was the ArUco marker detected reliably?
- What position did the camera estimate?
- What did we send to PX4?
- What was the vehicle state at that moment?
- Why did sending stop (no marker, no camera, no MAVLink data)?

---

## Overview

Logging is split into **three outputs**, all keyed by the same `run_id` (UTC timestamp, e.g. `20260916_093407`):

| Output | File / folder | Purpose |
|--------|----------------|---------|
| **Text log** | `logs/run_{run_id}.log` | Human-readable events, summaries, connection status |
| **CSV data log** | `logs/data_{run_id}.csv` | One row per loop — for spreadsheets and plotting |
| **Debug frames** | `logs/frames_{run_id}/` | JPEG snapshots when marker is detected (optional) |

On the Raspberry Pi you also see text output in the SSH terminal (`StreamHandler`).

```
send_poste_estimation.py
        │
        ├── setup_logging()  ──► logs/run_{run_id}.log  (+ terminal)
        │
        ├── FlightLogger     ──► logs/data_{run_id}.csv
        │                      └── logs/frames_{run_id}/frame_*.jpg
        │
        └── main_loop()      ──► calls logger.log_frame() every loop (or filtered)
```

---

## Configuration (`config.yaml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `LOG_DIR` | `"logs"` | Directory for all log files |
| `LOG_SUMMARY_INTERVAL_SEC` | `5` | How often to print/write a SUMMARY line (seconds) |
| `LOG_EVERY_FRAME` | `true` | If `false`, CSV only logs detections, sends, and `no_pos_att` skips |
| `LOG_SAVE_FRAMES` | `true` | Save JPEG snapshots when marker is detected |
| `LOG_FRAME_INTERVAL` | `30` | Save one frame every N **logged loops** while `detected=1` |
| `LOG_MAVLINK_MESSAGES` | `[HEARTBEAT, SYS_STATUS]` | Extra MAVLink messages to read from PX4 (ATTITUDE and LOCAL_POSITION_NED are always included) |

Example:

```yaml
LOG_DIR: "logs"
LOG_SUMMARY_INTERVAL_SEC: 5
LOG_EVERY_FRAME: true
LOG_SAVE_FRAMES: true
LOG_FRAME_INTERVAL: 30
LOG_MAVLINK_MESSAGES:
  - HEARTBEAT
  - SYS_STATUS
```

---

## How logging is implemented

### 1. Startup — `setup_logging()` in `send_poste_estimation.py`

- Creates `LOG_DIR` if missing.
- Generates `run_id` (UTC, `YYYYMMDD_HHMMSS`).
- Configures Python `logging` with:
  - **FileHandler** → `logs/run_{run_id}.log`
  - **StreamHandler** → terminal
- Returns `run_id` so the CSV and text log share the same ID.

### 2. Data logger — `FlightLogger` in `utils.py`

Instantiated in `main()`:

```python
logger = FlightLogger(
    log_dir,
    summary_interval,
    run_id=run_id,
    save_frames=config.get("LOG_SAVE_FRAMES", False),
    frame_interval=config.get("LOG_FRAME_INTERVAL", 30),
)
```

Each loop calls `record_frame()` → `logger.log_frame()` with plain Python values (tuples, floats), not raw pymavlink objects.

On exit (`Ctrl+C` or crash after `finally`), `logger.close()` writes a final SUMMARY and closes the CSV.

### 3. Main loop — what gets logged when

Every iteration of `main_loop()` follows this path:

```
Read MAVLink  →  Read camera  →  Detect marker  →  Compute NED  →  Send LANDING_TARGET
                      │                │                  │
                      ▼                ▼                  ▼
                 skip: no_frame   skip: no_marker   skip: no_pos_att   success (sent=1)
```

| Condition | `detected` | `sent` | `skip_reason` |
|-----------|------------|--------|---------------|
| Camera read failed | 0 | 0 | `no_frame` |
| No matching ArUco | 0 | 0 | `no_marker` |
| Marker seen, no pos/att from PX4 | 1 | 0 | `no_pos_att` |
| Marker seen, pos/att OK, LANDING_TARGET sent | 1 | 1 | *(empty)* |

Whether a row is written also depends on `LOG_EVERY_FRAME` (see below).

### 4. Debug frame snapshots

When `LOG_SAVE_FRAMES: true`, inside `FlightLogger.log_frame()`:

```python
if detected and frame is not None and self.loops % LOG_FRAME_INTERVAL == 0:
    cv2.imwrite(f"logs/frames_{run_id}/frame_{run_id}_{loops:06d}.jpg", frame)
```

- Only runs when the marker **was detected** on that loop.
- Uses **`self.loops`** (total logged rows), not camera FPS.
- With `LOG_FRAME_INTERVAL: 30`, roughly every 30th **logged** detection saves a JPEG.
- Frames are **raw camera images** (no overlay). Match them to CSV rows via the loop number in the filename.

At ~30 loop/s, interval 30 ≈ one snapshot per second while continuously detecting.

---

## CSV columns — `logs/data_{run_id}.csv`

Each row is one main-loop iteration (when logging is enabled for that iteration).

| Column | Type | Meaning |
|--------|------|---------|
| `timestamp` | ISO 8601 UTC | When the row was written |
| `loop_ms` | ms | Time for that loop iteration (camera + processing) |
| `detected` | 0/1 | ArUco marker with `EXPECTED_MARKER_ID` was found |
| `tvec_x`, `tvec_y`, `tvec_z` | m | Marker position in **camera frame** (OpenCV: x right, y down, z forward) from `solvePnP` |
| `dist_m` | m | Euclidean distance to marker: √(tvec_x² + tvec_y² + tvec_z²) |
| `body_x`, `body_y`, `body_z` | m | Marker in **body FRD** after `camera_to_frd()` |
| `veh_x`, `veh_y`, `veh_z` | m | Vehicle position in **local NED** from PX4 `LOCAL_POSITION_NED` |
| `roll`, `pitch`, `yaw` | rad | Vehicle attitude from PX4 `ATTITUDE` |
| `armed` | 0/1 | From MAVLink `HEARTBEAT` (`MAV_MODE_FLAG_SAFETY_ARMED`) |
| `flight_mode` | int | PX4 `custom_mode` from `HEARTBEAT` (PX4-specific mode number) |
| `pad_x`, `pad_y`, `pad_z` | m | **Landing pad position** sent in NED (= vehicle + body offset rotated to NED) |
| `diff_x`, `diff_y`, `diff_z` | m | `vehicle_NED − pad_NED` (should relate to marker offset; useful for sanity checks) |
| `sent` | 0/1 | `LANDING_TARGET` MAVLink message was sent this loop |
| `skip_reason` | string | Why nothing was sent: `no_frame`, `no_marker`, `no_pos_att`, or empty if sent |

Empty cells mean that quantity was not available (e.g. no marker → no `tvec_*`).

### Coordinate reminder

- **Camera tvec**: OpenCV optical frame.
- **body_***: FRD (forward, right, down) on the drone.
- **veh_*, pad_*, diff_***: Local NED origin as reported by PX4 (same frame as `LANDING_TARGET`).

---

## Text log — `logs/run_{run_id}.log`

### Startup lines

```
Starting precision landing (run_id=20260916_093407)
Sending initial heartbeat to register with router...
Waiting for heartbeat from PX4...
Connected to PX4 (sys=1 comp=0)
Requesting ATTITUDE and LOCAL_POSITION_NED at 10 Hz...
```

### SUMMARY lines (every `LOG_SUMMARY_INTERVAL_SEC`)

```
SUMMARY loops=567 detections=120 rate=21.2% sent=115 pos_ok=567 (100.0%) att_ok=567 (100.0%) no_frame=0 no_marker=447 no_pos_att=5 frames_saved=4
```

| Field | Meaning |
|-------|---------|
| `loops` | Total rows written to CSV so far |
| `detections` | Loops where marker was detected |
| `rate` | `detections / loops × 100` — **detection rate** |
| `sent` | Loops where `LANDING_TARGET` was sent |
| `pos_ok` / `att_ok` | Loops where vehicle position / attitude was available |
| `no_frame` | Camera read failures |
| `no_marker` | No ArUco (or wrong ID) |
| `no_pos_att` | Marker seen but PX4 data missing |
| `frames_saved` | Debug JPEGs written |

### Per-send lines (only when `sent=1`)

```
sent pad NED=(0.0, 0.0, 0.3) veh NED=(0.0, 0.0, 0.0) tvec=(0.01, -0.02, 0.28) diff=(-0.02, -0.02, -0.28)
```

Useful for live monitoring over SSH without opening the CSV.

### Frame save lines

```
Saved debug frame frame_20260916_093407_000030.jpg
```

### Shutdown

```
Stopped by user
SUMMARY loops=2090 ...
CSV log saved to logs/data_20260916_093407.csv
Debug frames saved to logs/frames_20260916_093407
```

---

## `LOG_EVERY_FRAME` behavior

| Setting | CSV rows |
|---------|----------|
| `true` | Every loop, including `no_marker` (large files, full detection-rate analysis) |
| `false` | Only when marker detected, LANDING_TARGET sent, or `no_pos_att` (smaller files) |

Summaries always count all loops that **were** logged.

---

## Analyzing a flight

### Quick check in the terminal

Watch SUMMARY lines while running:

- **`rate` near 0%** → marker not visible, wrong ID, wrong dictionary, or bad lighting.
- **`pos_ok` / `att_ok` below 100%** → MAVLink link or stream issue.
- **`no_pos_att` > 0 with detections** → marker seen but PX4 not publishing position/attitude yet.
- **`sent` < `detections`** → usually `no_pos_att` gaps.

### Spreadsheet / pandas

Load CSV and plot vs time:

- `dist_m` — marker distance over time
- `pad_x`, `pad_y`, `pad_z` — what you commanded
- `detected` — binary or rolling mean for detection rate
- `loop_ms` — performance / lag

Example (Python):

```python
import pandas as pd
df = pd.read_csv("logs/data_20260916_093407.csv", parse_dates=["timestamp"])
print(f"Detection rate: {df['detected'].mean():.1%}")
print(f"Send rate: {df['sent'].mean():.1%}")
df[df["detected"] == 1].plot(x="timestamp", y="dist_m")
```

### Correlate frames with CSV

Filename `frame_20260916_093407_000030.jpg` → loop **30** in that run. Find the CSV row where cumulative loop count matches (row index ≈ loop number if `LOG_EVERY_FRAME=true`).

---

## File sizes and Pi storage

- **CSV with `LOG_EVERY_FRAME: true`** at ~25–30 Hz → ~2000 rows/minute → plan disk space for long flights.
- **Debug frames** at 640×480 JPEG → ~50–150 KB each. Interval 30 at continuous detection ≈ 1 image/sec.
- For long autonomous runs, consider:
  - `LOG_EVERY_FRAME: false`
  - `LOG_SAVE_FRAMES: false` or larger `LOG_FRAME_INTERVAL`

---

## Code map

| File | Role |
|------|------|
| `send_poste_estimation.py` | `setup_logging()`, `record_frame()`, MAVLink → tuple conversion, passes `frame` to logger |
| `utils.py` | `FlightLogger` — CSV writer, summaries, frame snapshots |
| `config.yaml` | Log paths and intervals |

Key functions:

```text
main()
  setup_logging(log_dir)           → run_id, text log handlers
  FlightLogger(..., run_id=...)    → CSV + optional frames dir
  main_loop(..., logger)
    record_frame(logger, ...)      → respects LOG_EVERY_FRAME
      logger.log_frame(...)        → CSV row, summary timer, frame save, send log line
  logger.close()                   → final summary, close CSV
```

---

## What is *not* logged (yet)

- Raw `SYS_STATUS` fields (message is received if listed in config, but not written to CSV).
- Full MAVLink `.tlog` (use mavlink-router `Log = ...` in `/etc/mavlink-router/main.conf` for that).
- Annotated frames (bounding boxes) — snapshots are raw camera images only.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Empty `logs/` | Script not started from project dir, or `LOG_DIR` path wrong |
| CSV exists, no text log | `setup_logging()` not called before other log calls |
| `frames_saved=0` | `LOG_SAVE_FRAMES: false`, or no detections, or interval never hit |
| All rows `no_marker` | Marker not in view, wrong `EXPECTED_MARKER_ID` or `ARUCO_DICT_TYPE` |
| Detections but `sent=0` | All rows `no_pos_att` — wait for EKF/GPS local position |
| Huge log files | `LOG_EVERY_FRAME: true` on long runs — set to `false` or reduce flight time |

---

## Related files after a run

Example layout:

```text
logs/
  run_20260916_093407.log
  data_20260916_093407.csv
  frames_20260916_093407/
    frame_20260916_093407_000030.jpg
    frame_20260916_093407_000060.jpg
    ...
```

Keep `run_id` matching across all three for a single flight session.

# Doll Head–Torso — Full Documentation (EN)

Detect whether a doll is **assembled** (head + torso) or **disassembled**, and publish:
- a **state**: `assembled | disassembled | unknown`
- per-part **2D** info (bbox, centroid, area) and **3D** coordinates (XYZ in camera frame) for classes: `head`, `torso`, `ht` (assembled).

**Camera:** Intel RealSense D456  
**Model:** YOLOv8 segmentation (3 classes: `head`, `torso`, `ht`)  
**ROS nodes:** `yolo_seg_node.py` (visual debug) and `doll_state_node.py` (state + coordinates)

---

## 0) Quickstart (TL;DR)

```bash
# clone
git clone git@github.com:willito05/doll_head_torso.git
cd doll_head_torso

# download weights (dataset is optional and OFF by default)
./scripts/download_assets.sh

# build inside catkin workspace
cd ~/catkin_ws/src && ln -s $(pwd) .
cd .. && catkin_make && source devel/setup.bash

# run (use your launch name)
roslaunch doll_head_torso doll_state_cord.launch

# visualize
rqt_image_view /doll_state/debug_image
rostopic echo /doll_state/state

    Ensure your launch sets:
    <param name="model_path" value="$(find doll_head_torso)/weights/best.pt"/>.

1) Tested Environment

    Ubuntu 20.04 + ROS Noetic (Python 3)

    RealSense D456 (dev serial used: 333422300494)

    GPU optional (CUDA). CPU works (slower).

2) Fresh Ubuntu Install — All Dependencies
2.1 Install ROS Noetic + catkin

# apt sources
sudo apt update
sudo apt install -y curl gnupg lsb-release
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# ROS
sudo apt update
sudo apt install -y ros-noetic-desktop-full

# env
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# tools & rosdep
sudo apt install -y python3-rosdep python3-rosinstall python3-vcstools build-essential
sudo rosdep init || true
rosdep update

# workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws && catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

2.2 ROS packages used by this repo

sudo apt install -y \
  ros-noetic-realsense2-camera \
  ros-noetic-cv-bridge \
  ros-noetic-image-transport \
  ros-noetic-sensor-msgs \
  ros-noetic-std-msgs \
  ros-noetic-geometry-msgs \
  ros-noetic-rqt-image-view \
  ros-noetic-rqt-plot \
  jq unzip

2.3 Python libraries

# (optional) use a venv
# python3 -m venv ~/venv-doll && source ~/venv-doll/bin/activate

python3 -m pip install --upgrade pip
pip install ultralytics opencv-python numpy PyYAML
# Install torch matching your CUDA/OS from https://pytorch.org/get-started/locally/

    Depth alignment is required: RealSense depth must be aligned to color for correct 3D reprojection.

3) Repository Layout

doll_head_torso/
├─ launch/
│  ├─ camera.launch           # wraps realsense2_camera/rs_camera.launch
│  ├─ yolo_seg.launch         # camera + yolo_seg_node.py (visual debug)
│  └─ doll_state_cord.launch  # camera + doll_state_node.py (state + coords)
├─ nodes/
│  ├─ yolo_seg_node.py        # segmentation + annotated image
│  └─ doll_state_node.py      # robust state + coordinates (2D/3D)
├─ scripts/
│  └─ download_assets.sh      # downloads weights; dataset optional (off by default)
├─ weights/                   # best.pt (ignored by git; created by script)
├─ docs/                      # additional docs (this file)
├─ CMakeLists.txt
├─ package.xml
└─ .gitignore

4) Launch Files
4.1 launch/camera.launch

Wrapper around realsense2_camera/rs_camera.launch. Key args:

<launch>
  <arg name="enable_color" value="true"/>
  <arg name="enable_depth" value="true"/>
  <arg name="align_depth" value="true"/>
  <arg name="serial_no" value="333422300494"/>
  <arg name="color_width" value="640"/>
  <arg name="color_height" value="480"/>
  <arg name="color_fps" value="30"/>
  <arg name="depth_width" value="640"/>
  <arg name="depth_height" value="480"/>
  <arg name="depth_fps" value="30"/>

  <include file="$(find realsense2_camera)/launch/rs_camera.launch">
    <arg name="enable_color" value="$(arg enable_color)"/>
    <arg name="enable_depth" value="$(arg enable_depth)"/>
    <arg name="align_depth" value="$(arg align_depth)"/>
    <arg name="serial_no" value="$(arg serial_no)"/>
    <arg name="color_width" value="$(arg color_width)"/>
    <arg name="color_height" value="$(arg color_height)"/>
    <arg name="color_fps" value="$(arg color_fps)"/>
    <arg name="depth_width" value="$(arg depth_width)"/>
    <arg name="depth_height" value="$(arg depth_height)"/>
    <arg name="depth_fps" value="$(arg depth_fps)"/>
  </include>
</launch>

Publishes: /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw, /camera/color/camera_info.
4.2 launch/yolo_seg.launch

Camera + yolo_seg_node.py (visual testing). Example:

<launch>
  <include file="$(find doll_head_torso)/launch/camera.launch">
    <arg name="enable_color" value="true"/>
    <arg name="enable_depth" value="false"/>
    <arg name="align_depth" value="false"/>
  </include>

  <node pkg="doll_head_torso" type="yolo_seg_node.py" name="yolo_seg" output="screen">
    <param name="model_path" value="$(find doll_head_torso)/weights/best.pt"/>
    <param name="image_topic" value="/camera/color/image_raw"/>
    <param name="conf" value="0.33"/>
    <param name="iou" value="0.58"/>
    <param name="imgsz" value="640"/>
    <param name="device" value="0"/>     <!-- 0=GPU, 'cpu'=CPU -->
    <param name="process_every_n" value="1"/>
  </node>
</launch>

Publishes /yolo_seg/annotated (Image) for rqt.
4.3 launch/doll_state_cord.launch (recommended runtime)

<launch>
  <include file="$(find doll_head_torso)/launch/camera.launch">
    <arg name="enable_color" value="true"/>
    <arg name="enable_depth" value="true"/>
    <arg name="align_depth" value="true"/>
  </include>

  <node pkg="doll_head_torso" type="doll_state_node.py" name="doll_state" output="screen">
    <param name="model_path" value="$(find doll_head_torso)/weights/best.pt"/>
    <param name="image_topic" value="/camera/color/image_raw"/>
    <param name="depth_topic" value="/camera/aligned_depth_to_color/image_raw"/>
    <param name="camera_info_topic" value="/camera/color/camera_info"/>
    <param name="use_depth" value="true"/>

    <!-- Inference -->
    <param name="conf" value="0.33"/>
    <param name="iou" value="0.58"/>
    <param name="imgsz" value="640"/>
    <param name="device" value="0"/>
    <param name="process_every_n" value="1"/>

    <!-- Seg/Logic -->
    <param name="mask_threshold" value="0.5"/>
    <param name="min_conf_ht" value="0.5"/>
    <param name="ht_union_iou_min" value="0.5"/>
    <param name="overlap_thresh" value="0.22"/>
    <param name="center_dist_factor" value="0.55"/>
    <param name="min_vertical_margin" value="0.05"/>
    <param name="max_head_to_torso_area_ratio" value="0.8"/>

    <!-- Hysteresis -->
    <param name="min_frames_to_assembled" value="5"/>
    <param name="min_frames_to_disassembled" value="3"/>
    <param name="unknown_decay" value="1"/>

    <!-- Debug -->
    <param name="publish_debug" value="true"/>
  </node>
</launch>

5) Dataflow (Mermaid)

flowchart LR
  subgraph CAM[Intel RealSense D456]
    C["/camera/color/image_raw"]
    D["/camera/aligned_depth_to_color/image_raw"]
    I["/camera/color/camera_info"]
  end

  subgraph DS[doll_state_node.py]
    note["YOLOv8 seg (agnostic NMS)<br/>ht validation vs union(head,torso)<br/>geometry checks + hysteresis<br/>2D/3D coordinates"]
  end

  C -->|sensor_msgs/Image| DS
  D -->|sensor_msgs/Image| DS
  I -->|sensor_msgs/CameraInfo| DS

  DS -->|sensor_msgs/Image| DBG["/doll_state/debug_image"]
  DS -->|std_msgs/String| ST["/doll_state/state"]
  DS -->|std_msgs/String (JSON)| PJ["/doll_state/parts_json"]
  DS -->|geometry_msgs/PoseStamped| HP["/doll_state/head_pose"]
  DS -->|geometry_msgs/PoseStamped| TP["/doll_state/torso_pose"]
  DS -->|geometry_msgs/PoseStamped| HTP["/doll_state/ht_pose"]

6) Nodes & Topics
6.1 nodes/yolo_seg_node.py

    Sub: /camera/color/image_raw

    Pub: ~annotated → /yolo_seg/annotated (Image)

    Params: model_path, conf, iou, imgsz, device, process_every_n

6.2 nodes/doll_state_node.py (main)

    Sub:

        /camera/color/image_raw

        /camera/aligned_depth_to_color/image_raw (if use_depth=true)

        /camera/color/camera_info

    Pub:

        ~debug_image → /doll_state/debug_image (Image)

        ~state → /doll_state/state (String: assembled | disassembled | unknown)

        ~parts_json → /doll_state/parts_json (String JSON per frame)

        ~head_pose, ~torso_pose, ~ht_pose → PoseStamped (XYZ in camera frame)

parts_json schema (example):

{
  "frame_id": "camera_color_optical_frame",
  "parts": [
    {
      "class": "head|torso|ht",
      "conf": 0.0,
      "bbox": [x1, y1, x2, y2],
      "centroid_px": [u, v],
      "area_px": 1234,
      "point_camera": [X, Y, Z]  // meters, camera frame; null if depth unavailable
    }
  ]
}

7) Decision Logic (summary)

    Prefer class ht (assembled) if confidence ≥ min_conf_ht and mask overlaps union(head, torso) with IoU ≥ ht_union_iou_min.

    Else, pair head + torso via:

        mask overlap ≥ overlap_thresh or normalized centroid distance ≤ center_dist_factor

        geometry constraints: head above torso by ≥ min_vertical_margin of image height; head area < max_head_to_torso_area_ratio × torso area

    Temporal hysteresis stabilizes state across frames.

    Inference uses agnostic NMS; conf/iou thresholds are tunable.

8) Parameters (suggested defaults)

Inference
~model_path (str), ~conf=0.33, ~iou=0.58, ~imgsz=640, ~device=0 (or 'cpu'), ~process_every_n=1

Topics
~image_topic=/camera/color/image_raw, ~depth_topic=/camera/aligned_depth_to_color/image_raw, ~camera_info_topic=/camera/color/camera_info, ~use_depth=true

Seg/Logic
~mask_threshold=0.5, ~min_conf_ht=0.5, ~ht_union_iou_min=0.5,
~overlap_thresh=0.22, ~center_dist_factor=0.55,
~min_vertical_margin=0.05 (of image height), ~max_head_to_torso_area_ratio=0.8

Hysteresis
~min_frames_to_assembled=5, ~min_frames_to_disassembled=3, ~unknown_decay=1

Debug
~publish_debug=true
9) How to Inspect Coordinates (no robot required)

3D XYZ (meters, camera frame)

rostopic echo /doll_state/head_pose/pose/position
rostopic echo /doll_state/torso_pose/pose/position
rostopic echo /doll_state/ht_pose/pose/position

2D + details (JSON per frame)

rostopic echo /doll_state/parts_json
# Pretty-print one sample (requires jq)
rostopic echo -n1 /doll_state/parts_json | sed -n 's/^data: //p' | jq

Visual overlay

rqt_image_view /doll_state/debug_image

    PoseStamped.header.frame_id is the camera frame (e.g., camera_color_optical_frame).
    Use tf2 to transform to your robot base/world if needed.

10) Dataset & Weights

    Weights (weights/best.pt) are hosted as a GitHub Release asset and downloaded by scripts/download_assets.sh.
    By default, the script downloads only the model.

    Dataset (reference only): the full dataset used for training (images, labels, runs/logs, data.yaml) is published as a Release asset.
    It’s not required to run inference and not downloaded by default.
    If you want the script to fetch it, uncomment and fill DATASET_ZIP_URL, DATASET_DIR_NAME, UNZIP_TARGET in the script, then re-run it.

Download script (exact content used in this repo):

#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG (fill in if needed) =========
# Direct link to your best.pt hosted on a GitHub Release
WEIGHTS_URL="https://github.com/willito05/doll_head_torso/releases/download/v0.1.0/best.pt"

# Optional dataset ZIP (leave commented/empty to skip)
# DATASET_ZIP_URL="https://github.com/willito05/doll_head_torso/releases/download/v0.1.1/Dataset_full_v01.zip"
# DATASET_DIR_NAME="Dataset"      # root folder name inside the ZIP
# UNZIP_TARGET="."                # where to extract ('.' or 'datasets')
# =============================================

mkdir -p weights datasets

# ---- Weights (required) ----
if [ -f weights/best.pt ]; then
  echo "[OK] weights/best.pt already present (skip)"
else
  if [ -z "${WEIGHTS_URL:-}" ]; then
    echo "[ERROR] WEIGHTS_URL is empty. Please set a direct link to best.pt."
    exit 1
  fi
  echo "[DL] Downloading weights -> weights/best.pt"
  curl -fL "$WEIGHTS_URL" -o weights/best.pt
  echo "[OK] weights/best.pt downloaded."
fi

# ---- Dataset (optional) ----
if [ -n "${DATASET_ZIP_URL:-}" ]; then
  DATASET_DIR_NAME="${DATASET_DIR_NAME:-Dataset}"
  UNZIP_TARGET="${UNZIP_TARGET:-datasets}"

  if [ -d "$UNZIP_TARGET/$DATASET_DIR_NAME" ]; then
    echo "[OK] Dataset $UNZIP_TARGET/$DATASET_DIR_NAME already present (skip)"
  else
    echo "[DL] Downloading dataset ZIP ..."
    tmpzip="/tmp/dataset_$$.zip"
    curl -fL "$DATASET_ZIP_URL" -o "$tmpzip"
    echo "[UNZIP] Extracting to $UNZIP_TARGET/"
    mkdir -p "$UNZIP_TARGET"
    unzip -o "$tmpzip" -d "$UNZIP_TARGET"
    rm -f "$tmpzip"
    echo "[OK] Dataset extracted to $UNZIP_TARGET/$DATASET_DIR_NAME"
  fi
else
  echo "[INFO] No dataset URL set (skipping dataset)."
fi

# ---- Summary ----
echo "------------------------------------------"
[ -f weights/best.pt ] && echo "[OK] weights/best.pt ready"
if [ -n "${DATASET_ZIP_URL:-}" ] && [ -d "${UNZIP_TARGET:-datasets}/${DATASET_DIR_NAME:-Dataset}" ]; then
  echo "[OK] dataset ready at ${UNZIP_TARGET:-datasets}/${DATASET_DIR_NAME:-Dataset}"
else
  echo "[INFO] No local dataset (optional)"
fi
echo "Done."

11) Training Notes (optional)

Dataset YAML (example)

# datasets/doll_2parts/data.yaml
path: datasets/doll_2parts
train: images/train
val: images/val
names: [head, torso, ht]

Train with Ultralytics (example)

# Example command (adjust paths/model as needed)
yolo segment train \
  data=datasets/doll_2parts/data.yaml \
  model=yolov8n-seg.pt \
  imgsz=640 epochs=100 batch=16 \
  project=runs/segment name=train

# Export best weights to repo
mkdir -p weights && cp runs/segment/train/weights/best.pt weights/best.pt

Best practices

    Balance classes across head, torso, ht.

    Include edge cases (near-assembled, occlusions, lighting/angles).

    Precise masks near the neck interface.

    Moderate augmentations (±20° rotation, light blur, brightness/contrast).

12) Troubleshooting

Grey image in rqt_image_view

rqt_image_view /doll_state/debug_image
rostopic hz /doll_state/debug_image

No 3D coordinates

    Enable and align depth: enable_depth=true, align_depth=true (in camera.launch).

    Verify topics: /camera/aligned_depth_to_color/image_raw, /camera/color/camera_info.

Unstable state (assembled/disassembled)

    Increase conf (0.35–0.40) and iou (0.60).

    Tighten geometry: raise overlap_thresh, lower center_dist_factor.

    Increase min_frames_to_assembled.

Performance

    Reduce imgsz (512/480) or process_every_n=2.

    Prefer GPU (device=0) with a matching Torch/CUDA build.

13) Git & Releases (reference)

.gitignore (used here)

__pycache__/
*.pyc
Dataset/
datasets/
runs/
weights/
*.pt
*.onnx
*.bag
*.zip
.vscode/
.idea/

Minimal CMakeLists.txt

cmake_minimum_required(VERSION 3.0.2)
project(doll_head_torso)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs std_msgs geometry_msgs cv_bridge image_transport)
catkin_package()
catkin_install_python(PROGRAMS
  nodes/yolo_seg_node.py
  nodes/doll_state_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

Minimal package.xml

<?xml version="1.0"?>
<package format="2">
  <name>doll_head_torso</name>
  <version>0.1.0</version>
  <description>Doll head–torso detection and state estimation (YOLOv8 seg).</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>

  <buildtool_depend>catkin</buildtool_depend>

  <exec_depend>rospy</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>cv_bridge</exec_depend>
  <exec_depend>image_transport</exec_depend>
  <exec_depend>realsense2_camera</exec_depend>
</package>

Release flow (weights + dataset)

    Draft a GitHub Release (e.g., v0.1.0) → upload best.pt.

    Update WEIGHTS_URL in scripts/download_assets.sh.

    (Optional) Draft another Release (v0.1.1) → upload Dataset_full_*.zip.

    Keep dataset off by default (script variables commented), users can download manually from Releases.

    Never share private SSH keys. If using SSH, add only the public key to GitHub (ssh-ed25519 AAAA...).

14) Changelog (brief)

    v1: yolo_seg_node.py (annotation only)

    v2: doll_state_node.py (state + debug overlay)

    v3: Robustness: agnostic NMS, ht vs union(head∪torso), geometry checks, hysteresis

    v4: Publish per-part 2D/3D (parts_json, PoseStamped)
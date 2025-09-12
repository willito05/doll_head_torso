# Doll Head–Torso Detection & State Estimation – Handover Guide (EN)

**Owner package:** `doll_head_torso`
**Core goal:** Detect whether a doll is **assembled** (head+torso) or **disassembled**, and publish the decision plus **2D/3D coordinates** for each part so a robot arm can pick them up.

---

## 1) High‑level Overview

* **Camera:** Intel RealSense D456 (serial used during dev: `333422300494`).
* **Perception:** YOLOv8 **segmentation** model trained on 3 classes: `head`, `torso`, `ht` (assembled head+torso).
* **ROS node(s):**

  * `yolo_seg_node.py` → runs segmentation and publishes an **annotated image** (for visual checks).
  * `doll_state_node.py` → runs segmentation **with robust logic** to decide `assembled / disassembled / unknown`, **smooths** the decision over time (hysteresis), and publishes **coordinates** (2D/3D) per detected part.
* **Visualization:** `rqt_image_view` for images; `rostopic echo` / `rqt_plot` for numeric data.

### Data Flow (conceptual)

```
RealSense (color + aligned depth) --> /camera/color/image_raw
                                   --> /camera/aligned_depth_to_color/image_raw
                                   --> /camera/color/camera_info

                +---------------------------------------------+
                | doll_state_node.py                          |
                |  - YOLOv8 seg inference (agnostic NMS)      |
                |  - validate 'ht' vs union(head,torso)        |
                |  - geometry constraints (head above torso)   |
                |  - overlap / centroid distance checks        |
                |  - hysteresis (temporal smoothing)           |
                |  - 2D/3D coordinates for parts               |
                +-----------------+---------------------------+
                                  |
             +--------------------+------------------------+
             |                    |                        |
   /doll_state/debug_image   /doll_state/state     /doll_state/parts_json
        (Image)                 (String)                  (String JSON)
                                   |
                 +-----------------+------------------------+
                 |                 |                        |
   /doll_state/head_pose   /doll_state/torso_pose   /doll_state/ht_pose
        (PoseStamped)            (PoseStamped)            (PoseStamped)
```

---

## 2) Environment & Dependencies

* **ROS 1 (rospy)**, Python 3.
* **Core ROS packages:** `realsense2_camera`, `cv_bridge`, `image_transport`.
* **Python libs:** `ultralytics`, `torch`, `opencv-python`, `numpy`.
* **RealSense:** Depth **aligned to color** must be enabled for 3D points to match masks.

> Install `ultralytics/torch` via `pip` in your ROS Python environment. Ensure CUDA is available if using GPU (set `device=0` in params).

---

## 3) Package Layout

```
doll_head_torso/
├─ launch/
│  ├─ camera.launch          # wraps realsense2_camera/rs_camera.launch
│  ├─ yolo_seg.launch        # camera + yolo_seg_node.py (visual debug)
│  └─ doll_state.launch      # camera + doll_state_node.py (state + coords)
├─ nodes/
│  ├─ yolo_seg_node.py       # segmentation + annotated image
│  └─ doll_state_node.py     # robust state + coords (2D/3D)
└─ ...
```

**Model path used during dev:** `/doll/doll_2parts/runs/segment/train/weights/best.pt`
Update this path via ROS param `~model_path` if you relocate the weights.

---

## 4) Launch Files

### `launch/camera.launch`

Wrapper around `realsense2_camera/rs_camera.launch` exposing only required args.

Key args (examples):

```xml
<arg name="enable_color"       default="true"/>
<arg name="enable_depth"       default="true"/>
<arg name="align_depth"        default="true"/>
<arg name="serial_no"          default="333422300494"/>
<arg name="color_width"        default="640"/>
<arg name="color_height"       default="480"/>
<arg name="color_fps"          default="30"/>
<arg name="depth_width"        default="640"/>
<arg name="depth_height"       default="480"/>
<arg name="depth_fps"          default="30"/>
<arg name="enable_pointcloud"  default="false"/>
```

Publishes at least:

* `/camera/color/image_raw` (sensor\_msgs/Image)
* `/camera/aligned_depth_to_color/image_raw` (sensor\_msgs/Image)
* `/camera/color/camera_info` (sensor\_msgs/CameraInfo)

### `launch/yolo_seg.launch`

Starts camera + `yolo_seg_node.py` (for visual testing only).

* Publishes annotated image: `/yolo_seg/annotated`.

### `launch/doll_state.launch`

Starts camera + `doll_state_node.py` (main runtime). Enables **aligned depth**.

* Publishes state/coords topics (see below).

---

## 5) Nodes & Topics

### `nodes/yolo_seg_node.py`

* **Subscribes:** `/camera/color/image_raw`
* **Publishes:** `~annotated` → `/yolo_seg/annotated` (sensor\_msgs/Image)
* **Params:** `model_path`, `conf`, `iou`, `imgsz`, `device`, `process_every_n`

### `nodes/doll_state_node.py` (main)

* **Subscribes:**

  * Color: `/camera/color/image_raw`
  * Depth (aligned): `/camera/aligned_depth_to_color/image_raw` (if `use_depth=true`)
  * Camera intrinsics: `/camera/color/camera_info`
* **Publishes:**

  * `~debug_image` → `/doll_state/debug_image` (sensor\_msgs/Image)
  * `~state` → `/doll_state/state` (std\_msgs/String: `assembled|disassembled|unknown`)
  * `~parts_json` → `/doll_state/parts_json` (std\_msgs/String, JSON per frame)
  * `~head_pose` → `/doll_state/head_pose` (geometry\_msgs/PoseStamped, XYZ in camera frame)
  * `~torso_pose` → `/doll_state/torso_pose` (PoseStamped)
  * `~ht_pose` → `/doll_state/ht_pose` (PoseStamped)

**Message content in `parts_json`:**

```json
{
  "frame_id": "camera_color_optical_frame",
  "parts": [
    {
      "class": "head|torso|ht",
      "conf": 0.0..1.0,
      "bbox": [x1, y1, x2, y2],
      "centroid_px": [u, v],
      "area_px": N_pixels,
      "point_camera": [X, Y, Z]  // meters in camera frame, null if depth unavailable
    }
  ]
}
```

---

## 6) Decision Logic (summary)

1. **Primary assembled evidence:** class `ht` present with confidence ≥ `min_conf_ht` **and** its mask sufficiently overlaps the union of (`head ∪ torso`) (IoU ≥ `ht_union_iou_min`).
2. Else, evaluate `head` + `torso` pairs using:

   * **Mask overlap** (≥ `overlap_thresh`) or **normalized centroid distance** (≤ `center_dist_factor`).
   * **Geometry constraints:** head must be **above** torso by at least `min_vertical_margin` of image height; head area must be < `max_head_to_torso_area_ratio` × torso area.
3. If only one part is detected → `disassembled`.
4. Temporal **hysteresis**: require `min_frames_to_assembled` to confirm assembled; fewer frames (`min_frames_to_disassembled`) to fall back to disassembled.

Additional inference settings:

* **Agnostic NMS** (`agnostic_nms=True`) to reduce cross‑class duplicates.
* Tunable `conf`/`iou` thresholds for Ultralytics inference.

---

## 7) Parameters (with suggested defaults)

Set in `doll_state.launch` or via `rosparam`:

**Model/Inference**

* `~model_path` (str): path to YOLOv8 segmentation weights.
* `~conf` (float, default `0.33`)
* `~iou` (float, default `0.58`)
* `~imgsz` (int, default `640`)
* `~device` (str|int, default `0` → GPU #0; or `'cpu'`)
* `~process_every_n` (int, default `1` → process every frame)

**Topics**

* `~image_topic` (default `/camera/color/image_raw`)
* `~depth_topic` (default `/camera/aligned_depth_to_color/image_raw`)
* `~camera_info_topic` (default `/camera/color/camera_info`)
* `~use_depth` (bool, default `true`)

**Segmentation/Logic**

* `~mask_threshold` (float, default `0.5`)
* `~min_conf_ht` (float, default `0.5`)
* `~ht_union_iou_min` (float, default `0.5`)
* `~overlap_thresh` (float, default `0.22`)
* `~center_dist_factor` (float, default `0.55`)
* `~min_vertical_margin` (float, default `0.05` of image height)
* `~max_head_to_torso_area_ratio` (float, default `0.8`)

**Hysteresis**

* `~min_frames_to_assembled` (int, default `5`)
* `~min_frames_to_disassembled` (int, default `3`)
* `~unknown_decay` (int, default `1`)

**Debug**

* `~publish_debug` (bool, default `true`)

---

## 8) How to Run

### One‑time setup

```bash
chmod +x ~/catkin_ws/src/doll_head_torso/nodes/doll_state_node.py
source ~/catkin_ws/devel/setup.bash
```

### Start camera + state node (with aligned depth)

```bash
roslaunch doll_head_torso doll_state.launch
```

### Visual checks

```bash
# Annotated image with state overlay
rqt_image_view /doll_state/debug_image

# State text
rostopic echo /doll_state/state
```

### Coordinates (no robot required)

```bash
# 3D position (meters, camera frame)
rostopic echo /doll_state/head_pose/pose/position
rostopic echo /doll_state/torso_pose/pose/position
rostopic echo /doll_state/ht_pose/pose/position

# 2D+3D details (JSON per frame)
rostopic echo /doll_state/parts_json
# Pretty-print one sample (requires jq)
rostopic echo -n1 /doll_state/parts_json | sed -n 's/^data: //p' | jq
```

---

## 9) Troubleshooting

**Grey image in rqt\_image\_view**

* Open directly on the topic: `rqt_image_view /doll_state/debug_image`.
* Ensure the node is publishing: `rostopic hz /doll_state/debug_image`.

**No 3D coordinates (null point\_camera / empty PoseStamped)**

* Check that depth is enabled and **aligned**: `enable_depth=true`, `align_depth=true` in `camera.launch`.
* Verify topics: `/camera/aligned_depth_to_color/image_raw` and `/camera/color/camera_info` exist.

**Flaky assembled/disassembled decision**

* Increase `conf` to 0.35–0.40 and `iou` to 0.6.
* Tighten geometry: raise `overlap_thresh` (e.g., 0.25), lower `center_dist_factor`.
* Adjust hysteresis: increase `min_frames_to_assembled`.

**Performance issues**

* Lower `imgsz` (e.g., 512 or 480) or set `process_every_n=2`.
* Use GPU (`device=0`) and ensure correct CUDA drivers.

**Topic naming collisions**

* Use remapping or rename node with `name="doll_state"` per robot.

---

## 10) Re‑training Notes (if needed)

* Ensure balanced samples across `head`, `torso`, `ht`.
* Include **edge cases**: head near torso but not assembled, partial occlusions, varied lighting/angles.
* Encourage precise masks around the neck interface.
* Apply moderate augmentations (±20° rotation, slight blur, brightness/contrast jitter).

Model drop‑in: train with Ultralytics (seg), then update `~model_path`.

---

## 11) Future Work

* Replace `parts_json` (String) with custom messages (e.g., `DollPart.msg`, `DollDetections.msg`).
* Publish `visualization_msgs/MarkerArray` for RViz markers at 3D points.
* Add **tracking** across frames (e.g., ByteTrack/DeepSORT) to improve temporal consistency.
* Add grasp pose estimation (orientation) per part to directly feed a manipulator.

---

## 12) Changelog (brief)

* **v1 (initial):** `yolo_seg_node.py` for annotation only.
* **v2:** `doll_state_node.py` with assembled/disassembled logic and debug overlay.
* **v3:** Robustness upgrades: agnostic NMS, `ht` validation vs union(head∪torso), geometry constraints, hysteresis.
* **v4:** Coordinate publishing: 2D/3D per part (`parts_json`, `PoseStamped` topics).

---

## 13) Quick Command Cheat‑Sheet

```bash
# Launch
roslaunch doll_head_torso doll_state.launch

# Visual debug
rqt_image_view /doll_state/debug_image

# State
rostopic echo /doll_state/state

# 3D coords
rostopic echo /doll_state/head_pose/pose/position

# JSON details
rostopic echo -n1 /doll_state/parts_json | sed -n 's/^data: //p' | jq

# Topic scan
rostopic list | grep -E "doll_state|camera|image|depth|camera_info"
```

---

**Contact / Handover:**

* Model weights location: `/doll/doll_2parts/.../best.pt` (update param if moved)
* Camera used: RealSense D456 (serial `333422300494`)
* Primary entrypoint for production: `roslaunch doll_head_torso doll_state.launch`

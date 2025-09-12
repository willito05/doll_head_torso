#!/usr/bin/env python3
# coding: utf-8
import os
import math
import json
from collections import deque

import rospy
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO

# ---------- Utils ----------
def centroid_from_mask(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))  # (u, v)

def bbox_from_mask(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None
    x1, x2 = int(np.min(xs)), int(np.max(xs))
    y1, y2 = int(np.min(ys)), int(np.max(ys))
    return (x1, y1, x2, y2)

def area_from_mask(mask_bool):
    return float(np.count_nonzero(mask_bool))

def overlap_ratio_min(a, b):
    inter = np.logical_and(a, b).sum()
    amin = min(a.sum(), b.sum())
    if amin == 0:
        return 0.0
    return float(inter) / float(amin)

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    u = a.sum() + b.sum() - inter
    if u == 0:
        return 0.0
    return float(inter) / float(u)

def depth_to_meters(depth_arr):
    # RealSense: souvent 16UC1 en millimètres (aligned_depth_to_color)
    if depth_arr.dtype == np.uint16:
        return depth_arr.astype(np.float32) / 1000.0
    # 32FC1 déjà en mètres
    return depth_arr.astype(np.float32)

def median_depth_in_mask(depth_m, mask_bool):
    vals = depth_m[mask_bool]
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals > 0.05) & (vals < 10.0)]  # filtre grossier 5cm..10m
    if vals.size == 0:
        return None
    # médiane robuste
    return float(np.median(vals))

def backproject(u, v, z, fx, fy, cx, cy):
    # Repère caméra (optical): X pointant à droite, Y vers le bas, Z vers l'avant (convention ROS optical).
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return X, Y, Z

# ---------- Node ----------
class DollStateNode:
    def __init__(self):
        self.bridge = CvBridge()

        # --- Params
        self.model_path = rospy.get_param("~model_path", "weights/best.pt")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.use_depth = bool(rospy.get_param("~use_depth", True))  # active calcul XYZ

        self.conf = float(rospy.get_param("~conf", 0.33))
        self.iou  = float(rospy.get_param("~iou", 0.58))
        self.device = rospy.get_param("~device", "0")
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.process_every_n = int(rospy.get_param("~process_every_n", 1))

        # Seg/masks + logique
        self.mask_threshold = float(rospy.get_param("~mask_threshold", 0.5))
        self.min_conf_ht = float(rospy.get_param("~min_conf_ht", 0.5))
        self.ht_union_iou_min = float(rospy.get_param("~ht_union_iou_min", 0.5))

        self.overlap_thresh = float(rospy.get_param("~overlap_thresh", 0.22))
        self.center_dist_factor = float(rospy.get_param("~center_dist_factor", 0.55))
        self.min_vertical_margin = float(rospy.get_param("~min_vertical_margin", 0.05))  # % hauteur image
        self.max_head_to_torso_area_ratio = float(rospy.get_param("~max_head_to_torso_area_ratio", 0.8))

        # Hystérésis
        self.stable_state = "unknown"
        self.assembled_streak = 0
        self.disassembled_streak = 0
        self.min_frames_to_assembled = int(rospy.get_param("~min_frames_to_assembled", 5))
        self.min_frames_to_disassembled = int(rospy.get_param("~min_frames_to_disassembled", 3))
        self.unknown_decay = int(rospy.get_param("~unknown_decay", 1))

        # Debug
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        # Modèle
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            rospy.loginfo(f"[doll_state_node] Modèle chargé: {self.model_path} (device={self.device})")
        except Exception as e:
            rospy.logerr(f"[doll_state_node] Erreur chargement modèle: {e}")
            raise

        # Publishers
        self.pub_state = rospy.Publisher("~state", String, queue_size=1)
        self.pub_debug = rospy.Publisher("~debug_image", Image, queue_size=1) if self.publish_debug else None
        self.pub_parts_json = rospy.Publisher("~parts_json", String, queue_size=1)
        self.pub_head_pose = rospy.Publisher("~head_pose", PoseStamped, queue_size=1)
        self.pub_torso_pose = rospy.Publisher("~torso_pose", PoseStamped, queue_size=1)
        self.pub_ht_pose = rospy.Publisher("~ht_pose", PoseStamped, queue_size=1)

        # Subscriptions
        self.frame_idx = 0
        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1, buff_size=2**24)

        self.depth_msg = None
        self.cam_info = None
        self.fx = self.fy = self.cx = self.cy = None
        if self.use_depth:
            rospy.Subscriber(self.depth_topic, Image, self.cb_depth, queue_size=1, buff_size=2**24)
            rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_caminfo, queue_size=1)

        # Classes
        self.names = self.model.names
        self.name_to_id = {v: k for k, v in self.names.items()}
        for need in ["head", "torso", "ht"]:
            if need not in self.name_to_id:
                rospy.logwarn(f"[doll_state_node] Classe absente du modèle: {need}")

    # ---------- Callbacks ----------
    def cb_depth(self, msg):
        self.depth_msg = msg

    def cb_caminfo(self, msg):
        self.cam_info = msg
        # Intrinsèques depuis K (3x3)
        if len(msg.K) == 9:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]

    def cb_image(self, msg):
        self.frame_idx += 1
        if self.process_every_n > 1 and (self.frame_idx % self.process_every_n) != 0:
            return

        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img_h = img_bgr.shape[0]

        # Inference (agnostic nms)
        try:
            res = self.model.predict(
                source=img_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
                agnostic_nms=True
            )[0]
        except Exception as e:
            rospy.logwarn(f"[doll_state_node] Erreur inference: {e}")
            return

        # Décision
        cand, reason, dets, masks = self.decide_frame_state(res, img_h)

        # Hystérésis
        self.update_stable_state(cand)

        # Publier état
        self.pub_state.publish(String(self.stable_state))

        # Publier parties (2D + 3D si dispo)
        parts_payload = self.build_and_publish_parts(msg.header, dets, masks)
        self.pub_parts_json.publish(String(json.dumps(parts_payload)))

        # Debug image
        if self.pub_debug:
            try:
                vis = res.plot()
            except Exception:
                vis = img_bgr.copy()
            vis = self.overlay_debug(vis, self.stable_state, f"{cand} | {reason}")
            out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            out.header = msg.header
            self.pub_debug.publish(out)

    # ---------- Decision logic ----------
    def decide_frame_state(self, res, img_h):
        debug_objs = []
        if res is None or res.boxes is None or len(res.boxes) == 0:
            return "unknown", "no_detections", [], None

        classes = res.boxes.cls.detach().cpu().numpy().astype(int)
        confs   = res.boxes.conf.detach().cpu().numpy().astype(float)

        masks = None
        if res.masks is not None and res.masks.data is not None:
            raw = res.masks.data.detach().cpu().numpy()
            masks = raw > self.mask_threshold

        head_idxs, torso_idxs, ht_idxs = [], [], []
        for i, c in enumerate(classes):
            name = self.names.get(int(c), f"class_{int(c)}")
            conf = float(confs[i])
            debug_objs.append({"i": int(i), "cls": name, "conf": conf})
            if name == "head": head_idxs.append(i)
            elif name == "torso": torso_idxs.append(i)
            elif name == "ht": ht_idxs.append(i)

        # 1) ht validé vs union
        if ht_idxs:
            best_ht = max(ht_idxs, key=lambda i: confs[i])
            ht_ok = confs[best_ht] >= self.min_conf_ht
            if masks is not None and head_idxs and torso_idxs and ht_ok:
                union = np.zeros_like(masks[0], dtype=bool)
                for hi in head_idxs:
                    if hi < masks.shape[0]: union |= masks[hi]
                for ti in torso_idxs:
                    if ti < masks.shape[0]: union |= masks[ti]
                if best_ht < masks.shape[0]:
                    ht_mask = masks[best_ht]
                    if iou(ht_mask, union) < self.ht_union_iou_min:
                        ht_ok = False
            if ht_ok:
                return "assembled", "ht_present_validated", debug_objs, masks

        if not head_idxs and not torso_idxs:
            return "unknown", "neither_head_nor_torso", debug_objs, masks

        # 2) head + torso cohérents
        if head_idxs and torso_idxs and masks is not None:
            best_score = -1.0
            best_eval = None
            for hi in head_idxs:
                if hi >= masks.shape[0]: continue
                hmask = masks[hi]
                hcent = centroid_from_mask(hmask)
                hbox  = bbox_from_mask(hmask)
                harea = area_from_mask(hmask)
                for ti in torso_idxs:
                    if ti >= masks.shape[0]: continue
                    tmask = masks[ti]
                    tcent = centroid_from_mask(tmask)
                    tbox  = bbox_from_mask(tmask)
                    tarea = area_from_mask(tmask)

                    ov = overlap_ratio_min(hmask, tmask)
                    if hcent is None or tcent is None:
                        nd = 1e9
                    else:
                        dist = math.hypot(hcent[0]-tcent[0], hcent[1]-tcent[1])
                        def box_h(b): return 0 if b is None else max(1, (b[3]-b[1]+1))
                        scale = max(1.0, min(box_h(hbox), box_h(tbox)))
                        nd = dist / float(scale)

                    score = ov*2.0 + max(0.0, (1.5 - nd))
                    if score > best_score:
                        best_score = score
                        best_eval = (ov, nd, hcent, tcent, harea, tarea)

            if best_eval is not None:
                ov, nd, hcent, tcent, harea, tarea = best_eval
                geom_ok = True
                if hcent is None or tcent is None:
                    geom_ok = False
                else:
                    min_margin_px = self.min_vertical_margin * float(img_h)
                    if not (hcent[1] + min_margin_px < tcent[1]):
                        geom_ok = False
                if tarea <= 0 or harea >= self.max_head_to_torso_area_ratio * tarea:
                    geom_ok = False

                if geom_ok and ((ov >= self.overlap_thresh) or (nd <= self.center_dist_factor)):
                    return "assembled", f"head+torso overlap={ov:.2f} nd={nd:.2f} geom_ok", debug_objs, masks
                else:
                    return "disassembled", f"head+torso separated/geom_bad (ov={ov:.2f}, nd={nd:.2f})", debug_objs, masks

        # 3) une seule partie
        if (head_idxs and not torso_idxs) or (torso_idxs and not head_idxs):
            return "disassembled", "single_part_only", debug_objs, masks

        return "unknown", "fallback", debug_objs, masks

    # ---------- Hystérésis ----------
    def update_stable_state(self, candidate):
        if candidate == "assembled":
            self.assembled_streak += 1
            self.disassembled_streak = 0
        elif candidate == "disassembled":
            self.disassembled_streak += 1
            self.assembled_streak = 0
        else:
            self.assembled_streak = max(0, self.assembled_streak - self.unknown_decay)
            self.disassembled_streak = max(0, self.disassembled_streak - self.unknown_decay)

        if self.assembled_streak >= self.min_frames_to_assembled:
            self.stable_state = "assembled"
        elif self.disassembled_streak >= self.min_frames_to_disassembled:
            self.stable_state = "disassembled"

    # ---------- Publishing parts (2D/3D) ----------
    def build_and_publish_parts(self, header, dets, masks):
        """
        dets: liste [{i, cls, conf}]
        masks: np.bool_(N,H,W) ou None
        Publie head_pose / torso_pose / ht_pose si 3D dispo.
        Retourne payload JSON sérialisable.
        """
        payload = {"frame_id": header.frame_id, "parts": []}
        H = W = None
        if masks is not None:
            H, W = masks.shape[1], masks.shape[2]

        # Préparer profondeur/intrinsèques
        depth_m = None
        have_3d = False
        if self.use_depth and self.depth_msg is not None and self.cam_info is not None and all(v is not None for v in [self.fx, self.fy, self.cx, self.cy]):
            depth_arr = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
            depth_m = depth_to_meters(depth_arr)
            have_3d = (depth_m is not None)

        best_by_class = {}  # cls -> (pose, conf)
        for d in dets:
            cls = d["cls"]
            idx = d["i"]
            conf = d["conf"]

            # 2D
            centroid = None
            bbox = None
            area = None
            if masks is not None and idx < masks.shape[0]:
                m = masks[idx]
                centroid = centroid_from_mask(m)  # (u,v)
                bbox = bbox_from_mask(m)
                area = int(area_from_mask(m))
            # fallback bbox/centroid absent si pas de masks (rare en seg)

            # 3D
            point_cam = None
            if have_3d and centroid is not None:
                u, v = centroid
                u_i = int(round(u)); v_i = int(round(v))
                if 0 <= v_i < depth_m.shape[0] and 0 <= u_i < depth_m.shape[1]:
                    if masks is not None:
                        z = median_depth_in_mask(depth_m, masks[idx])
                    else:
                        z = float(depth_m[v_i, u_i])
                    if z and np.isfinite(z) and z > 0.05:
                        X, Y, Z = backproject(u, v, z, self.fx, self.fy, self.cx, self.cy)
                        point_cam = [float(X), float(Y), float(Z)]

            part = {
                "class": cls,
                "conf": float(conf),
                "bbox": [int(b) for b in bbox] if bbox else None,
                "centroid_px": [float(centroid[0]), float(centroid[1])] if centroid else None,
                "area_px": area,
                "point_camera": point_cam  # [X,Y,Z] en mètres, repère caméra
            }
            payload["parts"].append(part)

            # Mettre à jour meilleure pose par classe (si 3D dispo)
            if point_cam is not None:
                pose = PoseStamped()
                pose.header = header
                # Important: mettre le frame_id de la caméra (souvent fourni par depth_msg/camera_info)
                pose.header.frame_id = self.cam_info.header.frame_id if self.cam_info else header.frame_id
                pose.pose.position.x = point_cam[0]
                pose.pose.position.y = point_cam[1]
                pose.pose.position.z = point_cam[2]
                # orientation neutre (à adapter si tu estimes une orientation)
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = 0.0
                pose.pose.orientation.w = 1.0
                # garder la plus confiante
                if (cls not in best_by_class) or (conf > best_by_class[cls][1]):
                    best_by_class[cls] = (pose, conf)

        # Publier les meilleurs PoseStamped par classe
        if "head" in best_by_class:
            self.pub_head_pose.publish(best_by_class["head"][0])
        if "torso" in best_by_class:
            self.pub_torso_pose.publish(best_by_class["torso"][0])
        if "ht" in best_by_class:
            self.pub_ht_pose.publish(best_by_class["ht"][0])

        return payload

    # ---------- Overlay ----------
    def overlay_debug(self, img_bgr, state, reason):
        color = (0,255,0) if state=="assembled" else (0,0,255) if state=="disassembled" else (0,255,255)
        cv2.rectangle(img_bgr, (5,5), (560,70), (0,0,0), -1)
        cv2.putText(img_bgr, f"STATE: {state}", (12,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(img_bgr, reason[:70], (12,62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        return img_bgr

# ---------- main ----------
def main():
    rospy.init_node("doll_state_node")
    DollStateNode()
    rospy.loginfo("[doll_state_node] Démarré.")
    rospy.spin()

if __name__ == "__main__":
    main()

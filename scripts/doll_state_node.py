#!/usr/bin/env python3
# coding: utf-8
import os
import math
from collections import Counter, deque

import rospy
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


def centroid_from_mask(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))  # (cx, cy)


def bbox_from_mask(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None
    x1, x2 = int(np.min(xs)), int(np.max(xs))
    y1, y2 = int(np.min(ys)), int(np.max(ys))
    return (x1, y1, x2, y2)  # inclusive


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


class DollStateNode:
    def __init__(self):
        self.bridge = CvBridge()

        # --- Params (ROS)
        self.model_path = rospy.get_param("~model_path", "weights/best.pt")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.conf = float(rospy.get_param("~conf", 0.33))
        self.iou  = float(rospy.get_param("~iou", 0.58))
        self.device = rospy.get_param("~device", "0")   # "0" GPU id, "cpu" sinon
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.process_every_n = int(rospy.get_param("~process_every_n", 1))

        # Logique / seuils
        self.mask_threshold = float(rospy.get_param("~mask_threshold", 0.5))
        self.min_conf_ht = float(rospy.get_param("~min_conf_ht", 0.5))
        self.ht_union_iou_min = float(rospy.get_param("~ht_union_iou_min", 0.50))

        self.overlap_thresh = float(rospy.get_param("~overlap_thresh", 0.22))
        self.center_dist_factor = float(rospy.get_param("~center_dist_factor", 0.55))  # distance centroides / taille bbox
        self.min_vertical_margin = float(rospy.get_param("~min_vertical_margin", 0.05))  # % hauteur image
        self.max_head_to_torso_area_ratio = float(rospy.get_param("~max_head_to_torso_area_ratio", 0.8))

        # Hystérésis (asymétrique)
        self.stable_state = "unknown"
        self.assembled_streak = 0
        self.disassembled_streak = 0
        self.min_frames_to_assembled = int(rospy.get_param("~min_frames_to_assembled", 5))
        self.min_frames_to_disassembled = int(rospy.get_param("~min_frames_to_disassembled", 3))
        self.unknown_decay = int(rospy.get_param("~unknown_decay", 1))  # combien décrémenter les streaks sur unknown

        # Debug
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        # Chargement modèle
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            rospy.loginfo(f"[doll_state_node] Modèle chargé: {self.model_path} (device={self.device})")
        except Exception as e:
            rospy.logerr(f"[doll_state_node] Erreur chargement modèle: {e}")
            raise

        # Topics
        self.pub_state = rospy.Publisher("~state", String, queue_size=1)  # "assembled|disassembled|unknown"
        self.pub_debug = rospy.Publisher("~debug_image", Image, queue_size=1) if self.publish_debug else None

        self.frame_idx = 0
        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1, buff_size=2**24)

        # Classes
        self.names = self.model.names  # id -> name
        self.name_to_id = {v: k for k, v in self.names.items()}
        missing = [n for n in ["head", "torso", "ht"] if n not in self.name_to_id]
        if missing:
            rospy.logwarn(f"[doll_state_node] Classes manquantes dans le modèle: {missing}.")

    # ---------- Décision par frame ----------
    def decide_frame_state(self, res, img_h):
        """
        Retourne: ('assembled'|'disassembled'|'unknown', reason, debug_objects)
        """
        debug_objs = []

        if res is None or res.boxes is None or len(res.boxes) == 0:
            return "unknown", "no_detections", debug_objs

        # Récup
        classes = res.boxes.cls.detach().cpu().numpy().astype(int)
        confs   = res.boxes.conf.detach().cpu().numpy().astype(float)

        masks = None
        if res.masks is not None and res.masks.data is not None:
            raw = res.masks.data.detach().cpu().numpy()  # (N,H,W) floats
            masks = raw > self.mask_threshold

        # Split par classe + debug list
        head_idxs, torso_idxs, ht_idxs = [], [], []
        for i, c in enumerate(classes):
            name = self.names.get(int(c), f"class_{int(c)}")
            debug_objs.append({"i": int(i), "cls": name, "conf": float(confs[i])})
            if name == "head":
                head_idxs.append(i)
            elif name == "torso":
                torso_idxs.append(i)
            elif name == "ht":
                ht_idxs.append(i)

        # ---- 1) Règle 'ht' (mais avec validation forte) ----
        if ht_idxs:
            best_ht = max(ht_idxs, key=lambda i: confs[i])
            ht_ok = confs[best_ht] >= self.min_conf_ht
            if masks is not None and head_idxs and torso_idxs and ht_ok:
                # valider que ht couvre bien l'union head∪torso
                union = np.zeros_like(masks[0], dtype=bool)
                for hi in head_idxs: 
                    if hi < masks.shape[0]: union |= masks[hi]
                for ti in torso_idxs:
                    if ti < masks.shape[0]: union |= masks[ti]
                ht_mask = masks[best_ht] if best_ht < masks.shape[0] else None
                if ht_mask is not None:
                    if iou(ht_mask, union) < self.ht_union_iou_min:
                        ht_ok = False  # incohérence ht vs union
            if ht_ok:
                return "assembled", "ht_present_validated", debug_objs
            # sinon: on ignore ht (faux positif)

        # Ni head ni torso
        if not head_idxs and not torso_idxs:
            return "unknown", "neither_head_nor_torso", debug_objs

        # ---- 2) Tenter l'assemblage via (head + torso) ----
        if head_idxs and torso_idxs and masks is not None:
            best_score = -1.0
            best_pair = None
            best_eval = None

            for hi in head_idxs:
                if hi >= masks.shape[0]: 
                    continue
                hmask = masks[hi]
                hcent = centroid_from_mask(hmask)
                hbox = bbox_from_mask(hmask)
                harea = area_from_mask(hmask)
                for ti in torso_idxs:
                    if ti >= masks.shape[0]:
                        continue
                    tmask = masks[ti]
                    tcent = centroid_from_mask(tmask)
                    tbox = bbox_from_mask(tmask)
                    tarea = area_from_mask(tmask)

                    # Overlap et distance normalisée
                    ov = overlap_ratio_min(hmask, tmask)
                    if hcent is None or tcent is None:
                        norm_dist = 1e9
                    else:
                        dist = math.hypot(hcent[0]-tcent[0], hcent[1]-tcent[1])
                        # normalisation par la hauteur min de bbox (évite dépendance échelle)
                        def box_h(b):
                            return 0 if b is None else max(1, (b[3]-b[1]+1))
                        scale = max(1.0, min(box_h(hbox), box_h(tbox)))
                        norm_dist = dist / float(scale)

                    score = ov*2.0 + max(0.0, (1.5 - norm_dist))  # pondération simple
                    if score > best_score:
                        best_score = score
                        best_pair = (hi, ti)
                        best_eval = {
                            "ov": ov, 
                            "norm_dist": norm_dist,
                            "hcent": hcent, "tcent": tcent,
                            "harea": harea, "tarea": tarea
                        }

            if best_pair is not None and best_eval is not None:
                ov = best_eval["ov"]
                nd = best_eval["norm_dist"]
                hcent, tcent = best_eval["hcent"], best_eval["tcent"]
                harea, tarea = best_eval["harea"], best_eval["tarea"]

                # Contraintes géométriques fortes
                geom_ok = True
                if hcent is None or tcent is None:
                    geom_ok = False
                else:
                    min_margin_px = self.min_vertical_margin * float(img_h)
                    if not (hcent[1] + min_margin_px < tcent[1]):
                        geom_ok = False  # head doit être au-dessus du torso
                if tarea <= 0 or harea >= self.max_head_to_torso_area_ratio * tarea:
                    geom_ok = False  # head trop grand pour être réaliste

                # Décision assemblé si overlap OU proximité ET géométrie plausible
                if geom_ok and ((ov >= self.overlap_thresh) or (nd <= self.center_dist_factor)):
                    return "assembled", f"head+torso overlap={ov:.2f} nd={nd:.2f} geom_ok", debug_objs
                else:
                    return "disassembled", f"head+torso but separated/geom_bad (ov={ov:.2f}, nd={nd:.2f})", debug_objs

        # ---- 3) Une seule des deux parties détectée -> désassemblé ----
        if (head_idxs and not torso_idxs) or (torso_idxs and not head_idxs):
            return "disassembled", "single_part_only", debug_objs

        return "unknown", "fallback", debug_objs

    # ---------- Hystérésis ----------
    def update_stable_state(self, candidate):
        prev = self.stable_state

        if candidate == "assembled":
            self.assembled_streak += 1
            self.disassembled_streak = 0
        elif candidate == "disassembled":
            self.disassembled_streak += 1
            self.assembled_streak = 0
        else:
            # unknown: on “relâche” doucement
            self.assembled_streak = max(0, self.assembled_streak - self.unknown_decay)
            self.disassembled_streak = max(0, self.disassembled_streak - self.unknown_decay)

        if self.assembled_streak >= self.min_frames_to_assembled:
            self.stable_state = "assembled"
        elif self.disassembled_streak >= self.min_frames_to_disassembled:
            self.stable_state = "disassembled"
        # sinon on garde prev
        return prev, self.stable_state

    # ---------- Overlay ----------
    def overlay_debug(self, img_bgr, state, reason):
        color = (0,255,0) if state=="assembled" else (0,0,255) if state=="disassembled" else (0,255,255)
        cv2.rectangle(img_bgr, (5,5), (560,70), (0,0,0), -1)
        cv2.putText(img_bgr, f"STATE: {state}", (12,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(img_bgr, reason[:70], (12,62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        return img_bgr

    # ---------- Callback ----------
    def cb_image(self, msg):
        self.frame_idx += 1
        if self.process_every_n > 1 and (self.frame_idx % self.process_every_n) != 0:
            return

        # ROS -> OpenCV
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img_h = img_bgr.shape[0]

        # Inference (agnostic_nms=True pour limiter doublons inter-classes)
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

        # Décision instantanée
        candidate, reason, _dbg = self.decide_frame_state(res, img_h)
        # Hystérésis
        prev, stable = self.update_stable_state(candidate)

        # Publish état stable
        self.pub_state.publish(String(data=stable))

        # Debug image (résultats YOLO + overlay)
        if self.pub_debug:
            try:
                vis = res.plot()  # boxes + masks + labels
            except Exception:
                vis = img_bgr.copy()
            vis = self.overlay_debug(vis, stable, f"{candidate} | {reason}")
            out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            out.header = msg.header
            self.pub_debug.publish(out)

        # Logs throttlés
        rospy.loginfo_throttle(1.5, f"[doll_state_node] cand={candidate} -> stable={stable} | "
                                    f"Astreak={self.assembled_streak} Dstreak={self.disassembled_streak} | {reason}")


def main():
    rospy.init_node("doll_state_node")
    DollStateNode()
    rospy.loginfo("[doll_state_node] Démarré.")
    rospy.spin()


if __name__ == "__main__":
    main()

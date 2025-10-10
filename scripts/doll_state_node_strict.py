#!/usr/bin/env python3
# coding: utf-8

import math
import numpy as np
import cv2
import rospy

# Import the original node (same directory / package PYTHONPATH)
import doll_state_node_cord as base


# ---------- small helpers (strict contact tests) ----------
def _neck_gap_px(hbox, tbox):
    """Positive if there is a gap (torso above head bottom); negative means overlap.
    Boxes are [x1,y1,x2,y2] or None.
    """
    if not hbox or not tbox:
        return 1e9
    head_bottom = hbox[3]
    torso_top   = tbox[1]
    return torso_top - head_bottom


def _contact_pixels(hmask, tmask, dilate_iter=1):
    if hmask is None or tmask is None:
        return 0
    k = np.ones((3, 3), np.uint8)
    h = cv2.dilate(hmask.astype(np.uint8), k, iterations=dilate_iter).astype(bool)
    t = cv2.dilate(tmask.astype(np.uint8), k, iterations=dilate_iter).astype(bool)
    return int(np.logical_and(h, t).sum())


def _band_mask_from_box(mask, box, band_px, which="bottom"):
    if mask is None or box is None:
        return None
    x1, y1, x2, y2 = box
    if which == "bottom":
        yb1, yb2 = max(y1, y2 - band_px + 1), y2
    else:  # top
        yb1, yb2 = y1, min(y2, y1 + band_px - 1)
    m = np.zeros_like(mask, dtype=bool)
    m[yb1:yb2+1, x1:x2+1] = mask[yb1:yb2+1, x1:x2+1]
    return m


class DollStateNodeStrict(base.DollStateNode):
    def __init__(self):
        super(DollStateNodeStrict, self).__init__()
        # ------ new strict params ------
        self.neck_gap_px_max = int(rospy.get_param("~neck_gap_px_max", 6))
        self.min_contact_px = int(rospy.get_param("~min_contact_px", 80))
        self.depth_neck_gap_max_m = float(rospy.get_param("~depth_neck_gap_max_m", 0.01))

    # -------------- STRICT decision override --------------
    def decide_frame_state(self, res, img_h):
        """Call base logic first, then enforce strict neck contact.
        The post-filter assumes a single doll in view. If multiple, we pick the best head/torso by confidence.
        """
        # Run the parent's decision (keeps all your existing heuristics + stability)
        parent_state, parent_reason, _, _ = super(DollStateNodeStrict, self).decide_frame_state(res, img_h)

        # If no detections or already disassembled/unknown, we can still try to confirm strict if both parts exist.
        # Extract masks, classes, confidences
        if res is None or res.boxes is None or len(res.boxes) == 0:
            return parent_state, parent_reason, [], None

        classes = res.boxes.cls.detach().cpu().numpy().astype(int)
        confs   = res.boxes.conf.detach().cpu().numpy().astype(float)

        masks = None
        if (getattr(res, 'masks', None) is not None) and (res.masks.data is not None):
            raw = res.masks.data.detach().cpu().numpy()
            masks = raw > self.mask_threshold

        # Find best head & best torso by confidence (assumes one doll)
        head_idx = None
        torso_idx = None
        head_conf = -1.0
        torso_conf = -1.0

        for i, c in enumerate(classes):
            name = self.names.get(int(c), f"class_{int(c)}")
            if name.lower() == 'head' and confs[i] > head_conf:
                head_conf = confs[i]
                head_idx = i
            elif name.lower() == 'torso' and confs[i] > torso_conf:
                torso_conf = confs[i]
                torso_idx = i

        # If we don't have both head and torso, return parent's decision
        if head_idx is None or torso_idx is None or masks is None:
            return parent_state, parent_reason, [], masks

        # Build masks and boxes for strict checks
        hmask = masks[head_idx]
        tmask = masks[torso_idx]
        hbox  = base.bbox_from_mask(hmask)
        tbox  = base.bbox_from_mask(tmask)

        # Geometry: require head above torso as parent does
        geom_ok = False
        if hbox and tbox:
            hcy = (hbox[1] + hbox[3]) * 0.5
            tcy = (tbox[1] + tbox[3]) * 0.5
            min_margin_px = float(self.min_vertical_margin) * float(img_h)
            geom_ok = (hcy + min_margin_px < tcy)

        # Contact pixels & pixel gap
        gap = _neck_gap_px(hbox, tbox)
        cpx = _contact_pixels(hmask, tmask, dilate_iter=1)

        # Depth continuity at the neck band (if depth available)
        depth_ok = True
        have_3d = (self.use_depth and getattr(self, 'depth_image', None) is not None and np.any(np.isfinite(self.depth_image)))
        if have_3d:
            band_px = 5
            head_band = _band_mask_from_box(hmask, hbox, band_px, 'bottom')
            torso_band = _band_mask_from_box(tmask, tbox, band_px, 'top')
            zh = base.median_depth_in_mask(self.depth_image, head_band) if head_band is not None else None
            zt = base.median_depth_in_mask(self.depth_image, torso_band) if torso_band is not None else None
            if (zh is None) or (zt is None):
                depth_ok = False
            else:
                depth_ok = abs(zh - zt) <= self.depth_neck_gap_max_m

        # Also keep some of parent's coarse proximity cues
        # Compute overlap ratio (min of intersections) and normalized center distance
        ov = base.overlap_ratio_min(hmask, tmask)
        # centers
        hcent = base.centroid_from_mask(hmask)
        tcent = base.centroid_from_mask(tmask)
        if hcent is None or tcent is None:
            nd = 1e9
        else:
            dist = math.hypot(hcent[0]-tcent[0], hcent[1]-tcent[1])
            # normalize by min box height to be scale-invariant
            def box_h(b):
                return 0 if not b else max(1, (b[3]-b[1]+1))
            scale = max(1.0, min(box_h(hbox), box_h(tbox)))
            nd = dist / float(scale)

        assembled_strict = (
            geom_ok and
            ((ov >= self.overlap_thresh) or (nd <= self.center_dist_factor)) and
            (gap <= self.neck_gap_px_max) and
            (cpx >= self.min_contact_px) and
            (not have_3d or depth_ok)
        )

        # If the parent already says assembled, we require the strict test to pass too.
        # If the parent says disassembled/unknown but strict passes, we can promote to assembled.
        if assembled_strict:
            reason = f"strict_ok: ov={ov:.2f}, nd={nd:.2f}, gap={gap}px, cpx={cpx}, depth_ok={depth_ok}"
            return "assembled", reason, [], masks
        else:
            # fallback to parent's non-assembled outcome; if parent was assembled, downgrade
            if parent_state == 'assembled':
                reason = f"downgraded: no_strict_contact (ov={ov:.2f}, nd={nd:.2f}, gap={gap}px, cpx={cpx}, depth_ok={depth_ok})"
                return "disassembled", reason, [], masks
            return parent_state, parent_reason, [], masks


# ---------- main ----------
def main():
    rospy.init_node("doll_state_node_strict")
    DollStateNodeStrict()
    rospy.loginfo("[doll_state_node_strict] Démarré (strict contact mode).")
    rospy.spin()


if __name__ == "__main__":
    main()

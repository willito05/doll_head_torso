#!/usr/bin/env python3
# coding: utf-8
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

def depth_to_meters(depth_arr):
    if depth_arr.dtype == np.uint16:
        return depth_arr.astype(np.float32) / 1000.0
    return depth_arr.astype(np.float32)

class DepthBgFilter:
    def __init__(self):
        self.bridge = CvBridge()
        # Params
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        # Bande de profondeur gardée (en mètres)
        self.z_min = float(rospy.get_param("~z_min", 0.25))
        self.z_max = float(rospy.get_param("~z_max", 0.60))
        # Post-traitement masque
        self.morph_kernel = int(rospy.get_param("~morph_kernel", 5))
        self.blur = int(rospy.get_param("~blur", 0))  # 0 = désactivé

        # Pubs
        self.pub_fg = rospy.Publisher("~fg_only", Image, queue_size=1)
        self.pub_mask = rospy.Publisher("~mask", Image, queue_size=1)
        self.pub_debug = rospy.Publisher("~debug_image", Image, queue_size=1)

        # Subs
        self.color_msg = None
        self.depth_msg = None
        rospy.Subscriber(self.color_topic, Image, self.cb_color, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.depth_topic, Image, self.cb_depth, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_caminfo, queue_size=1)

        self.frame_id = None

    def cb_caminfo(self, msg):
        self.frame_id = msg.header.frame_id

    def cb_color(self, msg):
        self.color_msg = msg
        self.try_process()

    def cb_depth(self, msg):
        self.depth_msg = msg
        self.try_process()

    def try_process(self):
        if self.color_msg is None or self.depth_msg is None:
            return

        color = self.bridge.imgmsg_to_cv2(self.color_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
        depth_m = depth_to_meters(depth)

        # Masque profondeur
        mask = (depth_m >= self.z_min) & (depth_m <= self.z_max) & np.isfinite(depth_m)
        mask = mask.astype(np.uint8) * 255

        # Nettoyage morpho
        if self.morph_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # Optionnel : léger flou du FG pour lisser les bords
        fg = color.copy()
        fg[mask == 0] = 0
        if self.blur > 0:
            fg = cv2.GaussianBlur(fg, (self.blur | 1, self.blur | 1), 0)

        # Debug overlay
        dbg = color.copy()
        dbg[mask == 0] = (0, 0, 0)
        cv2.rectangle(dbg, (5,5), (420, 60), (0,0,0), -1)
        txt = f"Depth keep: {self.z_min:.2f}m..{self.z_max:.2f}m"
        cv2.putText(dbg, txt, (12,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

        # Publish
        out_fg = self.bridge.cv2_to_imgmsg(fg, "bgr8")
        out_fg.header = self.color_msg.header
        if self.frame_id:
            out_fg.header.frame_id = self.frame_id
        self.pub_fg.publish(out_fg)

        out_mask = self.bridge.cv2_to_imgmsg(mask, "mono8")
        out_mask.header = self.color_msg.header
        self.pub_mask.publish(out_mask)

        out_dbg = self.bridge.cv2_to_imgmsg(dbg, "bgr8")
        out_dbg.header = self.color_msg.header
        self.pub_debug.publish(out_dbg)

def main():
    rospy.init_node("depth_bg_filter_node")
    DepthBgFilter()
    rospy.loginfo("[depth_bg_filter_node] Démarré (filtre fond par profondeur).")
    rospy.spin()

if __name__ == "__main__":
    main()

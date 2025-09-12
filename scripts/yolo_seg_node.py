#!/usr/bin/env python3
import os
import rospy
import torch
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO

class YoloSegNode:
    def __init__(self):
        self.bridge = CvBridge()

        # Params
        self.model_path = rospy.get_param("~model_path", "weights/best.pt")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.conf = float(rospy.get_param("~conf", 0.25))
        self.iou  = float(rospy.get_param("~iou", 0.5))
        self.device = rospy.get_param("~device", "0")  # "0" pour GPU, "cpu" sinon
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.process_every_n = int(rospy.get_param("~process_every_n", 1))  # traite 1 frame sur n

        # Modèle
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            rospy.loginfo(f"[yolo_seg_node] Chargé: {self.model_path} sur device={self.device}")
        except Exception as e:
            rospy.logerr(f"[yolo_seg_node] Erreur chargement modèle: {e}")
            raise

        self.pub_annot = rospy.Publisher("~annotated", Image, queue_size=1)
        self.frame_idx = 0

        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1, buff_size=2**24)

    def cb_image(self, msg):
        self.frame_idx += 1
        if self.process_every_n > 1 and (self.frame_idx % self.process_every_n) != 0:
            return

        # ROS -> OpenCV (BGR)
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Inference
        try:
            # Ultralytics accepte BGR/ndarray directement
            res = self.model.predict(
                source=img_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False
            )[0]
        except Exception as e:
            rospy.logwarn(f"[yolo_seg_node] Erreur inference: {e}")
            return

        # Visualisation: résulats + masques projetés
        vis = res.plot()  # ndarray BGR avec boxes + masks + labels

        # Publie l’image annotée
        out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out.header = msg.header
        self.pub_annot.publish(out)

        # Log rapide
        if len(res.boxes) > 0:
            classes = self.model.names
            dets = [classes[int(c)] for c in res.boxes.cls]
            rospy.loginfo_throttle(2.0, f"[yolo_seg_node] Detections: {dets}")

def main():
    rospy.init_node("yolo_seg_node")
    YoloSegNode()
    rospy.loginfo("[yolo_seg_node] Démarré.")
    rospy.spin()

if __name__ == "__main__":
    main()

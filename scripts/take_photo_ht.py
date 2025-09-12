#!/usr/bin/env python3
import os, rospy, argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class PhotoTakerHT:
    def __init__(self, save_dir, rate_hz, max_count):
        self.bridge = CvBridge()
        self.save_dir = save_dir
        self.rate_hz  = rate_hz
        self.max_count = max_count
        os.makedirs(self.save_dir, exist_ok=True)
        self.count = 1
        self.last_img = None

        # Récupère chaque image, mais n'enregistre qu'au rythme voulu
        rospy.Subscriber('/camera/color/image_raw', Image, self.img_cb)
        self.timer = rospy.Timer(rospy.Duration(1.0/self.rate_hz), self.timer_cb)
        rospy.loginfo(f"[PhotoTakerHT] Sauvegarde dans {self.save_dir}, {self.rate_hz} Hz, max {self.max_count} images")

    def img_cb(self, msg):
        # On stocke toujours la dernière image reçue
        self.last_img = msg

    def timer_cb(self, event):
        if self.last_img is None or (self.max_count and self.count>self.max_count):
            if self.count>self.max_count:
                rospy.loginfo("[PhotoTakerHT] Atteint le nombre max, je stoppe.")
                rospy.signal_shutdown("Max reached")
            return

        cv_img = self.bridge.imgmsg_to_cv2(self.last_img, 'bgr8')
        fname = f"torso_{self.count:04d}.png"
        path = os.path.join(self.save_dir, fname)
        cv2.imwrite(path, cv_img)
        rospy.loginfo(f"[PhotoTakerHT] Sauvegardé {path}")
        self.count += 1

if __name__=='__main__':
    rospy.init_node('take_photos_ht')
    p = rospy.get_param
    save_dir  = p('~save_dir','~/doll/doll_2parts/photos')
    rate_hz   = float(p('~rate',1.0))
    max_count = int(p('~max_count',0))  # 0 = illimité
    PhotoTakerHT(save_dir, rate_hz, max_count)
    rospy.spin()

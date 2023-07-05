import rospy

from sensor_msgs.msg import Image
from yolov8_ros.srv import Yolov8, Yolov8Request
from yolov8_ros.msg import Box, Boxes
from cv_bridge import CvBridge

import cv2
import torch
import random

from ultralytics import YOLO
from ultralytics.tracker import BOTSORT, BYTETracker
from ultralytics.tracker.trackers.basetrack import BaseTrack
from ultralytics.yolo.utils import IterableSimpleNamespace, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_yaml
from ultralytics.yolo.engine.results import Results


class Yolo_service_tester():
    def __init__(self):
        rospy.init_node("yolov8_service_caller",anonymous=True)

        self.model_name = '' # the server uses "yolov8m-seg.pt" if empty or wrong name. 
                             # If you only want to detect persons and have their skeleton, 
                             # use "yolov8m-pose.pt" 
        self.class_to_detect = [] # the server detects all classes if empty list. 
                                  # If you want particular classes, look at the ReadMe to get the 
                                  # int corresponding to the classes you want. 
                                  # Example : "self.class_to_detect = [0,1]" if you want to detect persons and cars

        self.image_to_treat = Image() # the servers uses the current image if empty.


        self.req = Yolov8Request()
        self.req.model_name = self.model_name
        self.req.classes = self.class_to_detect
        self.image = self.image_to_treat

        self.client_ = rospy.ServiceProxy('yolov8_on_unique_frame',Yolov8)
        self.cv2_pub = rospy.Publisher("yolov8_image_with_bboxes", Image, queue_size=10)
        self.yolo_basic = YOLO("yolov8m.pt")
        self._class_to_color={}
        self.cv_bridge=CvBridge()
        self.analyse_photo()
        rospy.spin()




    def send_request(self):
        rospy.wait_for_service('yolov8_on_unique_frame')
        try :
            rospy.loginfo("Waiting for yolov8 service")
            bbox_list = self.client_(self.req)
            rospy.loginfo(bbox_list.boxes) 
        except:
            rospy.logwarn("Service call didn't work")

    def analyse_photo(self):
        cv_image = cv2.imread("./arena-couch/image_7.png")

        results = self.yolo_basic.predict(source=cv_image,verbose=False,stream=False,conf=0.5,mode="track", classes=[0,56,57])
        results: Results = results[0].cpu()
        

        boxes = Boxes()

        for box_data in results.boxes:

            box = Box()
            if box_data.id!=None and int(box_data.id)<=65535:
                box.ID = int(box_data.id)
            box.bbox_class = results.names[int(box_data.cls)]
            box.probability = float(box_data.conf)
            box.xmin = float(min(box_data.xyxy[0][0], box_data.xyxy[0][2]))
            box.ymin = float(min(box_data.xyxy[0][1], box_data.xyxy[0][3]))
            box.xmax = float(max(box_data.xyxy[0][0], box_data.xyxy[0][2]))
            box.ymax = float(max(box_data.xyxy[0][1], box_data.xyxy[0][3]))

            boxes.boxes.append(box)

        
            if box.bbox_class not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                box_data = random.randint(0, 255)
                self._class_to_color[box.bbox_class] = (r, g, box_data)
            color = self._class_to_color[box.bbox_class]
            min_pt = (int(box.xmin),int(box.ymin))
            max_pt =  (int(box.xmax),int(box.ymax))
            cv2.rectangle(cv_image, min_pt, max_pt, color, 2)
            label = "{} ({}) ({:.3f})".format(box.bbox_class, str(box.ID), box.probability)
            pos = (min_pt[0] + 5, min_pt[1] + 25)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, label, pos, font,
                        1, color, 1, cv2.LINE_AA)

        
        self.cv2_pub.publish((self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                               encoding='bgr8')))

if __name__=="__main__":
    a=Yolo_service_tester()
    # a.send_request()
   
    




            

     
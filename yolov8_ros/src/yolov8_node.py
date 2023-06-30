import rospy

import cv2
import torch
import random

from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.tracker import BOTSORT, BYTETracker
from ultralytics.tracker.trackers.basetrack import BaseTrack
from ultralytics.yolo.utils import IterableSimpleNamespace, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_yaml
from ultralytics.yolo.engine.results import Results

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from yolov8_ros.msg import Box, Boxes
from yolov8_ros.srv import Yolov8
#from vision_msgs.msg import Detection2D
#from vision_msgs.msg import ObjectHypothesisWithPose
#from vision_msgs.msg import Detection2DArray


class Yolo_ros():
    def __init__(self):
        self.yolov8_node()

    def create_tracker(self,tracker_yaml):

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in ["bytetrack", "botsort"], \
            f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker
      
    def image_cb(self,msg):

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

        results = self.yolo_continuous_treatment.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
        results: Results = results[0].cpu()

        # tracking
        det = results.boxes.numpy()

        if len(det) > 0:
            im0s = self.yolo_continuous_treatment.predictor.batch[2]
            im0s = im0s if isinstance(im0s, list) else [im0s]
            tracks = self.tracker.update(det, im0s[0])
            if len(tracks) > 0:
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))
        
        boxes = Boxes()
            
        for box_data in results.boxes:

            box = Box()
            box.ID = int(box_data.id)
            box.xyxy = box_data.xyxy[0]
            boxes.boxes.append(box)
        
        for box in boxes.boxes:
            x_min = min(box.xyxy[0], box.xyxy[2])
            y_min = min(box.xyxy[1], box.xyxy[3])
            x_max = max(box.xyxy[0], box.xyxy[2])
            y_max = max(box.xyxy[1], box.xyxy[3])
            
            for p in results.keypoints.data :
                is_in_box = True
                point_list = []
                for i in p:
                    point = Point()
                    point.x = i[0]
                    point.y = i[1]
                    point.z = i[2]
                    point_list.append(point)
                    if (point.x< x_min or point.x > x_max or point.y< y_min or point.y > y_max):
                        is_in_box = False
                        break
                if is_in_box:
                    box.skeleton=point_list
                    break

        rospy.loginfo("Continuous model publishing boxes. Number of boxes detected : " + str(len(boxes.boxes)))                            
        self.result_pub.publish(boxes)

    def image_cb_1(self,msg):

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

        results = self.yolo_1.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
        results: Results = results[0].cpu()

        # tracking
        det = results.boxes.numpy()

        if len(det) > 0:
            im0s = self.yolo_1.predictor.batch[2]
            im0s = im0s if isinstance(im0s, list) else [im0s]
            tracks = self.tracker.update(det, im0s[0])
            if len(tracks) > 0:
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))

        boxes = Boxes()

        for box_data in results.boxes:

            box = Box()
            box.ID = int(box_data.id)
            box.xyxy = box_data.xyxy[0]
            boxes.boxes.append(box)
        rospy.loginfo("Model 1 publishing boxes. Number of boxes detected : " + str(len(boxes.boxes)))
        self.result_pub_1.publish(boxes)


    def image_cb_2(self,msg):

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

        results = self.yolo_2.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
        results: Results = results[0].cpu()

        # tracking
        det = results.boxes.numpy()

        if len(det) > 0:
            im0s = self.yolo_2.predictor.batch[2]
            im0s = im0s if isinstance(im0s, list) else [im0s]
            tracks = self.tracker.update(det, im0s[0])
            if len(tracks) > 0:
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))

        boxes = Boxes()

        for box_data in results.boxes:

            box = Box()
            box.ID = int(box_data.id)
            box.xyxy = box_data.xyxy[0]
            boxes.boxes.append(box)
        rospy.loginfo("Model 2 publishing boxes. Number of boxes detected : " + str(len(boxes.boxes)))
        self.result_pub_2.publish(boxes)

            
    def yolov8_on_unique_frame_cb(self, req):

        cv_image = self.cv_bridge.imgmsg_to_cv2(req.image)

        results = self.yolo.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
        results: Results = results[0].cpu()

        # tracking
        det = results.boxes.numpy()

        if len(det) > 0:
            im0s = self.yolo.predictor.batch[2]
            im0s = im0s if isinstance(im0s, list) else [im0s]
            tracks = self.tracker.update(det, im0s[0])
            if len(tracks) > 0:
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))
        
        boxes = Boxes()
        
        # rospy.loginfo("keypoints : " + str(results.keypoints))
        

        for box_data in results.boxes:

            # rospy.loginfo("boxe ID = " + str(int(box_data.id)))
            rospy.loginfo("boxe bound = " + str(box_data.xyxy))
            box = Box()
            box.ID = int(box_data.id)
            box.xyxy = box_data.xyxy[0]
            boxes.boxes.append(box)
        
        for box in boxes.boxes:
            x_min = min(box.xyxy[0], box.xyxy[2])
            y_min = min(box.xyxy[1], box.xyxy[3])
            x_max = max(box.xyxy[0], box.xyxy[2])
            y_max = max(box.xyxy[1], box.xyxy[3])
            
            for p in results.keypoints.data :
                is_in_box = True
                point_list = []
                for i in p:
                    point = Point()
                    point.x = i[0]
                    point.y = i[1]
                    point.z = i[2]
                    point_list.append(point)
                    if ((point.x< x_min or point.x > x_max or point.y< y_min or point.y > y_max) and point.z >= 0.5):
                        is_in_box = False
                        break
                if is_in_box:
                    box.skeleton=point_list
                    break
        
        return boxes





    def yolov8_node(self):

        rospy.init_node('yolov8')

        # params
        rospy.set_param("model", "yolov8m-pose.pt")
        self.model_continuous_treatment = rospy.get_param("model")
        
        rospy.set_param("tracker", "bytetrack.yaml")
        tracker = rospy.get_param("tracker")
    
        rospy.set_param("device", "cuda:0")
        self.device = rospy.get_param("device")
        
        rospy.set_param("threshold", 0.5)
        self.threshold = rospy.get_param("threshold")
    
        _class_to_color = {}
        self.cv_bridge = CvBridge()
        rospy.loginfo("Creatings tracker\n")
        self.tracker = self.create_tracker(tracker)
            
        rospy.loginfo("Creatings models\n")
        self.yolo_continuous_treatment = YOLO(self.model_continuous_treatment)
        rospy.loginfo("First model created")
        self.yolo_1 = YOLO("yolov8m-seg.pt")
        rospy.loginfo("Second model created")
        self.yolo_2 = YOLO("yolov8m.pt")
        rospy.loginfo("Third model created")
        self.cv_bridge=CvBridge()

        # topcis
        # _pub = rospy.Publisher("detections",Detection2DArray, queue_size=10)
        # _dbg_pub = rospy.Publisher("dbg_image",Image,  10)
        video_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb)
        self.result_pub = rospy.Publisher("yolov8_result", Boxes, queue_size=10)
        
        video_sub_1 = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb_1)
        self.result_pub_1 = rospy.Publisher("yolov8_result_1", Boxes, queue_size=10)
        
        video_sub_2 = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb_2)
        self.result_pub_2 = rospy.Publisher("yolov8_result_2", Boxes, queue_size=10)

        #service 
        self.yolov8_srv = rospy.Service('yolov8_on_unique_frame', Yolov8, self.yolov8_on_unique_frame_cb)
        
        # list of already existing models
        existing_models = []
        existing_models.append("yolov8m-pose.pt")

        rospy.spin()
    
    
if __name__ == '__main__':
    
    node = Yolo_ros()
    
    




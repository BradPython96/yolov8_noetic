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

from yolov8_ros.msg import Box, Boxes, SkeletonPoint
from yolov8_ros.srv import Yolov8, Yolov8Response
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

        self.last_image = msg

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        header = msg.header
        header.stamp = rospy.Time.now()

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
        
        if len(results.boxes)>0:
            boxes = Boxes()
            for box_data in results.boxes:

                box = Box()
                if box_data.id!=None:
                    box.ID = int(box_data.id)
                box.bbox_class = results.names[int(box_data.cls)]
                box.probability = float(box_data.conf)
                box.xmin = float(min(box_data.xyxy[0][0], box_data.xyxy[0][2]))
                box.ymin = float(min(box_data.xyxy[0][1], box_data.xyxy[0][3]))
                box.xmax = float(max(box_data.xyxy[0][0], box_data.xyxy[0][2]))
                box.ymax = float(max(box_data.xyxy[0][1], box_data.xyxy[0][3]))

                boxes.boxes.append(box)

            if results.keypoints != None:

                for box in boxes.boxes:
                    x_min = box.xmin
                    y_min = box.ymin
                    x_max = box.xmax
                    y_max = box.ymax

                    for p in results.keypoints.data :
                        is_in_box = True
                        skeleton = []
                        for i in p:
                            skeleton_point = SkeletonPoint()
                            skeleton_point.x = i[0]
                            skeleton_point.y = i[1]
                            skeleton_point.conf = i[2]
                            skeleton.append(skeleton_point)
                            if (skeleton_point.conf > 0.5 and (skeleton_point.x< x_min or skeleton_point.x > x_max or skeleton_point.y< y_min or skeleton_point.y > y_max)):
                                is_in_box = False
                                break
                        if is_in_box:
                            box.skeleton=skeleton
                            break

            # rospy.loginfo("Publishing boxes. Number of boxes detected : " + str(len(boxes.boxes)))                            
            self.result_pub.publish(boxes)


        # if self.i == 1:
        #     rospy.loginfo(results.names)
        #     self.first_image_time = rospy.Time.now()
        # else:
        #     delta_t = (rospy.Time.now().to_nsec()) - (self.first_image_time.to_nsec())
        #     rospy.loginfo("Image treated per sec : " + str((self.i/delta_t)*1000000000))
        # self.i += 1

    def image_cb_1(self,msg):

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

        results = self.yolo_seg.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
        results: Results = results[0].cpu()

        # tracking
        det = results.boxes.numpy()

        if len(det) > 0:
            im0s = self.yolo_seg.predictor.batch[2]
            im0s = im0s if isinstance(im0s, list) else [im0s]
            tracks = self.tracker.update(det, im0s[0])
            if len(tracks) > 0:
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))

        if len(results.boxes)>0:
            boxes = Boxes()
            for box_data in results.boxes:

                box = Box()
                if box_data.id!=None:
                    box.ID = int(box_data.id)
                box.bbox_class = results.names[int(box_data.cls)]
                box.probability = float(box_data.conf)
                box.xmin = float(min(box_data.xyxy[0][0], box_data.xyxy[0][2]))
                box.ymin = float(min(box_data.xyxy[0][1], box_data.xyxy[0][3]))
                box.xmax = float(max(box_data.xyxy[0][0], box_data.xyxy[0][2]))
                box.ymax = float(max(box_data.xyxy[0][1], box_data.xyxy[0][3]))

                boxes.boxes.append(box)

            if results.masks != None:

                for box in boxes.boxes:
                    x_min = box.xmin
                    y_min = box.ymin
                    x_max = box.xmax
                    y_max = box.ymax
                    for p in results.masks.xy :
                        is_in_box = True
                        points_list = []
                        for i in p:
                            seg = Point()
                            seg.x = i[0]
                            seg.y = i[1]
                            points_list.append(seg)
                            if (seg.x< x_min or seg.x > x_max or seg.y< y_min or seg.y > y_max):
                                is_in_box = False
                                break
                        if is_in_box:
                            box.points_in_seg = points_list
                            break


            
            # rospy.loginfo(results.masks.xy)
            # rospy.loginfo("nb de boxes : "+ str(len(results.boxes)))
            # rospy.loginfo("taille totale : " + str(len(results.masks.data)) + "\n taille [] : " + str(len(results.masks.data[0])) + "\n taille [][] : " + str(len(results.masks.data[0][0])))
            self.result_pub_1.publish(boxes)



    def image_cb_2(self,msg):

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

        results = self.yolo_basic.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
        results: Results = results[0].cpu()

        # tracking
        det = results.boxes.numpy()

        if len(det) > 0:
            im0s = self.yolo_basic.predictor.batch[2]
            im0s = im0s if isinstance(im0s, list) else [im0s]
            tracks = self.tracker.update(det, im0s[0])
            if len(tracks) > 0:
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))

        if len(results.boxes)>0:
            boxes = Boxes()
            for box_data in results.boxes:

                box = Box()
                if box_data.id!=None:
                    box.ID = int(box_data.id)
                box.bbox_class = results.names[int(box_data.cls)]
                box.probability = float(box_data.conf)
                box.xmin = float(min(box_data.xyxy[0][0], box_data.xyxy[0][2]))
                box.ymin = float(min(box_data.xyxy[0][1], box_data.xyxy[0][3]))
                box.xmax = float(max(box_data.xyxy[0][0], box_data.xyxy[0][2]))
                box.ymax = float(max(box_data.xyxy[0][1], box_data.xyxy[0][3]))

                boxes.boxes.append(box)
            
            self.result_pub_2.publish(boxes)

            
    def yolov8_on_unique_frame_cb(self, req):

        rospy.loginfo("Service received")
        header = req.image.header
        header.stamp = rospy.Time.now()

        if req.image.height==0:
            image = self.last_image
        else :
            image = req.image
        
        cv_image = self.cv_bridge.imgmsg_to_cv2(image)


        cls = []
        for cl in req.classes:
            cls.append(int(cl))

        if len(cls)>0:
            rospy.loginfo("classes :" + str(cls))
        else :
            rospy.loginfo("All classes")
            cls=[i for i in range(80)]
                
        if req.model_name=="yolov8m-pose.pt":
            results = self.yolo_pose.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")    

        else :
            results = self.yolo_seg.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track", classes=cls)

        # else:
        #     results = self.yolo_basic.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track", classes=cls)

        results: Results = results[0].cpu()

        # tracking
        det = results.boxes.numpy()

        if len(det) > 0:
            if req.model_name=="yolov8m-pose.pt":
                im0s = self.yolo_pose.predictor.batch[2]
            elif req.model_name=="yolov8m-seg.pt":
                im0s = self.yolo_seg.predictor.batch[2]
            else :
                im0s = self.yolo_basic.predictor.batch[2]
            
            im0s = im0s if isinstance(im0s, list) else [im0s]
            tracks = self.tracker.update(det, im0s[0])
            if len(tracks) > 0:
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))

        boxes = Boxes()

        for box_data in results.boxes:

            box = Box()
            if box_data.id!=None:
                box.ID = int(box_data.id)
            box.bbox_class = results.names[int(box_data.cls)]
            box.probability = float(box_data.conf)
            box.xmin = float(min(box_data.xyxy[0][0], box_data.xyxy[0][2]))
            box.ymin = float(min(box_data.xyxy[0][1], box_data.xyxy[0][3]))
            box.xmax = float(max(box_data.xyxy[0][0], box_data.xyxy[0][2]))
            box.ymax = float(max(box_data.xyxy[0][1], box_data.xyxy[0][3]))

            boxes.boxes.append(box)
        
        if results.keypoints != None:
            for box in boxes.boxes:
                x_min = box.xmin
                y_min = box.ymin
                x_max = box.xmax
                y_max = box.ymax
                
                for p in results.keypoints.data :
                    is_in_box = True
                    skeleton = []
                    for i in p:
                        skeleton_point = SkeletonPoint()
                        skeleton_point.x = i[0]
                        skeleton_point.y = i[1]
                        skeleton_point.conf = i[2]
                        skeleton.append(skeleton_point)
                        if (skeleton_point.conf > 0.5 and (skeleton_point.x< x_min or skeleton_point.x > x_max or skeleton_point.y< y_min or skeleton_point.y > y_max)):
                            is_in_box = False
                            break
                    if is_in_box:
                        box.skeleton=skeleton
                        break

        if results.masks != None:

            for box in boxes.boxes:
                x_min = box.xmin
                y_min = box.ymin
                x_max = box.xmax
                y_max = box.ymax
                for p in results.masks.xy :
                    is_in_box = True
                    points_list = []
                    for i in p:
                        seg = Point()
                        seg.x = i[0]
                        seg.y = i[1]
                        points_list.append(seg)
                        if (seg.x< x_min or seg.x > x_max or seg.y< y_min or seg.y > y_max):
                            is_in_box = False
                            break
                    if is_in_box:
                        box.points_in_seg = points_list
                        break
        # rospy.loginfo("Response sent : " + str(boxes))
        return Yolov8Response(boxes.boxes)


    def yolov8_node(self):

        rospy.init_node('yolov8')

        # params
        rospy.set_param("model", "yolov8m-pose.pt")
        model_continuous_treatment = rospy.get_param("model")
        
        rospy.set_param("tracker", "bytetrack.yaml")
        tracker = rospy.get_param("tracker")
    
        rospy.set_param("device", "cuda:0")
        self.device = rospy.get_param("device")
        
        rospy.set_param("threshold", 0.5)
        self.threshold = rospy.get_param("threshold")
    
        self.cv_bridge = CvBridge()
        rospy.loginfo("Creatings tracker\n")
        self.tracker = self.create_tracker(tracker)
            
        rospy.loginfo("Creatings models\n")

        # self.yolo_continuous_treatment = YOLO(self.model_continuous_treatment)

        self.yolo_pose = YOLO("yolov8m-pose.pt")
        rospy.loginfo("First model created")
        self.yolo_seg = YOLO("yolov8m-seg.pt")
        rospy.loginfo("Second model created")
        self.yolo_basic = YOLO("yolov8m.pt")
        rospy.loginfo("Third model created")


        if model_continuous_treatment=="yolov8m-pose.pt":
            self.yolo_continuous_treatment = self.yolo_pose
        elif model_continuous_treatment=="yolov8m-seg.pt":
            self.yolo_continuous_treatment = self.yolo_seg
        elif model_continuous_treatment=="yolov8m.pt":
            self.yolo_continuous_treatment = self.yolo_basic

        self.cv_bridge=CvBridge()

        # counter
        self.i = 1

        self.last_image = Image()

        # topcis
        # _pub = rospy.Publisher("detections",Detection2DArray, queue_size=10)
        # _dbg_pub = rospy.Publisher("dbg_image",Image,  10)
        video_sub = rospy.Subscriber("/kinect2/hd/image_color", Image, self.image_cb, queue_size=1)
        self.result_pub = rospy.Publisher("yolov8_result", Boxes, queue_size=10)
        
        video_sub_1 = rospy.Subscriber("/kinect2/hd/image_color", Image, self.image_cb_1, queue_size=1)
        self.result_pub_1 = rospy.Publisher("yolov8_result_1", Boxes, queue_size=10)
        
        video_sub_2 = rospy.Subscriber("/kinect2/hd/image_color", Image, self.image_cb_2, queue_size=1)
        self.result_pub_2 = rospy.Publisher("yolov8_result_2", Boxes, queue_size=10)

        #service 
        self.yolov8_srv = rospy.Service('yolov8_on_unique_frame', Yolov8, self.yolov8_on_unique_frame_cb)
        
        # # list of already existing models
        # existing_models = []
        # existing_modeBoxesls.append("yolov8m-pose.pt")

        rospy.spin()
    
    
if __name__ == '__main__':
    
    node = Yolo_ros()
    
    




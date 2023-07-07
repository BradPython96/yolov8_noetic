
from yolov8_ros_msgs.srv import Yolov8, Yolov8Response

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

from yolov8_ros_msgs.msg import Box, Boxes, SkeletonPoint



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

        try :
            self.last_image = msg

            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            

            results = self.yolo_pose.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
            results: Results = results[0].cpu()

            # tracking
            det = results.boxes.numpy()

            if len(det) > 0:
                im0s = self.yolo_pose.predictor.batch[2]
                im0s = im0s if isinstance(im0s, list) else [im0s]
                tracks = self.tracker.update(det, im0s[0])
                if len(tracks) > 0:
                    results.update(boxes=torch.as_tensor(tracks[:, :-1]))
            #

            if len(results.boxes)>0:
                boxes = Boxes()
                boxes.header.seq=self.i_cb_pose
                boxes.header.stamp = rospy.Time.now()
                boxes.header.frame_id = msg.header.frame_id

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
                        
                # Display of  skeletons.
                # i = 0
                # for p in results.keypoints.data:
                #     for l in p :

                #         color = (0,0,0)
                #         min_pt = (int(l[0]-5),int(l[1]-5))
                #         max_pt =  (int(l[0]+5),int(l[1]+5))

                #         cv2.rectangle(cv_image, min_pt, max_pt, color, 2)
                #         label = "{}".format(str(i))
                #         pos = (min_pt[0] + 5, min_pt[1] + 25)
                #         font = cv2.FONT_HERSHEY_SIMPLEX
                #         cv2.putText(cv_image, label, pos, font,
                #                     1, color, 1, cv2.LINE_AA)
                #         i+=1
     
                # self.cv2_pub.publish((self.cv_bridge.cv2_to_imgmsg(cv_image,encoding=msg.encoding)))
                                               
                self.result_pub.publish(boxes)
                self.i_cb_pose+=1

                # if self.i_pose == 1:
                #     rospy.loginfo(results.names)
                #     self.first_image_time = rospy.Time.now()
                # else:
                #     delta_t = (rospy.Time.now().to_nsec()) - (self.first_image_time.to_nsec())
                #     rospy.loginfo("Image treated per sec : " + str((self.i_pose/delta_t)*1000000000))
        except:
            rospy.logerr("[Yolo_ros] Wrong images sent by the Kinect.")

    def image_cb_1(self,msg):

        try :
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

            results = self.yolo_seg.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
            results: Results = results[0].cpu()

            # # tracking
            # det = results.boxes.numpy()

            # if len(det) > 0:
            #     im0s = self.yolo_seg.predictor.batch[2]
            #     im0s = im0s if isinstance(im0s, list) else [im0s]
            #     tracks = self.tracker.update(det, im0s[0])
            #     if len(tracks) > 0:
            #         results.update(boxes=torch.as_tensor(tracks[:, :-1]))

            if len(results.boxes)>0:
                boxes = Boxes()
                boxes.header.seq=self.i_cb_seg
                boxes.header.stamp = rospy.Time.now()
                boxes.header.frame_id = msg.header.frame_id

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

                self.result_pub_1.publish(boxes)
                self.i_cb_seg+=1
        except:
            rospy.logerr("[Yolo_ros] Wrong images sent by the Kinect.")
    
    def image_cb_2(self,msg):

        try :
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

            results = self.yolo_basic.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")
            results: Results = results[0].cpu()

            # # tracking
            # det = results.boxes.numpy()

            # if len(det) > 0:
            #     im0s = self.yolo_basic.predictor.batch[2]
            #     im0s = im0s if isinstance(im0s, list) else [im0s]
            #     tracks = self.tracker.update(det, im0s[0])
            #     if len(tracks) > 0:
            #         results.update(boxes=torch.as_tensor(tracks[:, :-1]))

            if len(results.boxes)>0:
                boxes = Boxes()
                boxes.header.seq=self.i_cb_basic
                boxes.header.stamp = rospy.Time.now()
                boxes.header.frame_id = msg.header.frame_id
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

                self.result_pub_2.publish(boxes)
                self.i_cb_basic+=1
        except:
            rospy.logerr("[Yolo_ros] Wrong images sent by the Kinect.")
            
    def yolov8_on_unique_frame_cb(self, req):

        rospy.loginfo("[Yolo_ros] Service received.")
        

        if req.image == '' or  req.image == [] or req.image.encoding == '' or req.image.height==0 or req.image.data==[]:
            req.image = None

        if req.image == None:
            if self.last_image == None:
                rospy.logwarn("[Yolo_ros] Unable to get the first image from kinect. Returning error box.")
                error_boxes = Boxes()
                error_box = Box()
                error_box.bbox_class = "Error box"
                error_boxes.boxes.append(box)
                return (error_boxes)
            
            image = self.last_image
            
        else :
            image = req.image
        
        cv_image = self.cv_bridge.imgmsg_to_cv2(image)
        
        cls = []
        
        for cl in req.classes:
            if cl >=0 and cl <= 79:
                try :
                    cls.append(int(cl))
                except:
                    rospy.logwarn("[Yolo_ros] One of the classes asked is not an int. Ignoring it.")
            else :
                rospy.logwarn("[Yolo_ros] One of the classes asked is not in [0;79]. Ignoring it.")


        if len(cls)>0:
            rospy.loginfo("classes : " + str(cls))
        elif req.model_name=="yolov8m-pose.pt":
            rospy.loginfo("classes : [0], yolov8m-pose only detects persons")
        else :
            rospy.loginfo("All classes")
            cls=[i for i in range(80)]
                
        if req.model_name=="yolov8m-pose.pt":
            rospy.loginfo("Model : yolov8m-pose.pt\n")
            results = self.yolo_pose.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track")    

        elif  req.model_name=="yolov8m-seg.pt":
            rospy.loginfo("Model : yolov8m-seg.pt\n")
            results = self.yolo_seg.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track", classes=cls)

        else:
            rospy.loginfo("Model : yolov8m.pt\n")
            results = self.yolo_basic.predict(source=cv_image,verbose=False,stream=False,conf=self.threshold,mode="track", classes=cls)

        
        results: Results = results[0].cpu()
        
        # # tracking
        # det = results.boxes.numpy()

        # if len(det) > 0:
        #     if req.model_name=="yolov8m-pose.pt":
        #         im0s = self.yolo_pose.predictor.batch[2]
        #     else  :
        #         im0s = self.yolo_basic.predictor.batch[2]
        #     # else :
        #         # im0s = self.yolo_basic.predictor.batch[2]
            
        #     im0s = im0s if isinstance(im0s, list) else [im0s]
        #     tracks = self.tracker.update(det, im0s[0])
        #     if len(tracks) > 0:
        #         results.update(boxes=torch.as_tensor(tracks[:, :-1]))

        boxes = Boxes()
        boxes.header.seq=self.i_service
        boxes.header.stamp = rospy.Time.now()
        boxes.header.frame_id = image.header.frame_id
        self.nb_of_object={}
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

            if box.bbox_class in self.nb_of_object:
                self.nb_of_object[box.bbox_class]+=1
            else :
                self.nb_of_object[box.bbox_class]=1


            # Display of the detected bounding boxes.
            if self.show_flag:
                if box.bbox_class not in self._class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    self._class_to_color[box.bbox_class] = (r, g, b)
                color = self._class_to_color[box.bbox_class]


                min_pt = (int(box.xmin),int(box.ymin))
                max_pt =  (int(box.xmax),int(box.ymax))

                cv2.rectangle(cv_image, min_pt, max_pt, color, 2)
                label = "{} ({}) ({:.3f})".format(box.bbox_class, str(box.ID), box.probability)
                pos = (min_pt[0] + 5, min_pt[1] + 25)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_image, label, pos, font,
                            1, color, 1, cv2.LINE_AA)
        
        if self.show_flag:
            self.cv2_pub.publish((self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                               encoding=image.encoding)))
            
        
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
        self.i_service+=1


        if len(boxes.boxes)>0:
            str_to_print = "Yolov8 detected"
            for obj in self.nb_of_object:
                str_to_print += (" " + str(self.nb_of_object[obj]) + " " + obj + "(s) "  + "and")

            str_to_print = str_to_print.rstrip(" and")
            str_to_print+="."
            rospy.loginfo(str_to_print)
        else:
            rospy.loginfo("Yolov8 didn't detect anything.")

        return Yolov8Response(boxes)


    def yolov8_node(self):

        rospy.init_node('yolov8_node')
        
        ## Parameters ##

        rospy.set_param("tracker", "bytetrack.yaml")
        tracker = rospy.get_param("tracker")
    
        rospy.set_param("device", "cuda:0")
        self.device = rospy.get_param("device")
        
        rospy.set_param("threshold", 0.5)
        self.threshold = rospy.get_param("threshold")

        rospy.set_param("show", True) # If true, send the results as a sensor_msgs/Image on topic "/yolov8_image_with_bboxes"
        self.show_flag = rospy.get_param("show")

        rospy.set_param("print_service_results", True)
        self.print_service_result = rospy.get_param("print_service_results")
    
    
        rospy.loginfo("[Yolo_ros] Creating tracker.\n")
        self.tracker = self.create_tracker(tracker)
            

        # Counter
        self.i_cb_pose = 1
        self.i_cb_seg = 1
        self.i_cb_basic = 1
        self.i_service=1

        # Last kinect image received. 
        self.last_image = None


        ## Yolov8 ##

        rospy.loginfo("[Yolo_ros] Loading models.\n")

        # Avaiable models
        self.yolo_pose = YOLO("yolov8m-pose.pt")
        self.yolo_seg = YOLO("yolov8m-seg.pt")
        self.yolo_basic = YOLO("yolov8m.pt")

        # Subscriber to apply yolov8-pose on a sensor_msgs/Image image. Publishs the results on "yolov8_result_pose" topic
        rospy.Subscriber("/kinect2/hd/image_color", Image, self.image_cb, queue_size=1)
        self.result_pub = rospy.Publisher("yolov8_result_pose", Boxes, queue_size=1)

        # Subscriber to apply yolov8-seg on a sensor_msgs/Image image. Publishs the results on "yolov8_result_seg" topic
        rospy.Subscriber("/kinect2/hd/image_color", Image, self.image_cb_1, queue_size=1)
        self.result_pub_1 = rospy.Publisher("yolov8_result_seg", Boxes, queue_size=1)

        # Subscriber to apply yolov8 on a sensor_msgs/Image image. Publishs the results on "yolov8_result_basic" topic
        rospy.Subscriber("/kinect2/hd/image_color", Image, self.image_cb_2, queue_size=1)
        self.result_pub_2 = rospy.Publisher("yolov8_result_basic", Boxes, queue_size=1)

        # Service to call to apply a yolov8 on a sensor_msgs/Image image with a specific model. 
        self.yolov8_srv = rospy.Service('yolov8_on_unique_frame', Yolov8, self.yolov8_on_unique_frame_cb)

        # Dictionnary to count the number of objects detected
        self.nb_of_object={}
        

        ## Display ##

        # Bridge used to transform sensor_msgs_Image to cv2 and vice versa
        self.cv_bridge = CvBridge()

        # Dictionnary for cv2 display
        self._class_to_color={}

        # cv2 publisher for the display
        self.cv2_pub = rospy.Publisher("yolov8_image_with_bboxes", Image, queue_size=10)


        rospy.loginfo("[Yolo_ros] Ready to be used.\n")

        rospy.spin()
    
    
if __name__ == '__main__':
    
    node = Yolo_ros()
    
    




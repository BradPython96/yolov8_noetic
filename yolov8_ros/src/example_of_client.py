import rospy

from sensor_msgs.msg import Image
from yolov8_ros.srv import Yolov8, Yolov8Request

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




    def send_request(self):
        rospy.wait_for_service('yolov8_on_unique_frame')
        try :
            rospy.loginfo("Waiting for yolov8 service")
            bbox_list = self.client_(self.req)
            rospy.loginfo(bbox_list.boxes) 
        except:
            rospy.logwarn("Service call didn't work")
        

if __name__=="__main__":
    a=Yolo_service_tester()
    a.send_request()




            

     
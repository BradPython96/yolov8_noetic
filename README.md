# yolov8_noetic

## **To call the service :**

  **Request** **:**
  
    - string model_name  # the yolovo .pt model you want to use (list of model names below)
    - uint8[] classes  # a list of int to select the classes you want yolov8 to detect (int to string table bellow)
    - sensor_msgs/Image image  # the image treated by yolov8
    

  _List of avaiable models :_

    - "yolov8m.pt" # Basic detection, all classes if no filter
    - "yolov8m-seg.pt" # Detect and create a list of segmentations, all classes if no filter
    - "yolov8m-pose.pt" # Detect and create a list ok skeletons, detect only persons


  _Int to String classes table :_
  
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire   hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'

  **Response :**

  The service returns an object of type "Boxes" :
  
  Boxes :
  
    - header
    - Box[] boxes
   
  With Box type :
  
      - uint16 ID # box ID
      - string bbox_class # name of the box class
      - float64 probability
      - float64 xmin
      - float64 ymin
      - float64 xmax
      - float64 ymax
      - SkeletonPoint[] skeleton # list of points in the skeleton created by yolov8 for this box
        - float64 x
        - float64 y
        - float64 conf # confidence calculated by yolov8 for this point
      - geometry_msgs/Point[] points_in_seg # segmentation pixel list (pixel.z is always =0) created by yolov8 for this box

## **Get results via topics :**

  _Topics to listen to :_
  
    - yolov8_result_pose # correspond to model yolov8m-pose.pt, each box has an ID.
    - yolov8_result_seg # correspond to model yolov8m-seg.pt
    - yolov8_result_basic # correspond to model yolov8m.pt

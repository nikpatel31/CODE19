import numpy as np
import os
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util
from centroidtracker import CentroidTracker

# Define the video stream
cap = cv2.VideoCapture("D:/mini-project1/Tensorflow/workspace/drone_detect/images/valid/0000135_01638_d_0000152.jpg")  # Change only if you have more than one webcams
#cap = cv2.VideoCapture("D:/mini-project1/Tensorflow/workspace/drone_detect/drone_main.mp4")
#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
#out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))



# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'trained_graph/output_inference_graph_v1.pb/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('annotations', 'label.pbtxt')

# Number of classes to detect
NUM_CLASSES = 12


ct = CentroidTracker()
(H, W) = (None, None)


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"mosse": cv2.TrackerMOSSE_create
}

trackers = cv2.MultiTracker_create()

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            ret, image_np = cap.read()
            (success, box1) = trackers.update(image_np)
            if image_np is None :
                break
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            
            
            im_height, im_width, img_channel = image_np.shape
            
            THRESHOLD = 0.50                           # adjust your threshold here
            N = len(boxes)
            ind=np.where(scores>THRESHOLD)
            box_=boxes[ind]
            N_=box_.shape[0]
            labels=[]
            New_Box=[]
            rects=[]
            for i in range(N_):
                if len(box1) != 0:
                    break
                New_Box.append(boxes[0,i,0:2])
                ymin = boxes[0,i,0]
                xmin = boxes[0,i,1]
                ymax = boxes[0,i,2]
                xmax = boxes[0,i,3]
                
            
                (xminn, xmaxx, yminn, ymaxx) = (int(xmin * im_width),int( xmax * im_width), int(ymin * im_height), int(ymax * im_height))
                box=(xminn, yminn, xmaxx, ymaxx)
                rects.append(box)
               
                (startX, startY, endX, endY) = box
                cv2.rectangle(image_np, (startX, startY), (endX, endY),(0, 255, 0), 2)
                
            objects = ct.update(rects)
            for (objectID, centroid) in objects.items():
                if len(box1) != 0:
                    break
		      
                text = "ID {}".format(objectID)
		      
                cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		      
                cv2.circle(image_np, (centroid[0], centroid[1]), 3, (0, 255, 0), -1)
                
            for box11 in box1:
                (x, y, w, h) = [int(v) for v in box11]
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    

            
               
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            # Display output
            #out.write(image_np)

            heatmap=cv2.applyColorMap(image_np,cv2.COLORMAP_HOT)
            cv2.imshow('TEMPERATURE MAP', cv2.resize(heatmap, (800, 600)))
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            cv2.imwrite("id_ppt_heatV3.jpg",heatmap)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                box1 = cv2.selectROI("object detection", image_np, fromCenter=False,
                                    showCrosshair=True)
                tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
                trackers.add(tracker,image_np, box1)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                #out.release()
                cap.release()
                cv2.destroyAllWindows()
                break


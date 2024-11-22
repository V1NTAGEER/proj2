import cv2
import numpy as np
import time
import os
from collections import defaultdict

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.tracking_threshold = 0.3  # IOU threshold
        self.max_disappeared = 30  # Frames before considering object gone

    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def update(self, boxes, class_ids):
        current_objects = {}
        
        for box, class_id in zip(boxes, class_ids):
            matched = False
            for obj_id, (tracked_box, tracked_class, disappeared) in self.tracked_objects.items():
                if tracked_class == class_id:
                    iou = self.calculate_iou(box, tracked_box)
                    if iou > self.tracking_threshold:
                        current_objects[obj_id] = (box, class_id, 0)
                        matched = True
                        break
            
            if not matched:
                current_objects[self.next_id] = (box, class_id, 0)
                self.next_id += 1

        self.tracked_objects = current_objects
        return current_objects

class ObjectDetector:
    def __init__(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.weights_path = os.path.join(current_dir, "yolov4.weights")
            self.config_path = os.path.join(current_dir, "yolov4.cfg")
            self.labels_path = os.path.join(current_dir, "coco.names")
            
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"YOLOv4 weights file not found at: {self.weights_path}")
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"YOLOv4 config file not found at: {self.config_path}")
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(f"COCO labels file not found at: {self.labels_path}")

            print("Loading label files...")
            self.labels = open(self.labels_path).read().strip().split("\n")
            np.random.seed(42)
            self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
            
            self.confidence_threshold = 0.3
            self.nms_threshold = 0.4
            self.max_detections = 100
            
            print("Loading YOLO model...")
            self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
            
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA backend")
            except:
                print("CUDA not available, using CPU")
            
            self.ln = self.net.getLayerNames()
            self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            self.counts = defaultdict(int)
            self.tracker = ObjectTracker()
            self.unique_objects = set()
            print("Initialization complete!")

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def process_video(self, video_path=0):
        try:
            print("Opening video capture...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Error: Could not open video capture")
            
            cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Object Detection", 1280, 720)  # you can adjust the size to your preference
            print("Starting video capture. Press 'q' to quit.")
            frame_count = 0
            fps = 0
            start_time = time.time()
            while True: 
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                frame_count += 1
                if frame_count % 3 != 0:
                    continue

                (H, W) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True, crop=False)
                self.net.setInput(blob)
                layerOutputs = self.net.forward(self.ln)
                boxes = []
                confidences = []
                classIDs = []
                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if confidence > self.confidence_threshold:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - start_time)
                    start_time = time.time()
                if len(idxs) > 0:
                    filtered_boxes = [boxes[i] for i in idxs.flatten()]
                    filtered_classIDs = [classIDs[i] for i in idxs.flatten()]
                    filtered_confidences = [confidences[i] for i in idxs.flatten()]           
                    if len(filtered_boxes) > self.max_detections:
                        sorted_indices = np.argsort(filtered_confidences)[::-1][:self.max_detections]
                        filtered_boxes = [filtered_boxes[i] for i in sorted_indices]
                        filtered_classIDs = [filtered_classIDs[i] for i in sorted_indices]
                    tracked_objects = self.tracker.update(filtered_boxes, filtered_classIDs)
                    for obj_id, (box, class_id, _) in tracked_objects.items():
                        x, y, w, h = box
                        color = [int(c) for c in self.colors[class_id]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                        text = f"{self.labels[class_id]} #{obj_id}"
                        cv2.putText(frame, text, (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        if obj_id not in self.unique_objects:
                            self.unique_objects.add(obj_id)
                            self.counts[self.labels[class_id]] += 1
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset = 70
                for class_name, count in self.counts.items():
                    cv2.putText(frame, f"{class_name}: {count}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 30
                cv2.imshow("Object Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except Exception as e: 
            print(f"Error during video processing: {str(e)}")
        finally:
            print("Cleaning up...")
            cap.release()
            cv2.destroyAllWindows()
def main():
    try:
        print("Starting program...")
        detector = ObjectDetector()
        print("Starting video capture...")
        detector.process_video(0)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        print("Program finished. Press Enter to exit...")
        input()
if __name__ == "__main__":
    main()
    
#Press 'q' to quit the program
# press 'f' to toggle FullScreen
#Press 'f' to toggle fullscreen

#The numbers on screen show unique counts per object type

#The confidence threshold is set to 0.3 for more detections
#Required Files (must be in same folder as Python script):
#yolov4.weights - https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
#yolov4.cfg - https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg  
#coco.names - https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

#Install required packages in command prompt:
#pip install opencv-python numpy

#File Structure:
#Your-Folder/
#    ├── Object detection.py
#    ├── yolov4.weights
#    ├── yolov4.cfg
#    └── coco.names

#How to Run:
#1. Open command prompt
#2. Navigate to program folder
#3. Run: python "Object detection.py"

#Controls:
#Press 'q' to quit
#Press 'f' for fullscreen

#Display Info:
#Top left: FPS
#Below FPS: Object counts
#Boxes: Detected items
#Text above boxes: Object type and ID

#Performance Notes:
#Processes every 3rd frame
#Confidence threshold: 0.3
#Max detections: 100
#Uses CUDA if available (NVIDIA GPU)

#Common Issues:
#"File not found": Check YOLO files present
#Camera not working: Try changing video source (0 or 1)
#Slow performance: Using CPU instead of GPU
#High RAM usage: Normal for AI model

#Adjustable Parameters in code:
#self.confidence_threshold = 0.3  #Higher = fewer false detections
#self.nms_threshold = 0.4        #Box overlap detection
#self.max_detections = 100       #Max objects to track

#Tips:
#Good lighting helps
#Keep camera stable
#Clear background better
#Objects need to be visible
#Wait 2-3 seconds for startup

#System Requirements:
#Minimum: 4GB RAM, dual-core
#Recommended: 8GB RAM, quad-core
#Optional: NVIDIA GPU

#Troubleshooting:
#"No module named 'cv2'" → pip install opencv-python
#"No module named 'numpy'" → pip install numpy
#"Unable to open camera" → Check webcam
#"File not found" → Check YOLO files location

#Best Performance:
#Close other programs
#Use good lighting
#Position camera steady
#Use GPU if possible

#Safety:
#No data saved
#Local processing only
#Temporary counting only

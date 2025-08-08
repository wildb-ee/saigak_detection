import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

class YOLOCameraDetection:
    def __init__(self, model_path, confidence_threshold=0.5, device='auto'):
        """
        Initialize YOLO camera detection
        
        Args:
            model_path (str): Path to your trained YOLOv8n model (.pt file)
            confidence_threshold (float): Minimum confidence for detections
            device (str): Device to run inference on ('auto', 'cpu', 'cuda', 'mps')
        """
        # Load the trained model
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Get class names
        self.class_names = self.model.names
        
        # Colors for bounding boxes (BGR format for OpenCV)
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0)
        ]
        
    def detect_frame(self, frame):
        """
        Run detection on a single frame
        
        Args:
            frame: Input frame from camera
            
        Returns:
            annotated_frame: Frame with bounding boxes and labels
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, device=self.device)
        
        # Get the first result (single image)
        result = results[0]
        
        # Draw bounding boxes and labels
        annotated_frame = frame.copy()
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Get class name and color
                class_name = self.class_names[class_id]
                color = self.colors[class_id % len(self.colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class name and confidence
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame, 
                    (x1, y1 - label_size[1] - 10), 
                    (x1 + label_size[0], y1), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
        
        return annotated_frame
    
    def run_camera_detection(self, camera_index=0, window_name="YOLO Detection"):
        """
        Run real-time detection on camera feed
        
        Args:
            camera_index (int): Camera index (0 for default camera)
            window_name (str): Name of the display window
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting camera detection. Press 'q' to quit.")
        print("Press 's' to save current frame.")
        
        frame_count = 0
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Run detection
            detection_start = time.time()
            annotated_frame = self.detect_frame(frame)
            detection_time = time.time() - detection_start
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = fps_counter / elapsed_time
                print(f"FPS: {fps:.2f}, Detection time: {detection_time*1000:.1f}ms")
            
            # Add FPS text to frame
            cv2.putText(
                annotated_frame, 
                f"FPS: {1/detection_time:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Display frame
            cv2.imshow(window_name, annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved frame as {filename}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera detection stopped.")

# Usage example
if __name__ == "__main__":
    # Replace with your model path
    MODEL_PATH = "saiga_detection/yolov8_saiga/weights/best.pt"
    
    try:
        # Initialize detector
        detector = YOLOCameraDetection(
            model_path=MODEL_PATH,
            confidence_threshold=0.5,  # Adjust confidence threshold as needed
            device='auto'  # Will automatically choose best available device
        )
        
        # Start camera detection
        detector.run_camera_detection(
            camera_index=0,  # Use 0 for default camera, 1 for external camera, etc.
            window_name="YOLOv8n Live Detection"
        )
        
    except FileNotFoundError:
        print(f"Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH with the correct path to your trained model.")
    except Exception as e:
        print(f"Error: {e}")
        
# Alternative usage for batch processing or custom integration
def process_single_frame_example():
    """Example of processing a single frame"""
    MODEL_PATH = "path/to/your/yolov8n_model.pt"
    
    detector = YOLOCameraDetection(MODEL_PATH)
    
    # Capture single frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Process frame
        result_frame = detector.detect_frame(frame)
        
        # Save or display result
        cv2.imwrite("detection_result.jpg", result_frame)
        cv2.imshow("Detection", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
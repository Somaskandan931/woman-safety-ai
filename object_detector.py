import torch
from ultralytics import YOLO
import numpy as np


class ObjectDetector :
    """
    Object detector using YOLOv8 to detect people and objects in video frames
    """

    def __init__ ( self, model_path="models/yolov8n.pt", confidence_threshold=0.5 ) :
        """
        Initialize the object detector

        Args:
            model_path (str): Path to the YOLO model
            confidence_threshold (float): Minimum confidence threshold for detections
        """
        self.model = YOLO( model_path )
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names
        print( f"🔍 Object detector initialized with {len( self.class_names )} classes" )

    def detect ( self, frame ) :
        """
        Detect objects in a frame

        Args:
            frame (numpy.ndarray): Input frame for detection

        Returns:
            list: List of detection dictionaries with keys:
                  - 'class': Class name
                  - 'confidence': Detection confidence
                  - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
        """
        results = self.model( frame, stream=True, verbose=False )

        detections = []

        # Process detections
        for result in results :
            boxes = result.boxes

            for box in boxes :
                conf = float( box.conf[0] )

                # Filter by confidence threshold
                if conf < self.confidence_threshold :
                    continue

                # Get class ID and name
                cls_id = int( box.cls[0] )
                cls_name = self.class_names[cls_id]

                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append( {
                    'class' : cls_name,
                    'confidence' : conf,
                    'bbox' : [int( x1 ), int( y1 ), int( x2 ), int( y2 )]
                } )

        return detections

    def is_person_in_danger ( self, detections ) :
        """
        Check if detected people might be in danger based on surrounding objects

        Args:
            detections (list): List of detection dictionaries

        Returns:
            tuple: (is_danger, reason) where is_danger is a boolean and reason is a string
        """
        people = [d for d in detections if d['class'] == 'person']
        if not people :
            return False, ""

        # Check for weapons or dangerous objects
        dangerous_objects = ['knife', 'scissors', 'baseball bat', 'bottle']
        objects = [d for d in detections if d['class'] in dangerous_objects]

        if objects :
            # Check proximity of dangerous objects to people
            for person in people :
                p_x1, p_y1, p_x2, p_y2 = person['bbox']
                person_center = ((p_x1 + p_x2) // 2, (p_y1 + p_y2) // 2)

                for obj in objects :
                    o_x1, o_y1, o_x2, o_y2 = obj['bbox']
                    obj_center = ((o_x1 + o_x2) // 2, (o_y1 + o_y2) // 2)

                    # Calculate distance between centers
                    distance = np.sqrt(
                        (person_center[0] - obj_center[0]) ** 2 +
                        (person_center[1] - obj_center[1]) ** 2
                    )

                    # If the object is close to the person, consider it dangerous
                    # The threshold here is a heuristic, adjust as needed
                    threshold = max( p_x2 - p_x1, p_y2 - p_y1 ) * 1.5

                    if distance < threshold :
                        return True, f"Dangerous object ({obj['class']}) near person"

        return False, ""


if __name__ == "__main__" :
    # Simple test with an image
    import cv2

    detector = ObjectDetector()
    img = cv2.imread( "test_image.jpg" )
    if img is not None :
        detections = detector.detect( img )
        print( f"Found {len( detections )} objects:" )
        for det in detections :
            print( f"  - {det['class']} ({det['confidence']:.2f})" )
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle( img, (x1, y1), (x2, y2), (0, 255, 0), 2 )
            cv2.putText( img, det['class'], (x1, y1 - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )

        cv2.imshow( "Test Detection", img )
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
    else :
        print( "Could not load test image" )
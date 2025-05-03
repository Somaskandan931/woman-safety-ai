import cv2
import time
import threading
import os
from dotenv import load_dotenv
from object_detector import ObjectDetector
from pose_detector import PoseDetector
from voice_detector import VoiceDetector
from alert import AlertSystem
from utils.helpers import draw_bbox, draw_pose, read_config

# Load environment variables
load_dotenv()

# Read configuration
config = read_config( "config/config.yaml" )


class SafetySystem :
    def __init__ ( self ) :
        # Initialize detectors
        self.object_detector = ObjectDetector(
            model_path=config['models']['yolo_path'],
            confidence_threshold=config['thresholds']['object_detection']
        )

        self.pose_detector = PoseDetector(
            min_detection_confidence=config['thresholds']['pose_detection'],
            min_tracking_confidence=config['thresholds']['pose_tracking']
        )

        self.voice_detector = VoiceDetector(
            keywords=config['keywords']['emergency'],
            confidence_threshold=config['thresholds']['voice_detection']
        )

        # Initialize alert system
        self.alert_system = AlertSystem()

        # Camera setup
        self.camera_id = config['camera']['id']
        self.cap = cv2.VideoCapture( self.camera_id )

        # State variables
        self.running = False
        self.alert_cooldown = config['alert']['cooldown_seconds']
        self.last_alert_time = 0

        # Create output directory if it doesn't exist
        os.makedirs( "data/snapshots", exist_ok=True )

    def start_voice_detection ( self ) :
        """Start voice detection in a separate thread"""
        self.voice_detector.start_listening( callback=self.handle_voice_alert )

    def handle_voice_alert ( self, keyword ) :
        """Callback for when voice detector detects a keyword"""
        current_time = time.time()

        # Check if we're in cooldown period
        if current_time - self.last_alert_time < self.alert_cooldown :
            return

        print( f"⚠️ EMERGENCY VOICE KEYWORD DETECTED: {keyword}" )

        # Capture current frame for evidence
        ret, frame = self.cap.read()
        if ret :
            # Save snapshot
            timestamp = time.strftime( "%Y%m%d-%H%M%S" )
            snapshot_path = f"data/snapshots/voice_alert_{timestamp}.jpg"
            cv2.imwrite( snapshot_path, frame )

            # Send alert
            message = f"EMERGENCY: Voice keyword '{keyword}' detected!"
            self.alert_system.send_alert( message, snapshot_path )
            self.last_alert_time = current_time

    def handle_visual_alert ( self, frame, alert_type, details=None ) :
        """Handle visual alerts from pose or object detection"""
        current_time = time.time()

        # Check if we're in cooldown period
        if current_time - self.last_alert_time < self.alert_cooldown :
            return

        print( f"⚠️ EMERGENCY VISUAL SITUATION DETECTED: {alert_type}" )

        # Save snapshot
        timestamp = time.strftime( "%Y%m%d-%H%M%S" )
        snapshot_path = f"data/snapshots/visual_alert_{timestamp}.jpg"
        cv2.imwrite( snapshot_path, frame )

        # Send alert
        message = f"EMERGENCY: {alert_type} detected!"
        if details :
            message += f" Details: {details}"

        self.alert_system.send_alert( message, snapshot_path )
        self.last_alert_time = current_time

    def process_frame ( self, frame ) :
        """Process a single frame with all detectors"""
        # 1. Object detection
        detections = self.object_detector.detect( frame )

        # Draw bounding boxes and get person detections
        frame_with_boxes = draw_bbox( frame, detections )
        person_detections = [d for d in detections if d['class'] == 'person']

        # 2. Pose detection for each person
        for person in person_detections :
            x1, y1, x2, y2 = person['bbox']
            person_crop = frame[y1 :y2, x1 :x2]

            if person_crop.size == 0 :  # Skip empty crops
                continue

            pose_results = self.pose_detector.detect_pose( person_crop )

            # Check for suspicious pose
            if pose_results :
                frame = draw_pose( frame, pose_results, (x1, y1) )

                # Check if the pose indicates danger
                is_dangerous, alert_type = self.pose_detector.is_dangerous_pose( pose_results )

                if is_dangerous :
                    self.handle_visual_alert( frame, alert_type )

        return frame_with_boxes

    def run ( self ) :
        """Main loop to process video feed"""
        self.running = True

        # Start voice detection in a separate thread
        voice_thread = threading.Thread( target=self.start_voice_detection )
        voice_thread.daemon = True
        voice_thread.start()

        print( "🔍 AI Safety System is now running..." )
        print( "Press 'q' to quit" )

        while self.running :
            ret, frame = self.cap.read()

            if not ret :
                print( "❌ Failed to read from camera. Exiting..." )
                break

            # Process the frame
            processed_frame = self.process_frame( frame )

            # Display the frame
            cv2.imshow( 'AI Safety System', processed_frame )

            # Break loop on 'q' key
            if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) :
                self.running = False

        # Clean up
        self.voice_detector.stop_listening()
        self.cap.release()
        cv2.destroyAllWindows()
        print( "✅ AI Safety System has been stopped" )


if __name__ == "__main__" :
    safety_system = SafetySystem()
    try :
        safety_system.run()
    except KeyboardInterrupt :
        print( "System interrupted by user" )
    finally :
        if safety_system.cap.isOpened() :
            safety_system.cap.release()
        cv2.destroyAllWindows()
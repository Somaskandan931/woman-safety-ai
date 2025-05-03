import mediapipe as mp
import numpy as np
import cv2


class PoseDetector :
    """
    Pose detector using MediaPipe to detect human poses and identify dangerous situations
    """

    def __init__ ( self, min_detection_confidence=0.5, min_tracking_confidence=0.5 ) :
        """
        Initialize the pose detector

        Args:
            min_detection_confidence (float): Minimum confidence for pose detection
            min_tracking_confidence (float): Minimum confidence for pose tracking
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        print( "🧍 Pose detector initialized" )

    def detect_pose ( self, frame ) :
        """
        Detect human pose in a frame

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            dict: MediaPipe pose landmarks and world landmarks
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )

        # Process the frame and get pose landmarks
        results = self.pose.process( rgb_frame )

        if results.pose_landmarks :
            return {
                'landmarks' : results.pose_landmarks,
                'world_landmarks' : results.pose_world_landmarks
            }

        return None

    def is_dangerous_pose ( self, pose_results ) :
        """
        Check if the detected pose indicates a dangerous situation

        Args:
            pose_results (dict): MediaPipe pose landmarks

        Returns:
            tuple: (is_dangerous, alert_type) where is_dangerous is a boolean and
                  alert_type is a string describing the danger
        """
        if not pose_results or not pose_results['landmarks'] :
            return False, ""

        landmarks = pose_results['landmarks'].landmark

        # Extract key points
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Check for falling (head below hips)
        # In a normal standing pose, nose.y < hip.y (since y increases downward in image space)
        is_fallen = (nose.y > left_hip.y and nose.y > right_hip.y)
        if is_fallen :
            return True, "Fall Detected"

        # Check for hands raised (wrist above shoulder)
        # In a normal pose, wrist.y > shoulder.y
        hands_raised = (left_wrist.y < left_shoulder.y - 0.1 or
                        right_wrist.y < right_shoulder.y - 0.1)
        if hands_raised :
            return True, "Raised Hands Detected"

        # Check for crouching or lying down
        # Calculate vertical distance from shoulders to ankles
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        body_height = abs( ankle_y - shoulder_y )

        # If body height is small relative to the image, person might be crouching or lying
        if body_height < 0.2 :  # Threshold may need tuning
            return True, "Unusual Posture Detected"

        # Check for extreme body tilt
        shoulder_midpoint = [(left_shoulder.x + right_shoulder.x) / 2,
                             (left_shoulder.y + right_shoulder.y) / 2]
        hip_midpoint = [(left_hip.x + right_hip.x) / 2,
                        (left_hip.y + right_hip.y) / 2]

        # Calculate tilt angle of the spine
        dx = hip_midpoint[0] - shoulder_midpoint[0]
        dy = hip_midpoint[1] - shoulder_midpoint[1]
        spine_angle = abs( np.degrees( np.arctan2( dx, dy ) ) )

        # If spine is too tilted (not vertical)
        if spine_angle > 30 :  # Threshold may need tuning
            return True, "Unusual Body Position"

        return False, ""

    def draw_pose ( self, frame, pose_results ) :
        """
        Draw pose landmarks on a frame

        Args:
            frame (numpy.ndarray): Input frame
            pose_results (dict): MediaPipe pose landmarks

        Returns:
            numpy.ndarray: Frame with pose landmarks drawn
        """
        if not pose_results or not pose_results['landmarks'] :
            return frame

        # Draw the pose landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            pose_results['landmarks'],
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec( color=(0, 255, 0), thickness=2, circle_radius=2 ),
            self.mp_drawing.DrawingSpec( color=(0, 0, 255), thickness=2, circle_radius=2 )
        )

        return frame


if __name__ == "__main__" :
    # Simple test with a video file or webcam
    import cv2

    detector = PoseDetector()
    cap = cv2.VideoCapture( 0 )  # 0 for webcam, or provide a video file path

    while cap.isOpened() :
        ret, frame = cap.read()
        if not ret :
            break

        pose_results = detector.detect_pose( frame )
        if pose_results :
            frame = detector.draw_pose( frame, pose_results )
            is_dangerous, alert_type = detector.is_dangerous_pose( pose_results )
            if is_dangerous :
                cv2.putText( frame, f"ALERT: {alert_type}", (50, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )

        cv2.imshow( "Pose Detection", frame )
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) :
            break

    cap.release()
    cv2.destroyAllWindows()
import cv2
import yaml
import numpy as np
import os


def draw_bbox ( frame, detections ) :
    """
    Draw bounding boxes on a frame

    Args:
        frame (numpy.ndarray): Input frame
        detections (list): List of detection dictionaries

    Returns:
        numpy.ndarray: Frame with bounding boxes drawn
    """
    result = frame.copy()

    # Color mapping for different classes
    color_map = {
        'person' : (0, 255, 0),  # Green
        'knife' : (0, 0, 255),  # Red
        'scissors' : (0, 0, 255),
        'baseball bat' : (0, 0, 255),
        'bottle' : (255, 0, 0),  # Blue
        'default' : (255, 255, 0)  # Yellow
    }

    for detection in detections :
        # Get bbox coordinates
        x1, y1, x2, y2 = detection['bbox']

        # Get class name and confidence
        class_name = detection['class']
        confidence = detection['confidence']

        # Get color for this class
        color = color_map.get( class_name, color_map['default'] )

        # Draw bounding box
        cv2.rectangle( result, (x1, y1), (x2, y2), color, 2 )

        # Draw label background
        text = f"{class_name} {confidence:.2f}"
        text_size, _ = cv2.getTextSize( text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2 )
        cv2.rectangle( result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1 )

        # Draw text
        cv2.putText( result, text, (x1, y1 - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2 )

    return result


def draw_pose ( frame, pose_results, offset=(0, 0) ) :
    """
    Draw pose landmarks on a frame

    Args:
        frame (numpy.ndarray): Input frame
        pose_results (dict): MediaPipe pose landmarks
        offset (tuple): Offset to apply to landmarks (x, y)

    Returns:
        numpy.ndarray: Frame with pose landmarks drawn
    """
    import mediapipe as mp

    if not pose_results or not pose_results.get( 'landmarks' ) :
        return frame

    result = frame.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # If there's an offset, we need to adjust the landmarks
    if offset != (0, 0) :
        # Create a copy of the landmarks
        landmarks = mp.solutions.pose.PoseLandmark()
        landmarks.CopyFrom( pose_results['landmarks'] )

        # Adjust the landmarks
        for i in range( len( landmarks.landmark ) ) :
            landmarks.landmark[i].x += offset[0]
            landmarks.landmark[i].y += offset[1]

        # Draw adjusted landmarks
        mp_drawing.draw_landmarks(
            result,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec( color=(0, 255, 0), thickness=2, circle_radius=2 ),
            mp_drawing.DrawingSpec( color=(0, 0, 255), thickness=2, circle_radius=2 )
        )
    else :
        # Draw original landmarks
        mp_drawing.draw_landmarks(
            result,
            pose_results['landmarks'],
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec( color=(0, 255, 0), thickness=2, circle_radius=2 ),
            mp_drawing.DrawingSpec( color=(0, 0, 255), thickness=2, circle_radius=2 )
        )

    return result


def read_config ( config_path ) :
    """
    Read configuration from a YAML file

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: Configuration dictionary
    """
    # Default config in case file doesn't exist
    default_config = {
        'camera' : {
            'id' : 0  # Default to first webcam
        },
        'models' : {
            'yolo_path' : 'models/yolov8n.pt'
        },
        'thresholds' : {
            'object_detection' : 0.5,
            'pose_detection' : 0.5,
            'pose_tracking' : 0.5,
            'voice_detection' : 0.7
        },
        'keywords' : {
            'emergency' : ['help', 'save me', 'stop', 'emergency', 'please help']
        },
        'alert' : {
            'cooldown_seconds' : 60  # Wait 60 seconds between alerts
        }
    }

    # Create config directory if it doesn't exist
    os.makedirs( os.path.dirname( config_path ), exist_ok=True )

    try :
        if os.path.exists( config_path ) :
            with open( config_path, 'r' ) as f :
                config = yaml.safe_load( f )
                print( f"Configuration loaded from {config_path}" )
        else :
            # Create default config file
            with open( config_path, 'w' ) as f :
                yaml.dump( default_config, f, default_flow_style=False )
                print( f"Default configuration created at {config_path}" )
            config = default_config
    except Exception as e :
        print( f"Error loading configuration: {e}" )
        print( "Using default configuration" )
        config = default_config

    return config


def calculate_fps ( prev_frame_time ) :
    """
    Calculate frames per second

    Args:
        prev_frame_time (float): Time of previous frame

    Returns:
        tuple: (current_frame_time, fps)
    """
    import time

    # Get current time
    current_time = time.time()

    # Calculate FPS
    fps = 1 / (current_time - prev_frame_time)

    return current_time, fps


def overlay_text ( frame, text, position, color=(0, 255, 0) ) :
    """
    Overlay text on a frame with a semi-transparent background

    Args:
        frame (numpy.ndarray): Input frame
        text (str): Text to overlay
        position (tuple): Position (x, y)
        color (tuple): Text color

    Returns:
        numpy.ndarray: Frame with text overlay
    """
    result = frame.copy()

    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize( text, font, font_scale, thickness )

    # Calculate background rectangle coordinates
    x, y = position
    bg_rect = (x, y - text_size[1] - 5, x + text_size[0], y + 5)

    # Create overlay for semi-transparent background
    overlay = result.copy()
    cv2.rectangle( overlay, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]),
                   (0, 0, 0), -1 )

    # Apply overlay with transparency
    alpha = 0.6
    cv2.addWeighted( overlay, alpha, result, 1 - alpha, 0, result )

    # Draw text
    cv2.putText( result, text, (x, y), font, font_scale, color, thickness )

    return result
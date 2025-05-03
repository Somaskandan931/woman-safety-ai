import streamlit as st
import cv2
import numpy as np
import os
import json
import pandas as pd
import time
from datetime import datetime
import threading
import matplotlib.pyplot as plt
from PIL import Image

from object_detector import ObjectDetector
from pose_detector import PoseDetector
from utils.helpers import draw_bbox, draw_pose, read_config

# Read configuration
config = read_config("config/config.yaml")

# Global variables for video capture
video_capture = None
run_video = False
video_thread = None


class Dashboard:
    def __init__(self):
        self.object_detector = None
        self.pose_detector = None
        self.alert_history = []
        self.history_file = "data/alert_history.json"

        # Load alert history
        self._load_history()

    def _load_history(self):
        """Load alert history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.alert_history = json.load(f)
        except Exception as e:
            st.error(f"Error loading alert history: {e}")
            self.alert_history = []

    def initialize_models(self):
        """Initialize detection models if not already initialized"""
        if self.object_detector is None:
            with st.spinner("Loading object detection model..."):
                self.object_detector = ObjectDetector(
                    model_path=config['models']['yolo_path'],
                    confidence_threshold=config['thresholds']['object_detection']
                )

        if self.pose_detector is None:
            with st.spinner("Loading pose detection model..."):
                self.pose_detector = PoseDetector(
                    min_detection_confidence=config['thresholds']['pose_detection'],
                    min_tracking_confidence=config['thresholds']['pose_tracking']
                )

    def run_video_feed(self, camera_id):
        """Process video feed in a separate thread"""
        global video_capture, run_video

        video_capture = cv2.VideoCapture(camera_id)
        run_video = True

        while run_video:
            ret, frame = video_capture.read()

            if not ret:
                st.error("Failed to capture video. Check camera connection.")
                break

            # Process the frame if models are initialized
            if self.object_detector is not None and self.pose_detector is not None:
                # Object detection
                detections = self.object_detector.detect(frame)
                frame_with_boxes = draw_bbox(frame, detections)

                # Pose detection for each person
                person_detections = [d for d in detections if d['class'] == 'person']
                for person in person_detections:
                    x1, y1, x2, y2 = person['bbox']
                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size == 0:  # Skip empty crops
                        continue

                    pose_results = self.pose_detector.detect_pose(person_crop)

                    if pose_results:
                        frame_with_boxes = draw_pose(frame_with_boxes, pose_results, (x1, y1))

                        # Check for suspicious pose
                        is_dangerous, alert_type = self.pose_detector.is_dangerous_pose(pose_results)

                        if is_dangerous:
                            # Add text overlay with alert
                            cv2.putText(frame_with_boxes,
                                      f"ALERT: {alert_type}",
                                      (50, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1,
                                      (0, 0, 255),
                                      2)
            else:
                frame_with_boxes = frame

            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)

            # Update the Streamlit image
            if 'video_placeholder' in st.session_state:
                st.session_state.current_frame = frame_rgb

            time.sleep(0.03)  # ~30 FPS

    def display_home(self):
        """Display the home/dashboard page"""
        st.title("🛡️ AI-Powered Woman Safety System")
        st.subheader("Real-time Monitoring Dashboard")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Video feed display
            st.subheader("📹 Live Camera Feed")

            # Camera selection
            camera_options = ["Default Camera (0)"]
            camera_options.extend([f"Camera {i}" for i in range(1, 5)])
            selected_camera = st.selectbox("Select Camera", camera_options)
            camera_id = int(selected_camera.split("(")[1].split(")")[0]) if "(" in selected_camera else 0

            # Initialize video placeholder
            if 'video_placeholder' not in st.session_state:
                st.session_state.video_placeholder = st.empty()
                st.session_state.current_frame = None

            # Start/Stop video
            col_start, col_stop = st.columns(2)

            with col_start:
                if st.button("Start Camera", type="primary"):
                    self.initialize_models()  # Make sure models are loaded

                    # Stop any existing thread
                    global run_video, video_thread
                    run_video = False
                    if video_thread is not None and video_thread.is_alive():
                        video_thread.join()

                    # Start new thread
                    video_thread = threading.Thread(target=self.run_video_feed, args=(camera_id,))
                    video_thread.daemon = True
                    video_thread.start()

                    st.success("Camera started")

            with col_stop:
                if st.button("Stop Camera", type="secondary"):
                    run_video = False
                    if video_capture is not None:
                        video_capture.release()
                    st.session_state.current_frame = None
                    st.success("Camera stopped")

            # Display the video feed
            if st.session_state.current_frame is not None:
                st.session_state.video_placeholder.image(
                    st.session_state.current_frame,
                    caption="Live Feed",
                    use_column_width=True
                )
            else:
                # Display placeholder image
                st.session_state.video_placeholder.info("Camera not active. Press 'Start Camera' to begin.")

        with col2:
            # System status and stats
            st.subheader("📊 System Status")

            status_placeholder = st.empty()

            # Refresh system status periodically
            current_time = datetime.now().strftime("%H:%M:%S")

            # Check if models are loaded
            models_loaded = self.object_detector is not None and self.pose_detector is not None
            camera_active = run_video and video_capture is not None and video_capture.isOpened()

            status_df = pd.DataFrame({
                "Component": ["Camera", "Object Detection", "Pose Detection", "Last Update"],
                "Status": [
                    "✅ Active" if camera_active else "❌ Inactive",
                    "✅ Loaded" if models_loaded else "❌ Not Loaded",
                    "✅ Loaded" if models_loaded else "❌ Not Loaded",
                    current_time
                ]
            })

            status_placeholder.dataframe(status_df, hide_index=True, use_container_width=True)

            # Recent alerts section
            st.subheader("🚨 Recent Alerts")

            if not self.alert_history:
                st.info("No recent alerts")
            else:
                # Show most recent 5 alerts
                recent_alerts = self.alert_history[-5:]

                for alert in reversed(recent_alerts):
                    with st.expander(f"{alert['timestamp']} - {alert['message'][:30]}..."):
                        st.write(f"**Message:** {alert['message']}")
                        st.write(f"**Time:** {alert['timestamp']}")

                        if alert.get('location_url'):
                            st.write(f"**Location:** [View on Map]({alert['location_url']})")

                        if alert.get('image_path') and os.path.exists(alert['image_path']):
                            try:
                                img = Image.open(alert['image_path'])
                                st.image(img, caption="Alert Image", use_column_width=True)
                            except Exception as e:
                                st.error(f"Could not load image: {e}")

    def display_analytics(self):
        """Display analytics and historical data"""
        st.title("📈 Safety Analytics")

        # Load alert history
        self._load_history()

        if not self.alert_history:
            st.info("No alert data available for analysis")
            return

        # Convert alert history to DataFrame
        df = pd.DataFrame(self.alert_history)

        # Extract date from timestamp
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        # Alerts over time
        st.subheader("Alerts Over Time")

        # Group by date and count
        alerts_by_date = df.groupby('date').size().reset_index(name='count')

        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(alerts_by_date['date'], alerts_by_date['count'], color='coral')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Alerts')
        ax.set_title('Alerts by Date')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Rotate date labels if needed
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        # Alert types
        st.subheader("Alert Types")

        # Extract alert types from messages
        def extract_alert_type(message):
            if "Fall" in message:
                return "Fall Detected"
            elif "Raised Hand" in message:
                return "Raised Hands"
            elif "Voice" in message:
                return "Voice Alert"
            elif "Unusual" in message:
                return "Unusual Posture"
            else:
                return "Other"

        df['alert_type'] = df['message'].apply(extract_alert_type)

        # Group by alert type and count
        alert_types = df['alert_type'].value_counts().reset_index()
        alert_types.columns = ['Alert Type', 'Count']

        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(alert_types['Count'], labels=alert_types['Alert Type'], autopct='%1.1f%%',
              shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        st.pyplot(fig)

        # Display raw data
        st.subheader("Alert Log")

        # Create a more readable DataFrame for display
        display_df = df[['timestamp', 'message']].copy()
        display_df.columns = ['Timestamp', 'Alert Message']

        st.dataframe(display_df, use_container_width=True)

    def display_settings(self):
        """Display and update system settings"""
        st.title("⚙️ System Settings")

        # Load current config
        current_config = read_config("config/config.yaml")

        with st.form("settings_form"):
            st.subheader("Detection Settings")

            # Object detection threshold
            object_threshold = st.slider(
                "Object Detection Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=float(current_config['thresholds']['object_detection']),
                step=0.05
            )

            # Pose detection threshold
            pose_threshold = st.slider(
                "Pose Detection Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=float(current_config['thresholds']['pose_detection']),
                step=0.05
            )

            st.subheader("Alert Settings")

            # Emergency keywords
            default_keywords = ", ".join(current_config['keywords']['emergency'])
            keywords = st.text_input("Emergency Voice Keywords (comma-separated)",
                                  value=default_keywords)

            # Notification settings
            notify_authorities = st.checkbox("Notify Authorities on Critical Alerts",
                                          value=current_config['notifications']['authorities'])

            notify_emergency_contacts = st.checkbox("Notify Emergency Contacts",
                                                 value=current_config['notifications']['emergency_contacts'])

            # Submit button
            submitted = st.form_submit_button("Save Settings")

            if submitted:
                # Update config
                current_config['thresholds']['object_detection'] = float(object_threshold)
                current_config['thresholds']['pose_detection'] = float(pose_threshold)
                current_config['keywords']['emergency'] = [k.strip() for k in keywords.split(",")]
                current_config['notifications']['authorities'] = notify_authorities
                current_config['notifications']['emergency_contacts'] = notify_emergency_contacts

                # Save config
                try:
                    with open("config/config.yaml", 'w') as f:
                        import yaml
                        yaml.dump(current_config, f)
                    st.success("Settings saved successfully!")

                    # Reinitialize models with new settings if they're loaded
                    if self.object_detector is not None:
                        self.object_detector.confidence_threshold = object_threshold

                    if self.pose_detector is not None:
                        self.pose_detector.min_detection_confidence = pose_threshold

                except Exception as e:
                    st.error(f"Error saving configuration: {e}")

        # Outside of form - Emergency contact settings
        st.subheader("🏥 Emergency Contacts")

        # Load current contacts
        contacts_file = "data/emergency_contacts.json"
        contacts = []

        try:
            if os.path.exists(contacts_file):
                with open(contacts_file, 'r') as f:
                    contacts = json.load(f)
        except Exception as e:
            st.error(f"Error loading emergency contacts: {e}")

        # Display current contacts
        if contacts:
            st.write("Current Emergency Contacts:")

            contact_df = pd.DataFrame(contacts)
            st.dataframe(contact_df, use_container_width=True)

            # Delete contact option
            if st.button("Delete Selected Contact"):
                st.warning("Feature not implemented yet. Please edit the JSON file directly.")

        # Add new contact form
        st.write("Add New Emergency Contact:")

        with st.form("add_contact_form"):
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Name")
                phone = st.text_input("Phone Number")

            with col2:
                email = st.text_input("Email")
                relation = st.text_input("Relation")

            add_contact = st.form_submit_button("Add Contact")

            if add_contact:
                if name and (phone or email):
                    new_contact = {
                        "name": name,
                        "phone": phone,
                        "email": email,
                        "relation": relation
                    }

                    contacts.append(new_contact)

                    try:
                        with open(contacts_file, 'w') as f:
                            json.dump(contacts, f, indent=4)
                        st.success(f"Added {name} to emergency contacts!")
                    except Exception as e:
                        st.error(f"Error saving contact: {e}")
                else:
                    st.error("Name and either Phone or Email are required!")

    def display_help(self):
        """Display help and documentation"""
        st.title("❓ Help & Documentation")

        st.markdown("""
        # AI-Powered Woman Safety System - User Guide

        This system uses artificial intelligence to monitor video feeds and detect potentially dangerous situations.

        ## Features

        ### 📹 Real-time Monitoring
        - Object detection to identify people and objects
        - Pose detection to analyze body language
        - Voice detection for emergency keywords

        ### 🚨 Alert System
        - Automatic alerts for suspicious activities
        - Emergency notification to authorities and contacts
        - Recording of incidents for documentation

        ### 📊 Analytics
        - Historical data analysis
        - Incident patterns and statistics
        - Export capabilities for documentation

        ## How to Use

        ### Dashboard
        1. Select your camera from the dropdown menu
        2. Click "Start Camera" to begin monitoring
        3. The system will automatically detect potential threats
        4. Alerts will be displayed in the right panel

        ### Settings
        - Adjust detection thresholds to reduce false positives
        - Configure emergency keywords for voice detection
        - Manage emergency contacts

        ### Analytics
        - View historical alerts and patterns
        - Generate reports for documentation

        ## Privacy & Security

        This system processes all video locally and does not upload footage to any external servers.
        Alert data is stored locally on your device and can be deleted at any time.

        ## Support

        For technical support or questions, please contact:
        - Email: support@womensafety.ai
        - Phone: +1-800-SAFE-NOW
        """)

    def add_mock_alert(self):
        """Add a mock alert for testing"""
        alert_types = [
            "Fall Detected - Person may need assistance",
            "Raised Hands Detected - Possible surrender gesture",
            "Voice Alert - Emergency keyword detected: 'help'",
            "Unusual Posture Detected - Possible altercation"
        ]

        # Create alert
        new_alert = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": np.random.choice(alert_types),
            "location_url": "https://maps.google.com/?q=37.7749,-122.4194",
            "severity": "High",
            "image_path": "data/sample_alert.jpg"  # Sample image path
        }

        # Add to history
        self.alert_history.append(new_alert)

        # Save to file
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.alert_history, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving alert: {e}")
            return False


def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="AI-Powered Woman Safety System",
        page_icon="🛡️",
        layout="wide"
    )

    # Initialize dashboard
    dashboard = Dashboard()

    # Create sidebar menu
    st.sidebar.title("🛡️ Safety System")

    menu_options = {
        "Dashboard": "🏠 Dashboard",
        "Analytics": "📈 Analytics",
        "Settings": "⚙️ Settings",
        "Help": "❓ Help & Documentation"
    }

    selection = st.sidebar.radio("Navigation", list(menu_options.values()))

    # Debug tools (hidden in production)
    if st.sidebar.checkbox("Show Debug Tools", False):
        st.sidebar.subheader("Debugging")

        if st.sidebar.button("Add Test Alert"):
            if dashboard.add_mock_alert():
                st.sidebar.success("Test alert added!")
            else:
                st.sidebar.error("Failed to add test alert")

    # Display selected page
    if selection == menu_options["Dashboard"]:
        dashboard.display_home()
    elif selection == menu_options["Analytics"]:
        dashboard.display_analytics()
    elif selection == menu_options["Settings"]:
        dashboard.display_settings()
    elif selection == menu_options["Help"]:
        dashboard.display_help()

    # Cleanup on app close - this may not actually run in Streamlit
    def cleanup():
        global video_capture, run_video
        run_video = False
        if video_capture is not None:
            video_capture.release()

    # Register cleanup
    import atexit
    atexit.register(cleanup)


if __name__ == "__main__":
    main()
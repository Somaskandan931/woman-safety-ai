import os
from twilio.rest import Client
from dotenv import load_dotenv
import time
from datetime import datetime
import json

# Optional: Try to import GPS utilities
try :
    from utils.gps_utils import get_current_location

    GPS_AVAILABLE = True
except ImportError :
    GPS_AVAILABLE = False


class AlertSystem :
    """
    Alert system for sending emergency alerts via SMS/WhatsApp
    """

    def __init__ ( self ) :
        """Initialize the alert system with Twilio credentials"""
        # Load environment variables
        load_dotenv()

        # Twilio credentials
        self.account_sid = os.getenv( "TWILIO_ACCOUNT_SID" )
        self.auth_token = os.getenv( "TWILIO_AUTH_TOKEN" )
        self.twilio_phone = os.getenv( "TWILIO_PHONE" )
        self.recipient_phone = os.getenv( "USER_PHONE" )

        # Alert history
        self.alert_history = []
        self.history_file = "data/alert_history.json"

        # Create client if credentials are available
        if self.account_sid and self.auth_token :
            self.client = Client( self.account_sid, self.auth_token )
            print( "📱 Alert system initialized with Twilio" )
        else :
            self.client = None
            print( "⚠️ Warning: Twilio credentials not found. Alerts will be logged but not sent." )

        # Create directory for alerts if it doesn't exist
        os.makedirs( "data/snapshots", exist_ok=True )

        # Try to load alert history
        self._load_history()

    def send_alert ( self, message, image_path=None ) :
        """
        Send an emergency alert

        Args:
            message (str): Alert message
            image_path (str, optional): Path to image to attach
        """
        timestamp = datetime.now().strftime( "%Y-%m-%d %H:%M:%S" )

        # Add location information if available
        location_str = ""
        location_url = ""

        if GPS_AVAILABLE :
            try :
                lat, lon = get_current_location()
                location_url = f"https://maps.google.com/?q={lat},{lon}"
                location_str = f"\nLocation: {location_url}"
            except Exception as e :
                print( f"Could not get location: {e}" )

        full_message = f"⚠️ EMERGENCY ALERT ⚠️\n{timestamp}\n{message}{location_str}"

        # Log the alert
        alert_data = {
            "timestamp" : timestamp,
            "message" : message,
            "image_path" : image_path,
            "location_url" : location_url
        }
        self.alert_history.append( alert_data )
        self._save_history()

        # Print alert to console
        print( "=" * 50 )
        print( full_message )
        print( "=" * 50 )

        # Send SMS via Twilio if configured
        if self.client and self.twilio_phone and self.recipient_phone :
            try :
                if image_path and os.path.exists( image_path ) :
                    # Send MMS with image
                    message = self.client.messages.create(
                        body=full_message,
                        from_=self.twilio_phone,
                        to=self.recipient_phone,
                        media_url=[f"file://{os.path.abspath( image_path )}"]
                    )
                    print( f"Alert sent with image: {message.sid}" )
                else :
                    # Send SMS without image
                    message = self.client.messages.create(
                        body=full_message,
                        from_=self.twilio_phone,
                        to=self.recipient_phone
                    )
                    print( f"Alert sent: {message.sid}" )
            except Exception as e :
                print( f"Error sending alert: {e}" )
        else :
            print( "Twilio not configured. Alert logged but not sent." )

    def send_whatsapp_alert ( self, message, image_path=None ) :
        """
        Send an emergency alert via WhatsApp

        Args:
            message (str): Alert message
            image_path (str, optional): Path to image to attach
        """
        if not self.client :
            print( "Twilio not configured. Cannot send WhatsApp alert." )
            return

        # WhatsApp uses a different format for the 'from' number
        whatsapp_from = f"whatsapp:{self.twilio_phone}"
        whatsapp_to = f"whatsapp:{self.recipient_phone}"

        try :
            if image_path and os.path.exists( image_path ) :
                # Send media message
                message = self.client.messages.create(
                    body=message,
                    from_=whatsapp_from,
                    to=whatsapp_to,
                    media_url=[f"file://{os.path.abspath( image_path )}"]
                )
            else :
                # Send text-only message
                message = self.client.messages.create(
                    body=message,
                    from_=whatsapp_from,
                    to=whatsapp_to
                )

            print( f"WhatsApp alert sent: {message.sid}" )

        except Exception as e :
            print( f"Error sending WhatsApp alert: {e}" )

    def _load_history ( self ) :
        """Load alert history from file"""
        try :
            if os.path.exists( self.history_file ) :
                with open( self.history_file, 'r' ) as f :
                    self.alert_history = json.load( f )
                print( f"Loaded {len( self.alert_history )} previous alerts from history" )
        except Exception as e :
            print( f"Error loading alert history: {e}" )
            self.alert_history = []

    def _save_history ( self ) :
        """Save alert history to file"""
        try :
            with open( self.history_file, 'w' ) as f :
                json.dump( self.alert_history, f, indent=2 )
        except Exception as e :
            print( f"Error saving alert history: {e}" )
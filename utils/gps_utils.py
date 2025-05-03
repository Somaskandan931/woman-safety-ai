"""
GPS utilities for getting the current location
This module tries several methods to get the device's location.
"""

import requests
import time
import os
import json

# Cache for the location to avoid excessive API calls
_location_cache = {
    'timestamp' : 0,
    'lat' : None,
    'lon' : None,
    'cache_duration' : 60  # Cache duration in seconds
}


def get_current_location () :
    """
    Get the current location using various methods

    Returns:
        tuple: (latitude, longitude)
    """
    global _location_cache

    # Check if we have a recent cached location
    current_time = time.time()
    if (_location_cache['lat'] is not None and
            current_time - _location_cache['timestamp'] < _location_cache['cache_duration']) :
        return _location_cache['lat'], _location_cache['lon']

    # Try different methods to get location

    # 1. Try to use the IP-API service
    try :
        location = get_location_from_ip()
        if location :
            _location_cache['lat'] = location[0]
            _location_cache['lon'] = location[1]
            _location_cache['timestamp'] = current_time
            return location
    except Exception as e :
        print( f"Could not get location from IP: {e}" )

    # 2. Try to load from a config file
    try :
        location = get_location_from_config()
        if location :
            _location_cache['lat'] = location[0]
            _location_cache['lon'] = location[1]
            _location_cache['timestamp'] = current_time
            return location
    except Exception as e :
        print( f"Could not get location from config: {e}" )

    # 3. Use a dummy location as last resort
    print( "⚠️ Warning: Using default location (0, 0). Please configure actual location." )
    return 0.0, 0.0


def get_location_from_ip () :
    """
    Get location from IP address using free IP-API service

    Returns:
        tuple: (latitude, longitude)
    """
    try :
        response = requests.get( 'http://ip-api.com/json/', timeout=5 )
        data = response.json()

        if data['status'] == 'success' :
            return data['lat'], data['lon']
    except Exception as e :
        print( f"IP location error: {e}" )

    return None


def get_location_from_config () :
    """
    Get location from a config file

    Returns:
        tuple: (latitude, longitude)
    """
    config_file = os.path.join( 'config', 'location.json' )

    # Check if config file exists
    if not os.path.exists( config_file ) :
        # Create a default config file
        os.makedirs( 'config', exist_ok=True )

        default_config = {
            'latitude' : 0.0,
            'longitude' : 0.0,
            'notes' : "Please update with your actual location"
        }

        with open( config_file, 'w' ) as f :
            json.dump( default_config, f, indent=2 )

        print( f"Created default location config at {config_file}. Please update it." )
        return None

    # Load config file
    with open( config_file, 'r' ) as f :
        config = json.load( f )

    # Get location
    lat = config.get( 'latitude' )
    lon = config.get( 'longitude' )

    if lat == 0.0 and lon == 0.0 :
        print( "⚠️ Warning: Using default location (0, 0). Please update config/location.json" )

    return lat, lon


# Optional: Integration with a GPS hardware module
# This is device-specific and requires additional hardware
def get_location_from_gps_module () :
    """
    Get location from a GPS module (requires additional hardware)

    Returns:
        tuple: (latitude, longitude)
    """
    try :
        # This is just a placeholder. Implementation depends on hardware.
        # For Raspberry Pi, you might use a USB GPS module with pyserial or gpsd.

        # Example using gpsd-py3 library:
        # from gps3 import gps3
        # gpsd_socket = gps3.GPSDSocket()
        # data_stream = gps3.DataStream()
        # gpsd_socket.connect()
        # gpsd_socket.watch()
        # for new_data in gpsd_socket:
        #     if new_data:
        #         data_stream.unpack(new_data)
        #         if data_stream.TPV['lat'] != 'n/a':
        #             return data_stream.TPV['lat'], data_stream.TPV['lon']

        print( "GPS module support not implemented" )
        return None
    except Exception as e :
        print( f"GPS hardware error: {e}" )
        return None


if __name__ == "__main__" :
    # Test the module
    print( "Testing location services..." )
    lat, lon = get_current_location()
    print( f"Current location: {lat}, {lon}" )

    if lat == 0 and lon == 0 :
        print( "Please update config/location.json with your actual coordinates" )
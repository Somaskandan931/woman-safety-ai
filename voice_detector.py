import speech_recognition as sr
import threading
import time
import os


class VoiceDetector :
    """
    Voice detector that listens for emergency keywords
    """

    def __init__ ( self, keywords=None, confidence_threshold=0.7 ) :
        """
        Initialize the voice detector

        Args:
            keywords (list): List of emergency keywords to detect
            confidence_threshold (float): Minimum confidence for keyword detection
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Default emergency keywords if none provided
        self.keywords = keywords or ["help", "save me", "please stop", "emergency"]
        self.confidence_threshold = confidence_threshold

        # State variables
        self.is_listening = False
        self.listen_thread = None
        self.callback = None

        # Adjust for ambient noise
        with self.microphone as source :
            self.recognizer.adjust_for_ambient_noise( source )

        print( f"🎤 Voice detector initialized with keywords: {self.keywords}" )

    def start_listening ( self, callback=None ) :
        """
        Start listening for keywords in a separate thread

        Args:
            callback (function): Callback function when a keyword is detected
        """
        if self.is_listening :
            print( "Voice detector is already listening" )
            return

        self.is_listening = True
        self.callback = callback

        # Start listening in a separate thread
        self.listen_thread = threading.Thread( target=self._listen_loop )
        self.listen_thread.daemon = True
        self.listen_thread.start()

        print( "Voice detector started listening" )

    def stop_listening ( self ) :
        """Stop the voice detector"""
        self.is_listening = False
        if self.listen_thread and self.listen_thread.is_alive() :
            self.listen_thread.join( timeout=1 )
        print( "Voice detector stopped" )

    def _listen_loop ( self ) :
        """Main listening loop"""
        while self.is_listening :
            try :
                # Use the microphone as the audio source
                with self.microphone as source :
                    print( "Listening..." )
                    audio = self.recognizer.listen( source, timeout=1, phrase_time_limit=5 )

                try :
                    # Use Google's speech recognition
                    text = self.recognizer.recognize_google( audio, show_all=False ).lower()
                    print( f"Heard: {text}" )

                    # Check for keywords
                    for keyword in self.keywords :
                        if keyword.lower() in text :
                            print( f"Detected keyword: {keyword}" )
                            if self.callback :
                                self.callback( keyword )
                            break

                except sr.UnknownValueError :
                    # Speech was unintelligible
                    pass
                except sr.RequestError as e :
                    print( f"Could not request results from Google Speech Recognition service: {e}" )

                    # Try offline recognition with Vosk if available
                    try :
                        from vosk import Model, KaldiRecognizer
                        if os.path.exists( "models/vosk-model-small-en-us-0.15" ) :
                            print( "Falling back to offline Vosk recognition" )
                            model = Model( "models/vosk-model-small-en-us-0.15" )
                            rec = KaldiRecognizer( model, 16000 )
                            rec.AcceptWaveform( audio.get_raw_data() )
                            result = rec.Result()
                            # Process Vosk results here
                    except ImportError :
                        print( "Vosk is not installed for offline recognition" )

            except sr.WaitTimeoutError :
                # Timeout waiting for speech input
                pass
            except Exception as e :
                print( f"Error in voice detection: {e}" )
                time.sleep( 1 )  # Prevent tight loop in case of recurring errors

        print( "Voice listening loop ended" )


class KeywordDetector :
    """
    Simple alternative implementation using pocketsphinx for offline keyword spotting
    This is more efficient for detecting specific keywords without transcribing all speech
    """

    def __init__ ( self, keywords=None ) :
        try :
            from pocketsphinx import LiveSpeech

            # Default keywords if none provided
            self.keywords = keywords or ["help", "save", "stop", "emergency"]

            # Set up LiveSpeech with the keywords
            self.live_speech = LiveSpeech(
                verbose=False,
                sampling_rate=16000,
                buffer_size=2048,
                no_search=False,
                full_utt=False,
                keyphrase=' '.join( self.keywords ),
                kws_threshold=1e-20  # Adjust this threshold as needed
            )

            self.is_listening = False
            self.listen_thread = None
            self.callback = None

            print( f"Keyword detector initialized with keywords: {self.keywords}" )

        except ImportError :
            print( "PocketSphinx not installed. Please install it for this feature." )
            self.live_speech = None

    def start_listening ( self, callback=None ) :
        """Start listening for keywords"""
        if not self.live_speech :
            print( "PocketSphinx not available" )
            return

        if self.is_listening :
            return

        self.is_listening = True
        self.callback = callback

        # Start in a new thread
        self.listen_thread = threading.Thread( target=self._listen_loop )
        self.listen_thread.daemon = True
        self.listen_thread.start()

    def _listen_loop ( self ) :
        """Main listening loop for keyword detection"""
        for phrase in self.live_speech :
            if not self.is_listening :
                break

            text = str( phrase ).lower()
            print( f"Heard keyword: {text}" )

            if self.callback :
                self.callback( text )

    def stop_listening ( self ) :
        """Stop keyword detection"""
        self.is_listening = False
        if self.listen_thread and self.listen_thread.is_alive() :
            self.listen_thread.join( timeout=1 )


if __name__ == "__main__" :
    # Simple test
    def on_keyword ( keyword ) :
        print( f"⚠️ EMERGENCY KEYWORD DETECTED: {keyword}" )


    detector = VoiceDetector()
    try :
        detector.start_listening( callback=on_keyword )
        print( "Listening for emergency keywords... (Press Ctrl+C to stop)" )
        while True :
            time.sleep( 1 )
    except KeyboardInterrupt :
        print( "\nStopping..." )
    finally :
        detector.stop_listening()
import pyaudio
import wave
import threading
import time
import numpy as np

class AudioRecorder:
    def __init__(self, channels=1, rate=44100, chunk=1024, format_=pyaudio.paInt16):
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.format = format_
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.recorder_thread = None
        
    def start_recording(self):
        """Start recording audio from the default microphone"""
        self.frames = []
        self.is_recording = True
        
        # Start recording in a separate thread
        self.recorder_thread = threading.Thread(target=self._record)
        self.recorder_thread.daemon = True
        self.recorder_thread.start()
        
    def stop_recording(self):
        """Stop the ongoing recording"""
        self.is_recording = False
        if self.recorder_thread and self.recorder_thread.is_alive():
            self.recorder_thread.join()
        
    def _record(self):
        """Internal method to record audio in a separate thread"""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            while self.is_recording:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
                
        except Exception as e:
            print(f"Error during recording: {e}")
            
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
    def save_recording(self, filename):
        """Save the recorded audio to a WAV file"""
        if not self.frames:
            return False
            
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
            return True
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False
            
    def close(self):
        """Clean up resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        
    def get_rms_levels(self):
        """Get RMS levels for the last recorded chunk for visualization"""
        if not self.frames:
            return 0
            
        # Process the latest frame
        try:
            latest = np.frombuffer(self.frames[-1], dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(latest)))
            return rms / 32767  # Normalize to 0-1
        except:
            return 0

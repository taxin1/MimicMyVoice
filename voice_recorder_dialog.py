import os
import tempfile
import time
import numpy as np
import soundfile as sf
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtMultimedia import QAudioRecorder, QAudioEncoderSettings, QMultimedia

# Import recorder utils but handle import error
try:
    from recorder_utils import AudioRecorder
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

class VoiceRecorderDialog(QDialog):
    recording_complete = pyqtSignal(str)  # Signal to emit when recording is completed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Record Your Voice")
        self.setMinimumSize(400, 250)
        self.recorded_path = None
        self.max_duration = 60 
        self.recorder = QAudioRecorder()
        self.is_recording = False  
        self.setup_ui()
        self.setup_recorder()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Click 'Start Recording' to begin recording your voice.\n"
                             "Click 'Stop Recording' when you're finished.\n"
                             "Speak clearly and consistently for best results.")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        # Status label
        self.status_label = QLabel("Ready to record")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Timer display for recording
        self.time_label = QLabel("00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(self.time_label)
        
        # Volume level indicator
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        layout.addWidget(self.level_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        self.listen_btn = QPushButton("Listen")
        self.listen_btn.clicked.connect(self.listen_recording)
        self.listen_btn.setEnabled(False)
        
        self.use_btn = QPushButton("Use Recording")
        self.use_btn.clicked.connect(self.use_recording)
        self.use_btn.setEnabled(False)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.listen_btn)
        button_layout.addWidget(self.use_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def setup_recorder(self):
        # Setup audio recorder
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "recording.wav")
        
        # Determine which recording implementation to use
        self.use_pyaudio = False
        
        try:
            # Configure QAudioRecorder settings
            audio_settings = QAudioEncoderSettings()
            audio_settings.setCodec("audio/pcm")
            audio_settings.setQuality(QMultimedia.HighQuality)
            audio_settings.setSampleRate(44100)
            audio_settings.setChannelCount(1)  # Mono
            
            # Apply settings to recorder
            self.recorder.setEncodingSettings(audio_settings)
            self.recorder.setOutputLocation(Qt.QUrl.fromLocalFile(self.output_file))
            
        except Exception:
            # Fall back to PyAudio if QAudioRecorder fails
            if PYAUDIO_AVAILABLE:
                self.use_pyaudio = True
                self.pyaudio_recorder = AudioRecorder()
            else:
                QMessageBox.critical(self, "Recording Error", 
                                  "Audio recording is not available. Please install PyAudio.")
                
        # Setup progress timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        # Reset UI
        self.time_label.setText("00:00")
        self.level_bar.setValue(0)
        self.status_label.setText("Recording... Press 'Stop Recording' when finished")
        self.record_btn.setText("Stop Recording")
        self.listen_btn.setEnabled(False)
        self.use_btn.setEnabled(False)
        
        # Start recording based on available implementation
        if self.use_pyaudio:
            self.pyaudio_recorder.start_recording()
        else:
            self.recorder.record()
        
        # Set recording state
        self.is_recording = True
        
        # Start progress timer
        self.elapsed_time = 0
        self.timer.start(100)  # Update every 100ms for timer display
        
    def stop_recording(self):
        # Stop recording based on implementation
        if self.use_pyaudio:
            self.pyaudio_recorder.stop_recording()
        else:
            self.recorder.stop()
            
        # Update recording state
        self.is_recording = False
        
        self.timer.stop()
        self.status_label.setText("Recording complete!")
        self.record_btn.setText("Record Again")
        
        # Process recorded audio to ensure PCM WAV format
        try:
            # Create a permanent file copy
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            self.recorded_path = tmp_wav.name
            
            if self.use_pyaudio:
                # Save PyAudio recording
                self.pyaudio_recorder.save_recording(self.recorded_path)
            else:
                # Convert Qt recording to standard format using soundfile
                data, sr = sf.read(self.output_file)
                sf.write(self.recorded_path, data, sr, subtype="PCM_16")
            
            # Enable buttons only if recording was successful
            self.listen_btn.setEnabled(True)
            self.use_btn.setEnabled(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to process recording: {str(e)}")
            self.status_label.setText("Recording failed!")
            self.listen_btn.setEnabled(False)
            self.use_btn.setEnabled(False)
        
    def update_progress(self):
        # Only update if we're still recording
        if not self.is_recording:
            return
            
        self.elapsed_time += 1
        
        # Calculate minutes and seconds
        total_seconds = self.elapsed_time // 10  # Since timer fires every 100ms
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        self.time_label.setText(f"{minutes:02d}:{seconds:02d}")
        
        # Update audio level indicator
        if self.use_pyaudio and hasattr(self, 'pyaudio_recorder'):
            try:
                # Get audio level from PyAudio recorder
                level = int(self.pyaudio_recorder.get_rms_levels() * 100)
                self.level_bar.setValue(level)
            except Exception:
                # Fallback if get_rms_levels fails
                import random
                self.level_bar.setValue(random.randint(10, 80))
        else:
            # For QAudioRecorder, we can't easily get the audio level
            # Use random values for visual feedback
            import random
            self.level_bar.setValue(random.randint(10, 80))
            
        # If we've reached the maximum duration (for safety), warn the user
        if total_seconds >= self.max_duration:
            self.status_label.setText("Maximum recording length reached!")
            # But don't auto-stop, let the user decide
            
    def listen_recording(self):
        if self.recorded_path and os.path.exists(self.recorded_path):
            # Use the parent's audio player if available
            if hasattr(self.parent(), 'player'):
                from PyQt5.QtCore import QUrl
                from PyQt5.QtMultimedia import QMediaContent
                
                self.parent().player.setMedia(QMediaContent(QUrl.fromLocalFile(self.recorded_path)))
                self.parent().player.play()
            else:
                QMessageBox.information(self, "Playback Not Available", 
                                      "Audio playback is not available in this dialog.")
        
    def use_recording(self):
        if self.recorded_path and os.path.exists(self.recorded_path):
            self.recording_complete.emit(self.recorded_path)
            self.accept()
        else:
            QMessageBox.warning(self, "No Recording", "No recording available to use.")
    
    def closeEvent(self, event):
        # Stop recording if it's still in progress
        if self.is_recording:
            self.stop_recording()
        
        # Clean up resources
        if self.use_pyaudio:
            self.pyaudio_recorder.close()
        
        # Don't delete the recorded_path file as it might be used by the parent
        # It will be managed by the parent window
        event.accept()

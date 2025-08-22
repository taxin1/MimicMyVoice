import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QSpinBox,
    QMessageBox, QCheckBox, QComboBox,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QFont
import librosa
import numpy as np
from audio_utils import convert_to_pcm_wav, extract_lpc_env, N_FFT
from spectral_plot import SpectralPlot
from audio_processor import AudioProcessor
from tts_dialog import TTSDialog
from voice_recorder_dialog import VoiceRecorderDialog

class VoiceConversionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ref_path = None
        self.tts_path = None
        self.temp_tts_path = None
        self.output_path = None
        self.init_ui()
        self.player = QMediaPlayer()

    def init_ui(self):
        self.setWindowTitle("LPC Residual Substitution Voice Conversion")
        self.setGeometry(100, 100, 1200, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title_label = QLabel("LPC Residual Substitution Voice Conversion")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        file_group = QGroupBox("Audio Files")
        file_layout = QVBoxLayout(file_group)

        # Reference audio controls
        ref_layout = QHBoxLayout()
        self.ref_label = QLabel("Reference Speaker: Not selected")
        self.ref_btn = QPushButton("Browse Reference Audio")
        self.ref_btn.clicked.connect(self.load_reference_audio)
        self.record_ref_btn = QPushButton("Record Your Voice")
        self.record_ref_btn.clicked.connect(self.record_reference_audio)
        ref_layout.addWidget(self.ref_label)
        ref_layout.addWidget(self.ref_btn)
        ref_layout.addWidget(self.record_ref_btn)
        file_layout.addLayout(ref_layout)
        
        # Noise reduction options
        noise_reduction_layout = QHBoxLayout()
        self.noise_reduction_cb = QCheckBox("Enable Noise Reduction")
        self.noise_reduction_cb.setChecked(True)
        self.noise_method_combo = QComboBox()
        self.noise_method_combo.addItem("Spectral Subtraction", "spectral_subtraction")
        self.noise_method_combo.addItem("Median Filter", "median_filter")
        self.denoise_btn = QPushButton("Apply Noise Reduction")
        self.denoise_btn.clicked.connect(self.apply_noise_reduction)
        self.denoise_btn.setEnabled(False)
        
        noise_reduction_layout.addWidget(self.noise_reduction_cb)
        noise_reduction_layout.addWidget(self.noise_method_combo)
        noise_reduction_layout.addWidget(self.denoise_btn)
        file_layout.addLayout(noise_reduction_layout)

        tts_layout = QHBoxLayout()
        self.tts_label = QLabel("TTS Audio: Not selected")
        self.tts_btn = QPushButton("Browse TTS Audio")
        self.tts_btn.clicked.connect(self.load_tts_audio)
        self.create_tts_btn = QPushButton("Create TTS Audio")
        self.create_tts_btn.clicked.connect(self.create_tts_audio)
        tts_layout.addWidget(self.tts_label)
        tts_layout.addWidget(self.tts_btn)
        tts_layout.addWidget(self.create_tts_btn)
        file_layout.addLayout(tts_layout)
        layout.addWidget(file_group)

        params_group = QGroupBox("Processing Parameters")
        params_layout = QHBoxLayout(params_group)

        frame_len_layout = QVBoxLayout()
        frame_len_layout.addWidget(QLabel("Frame Length:"))
        self.frame_len_spin = QSpinBox()
        self.frame_len_spin.setRange(256, 4096)
        self.frame_len_spin.setValue(1024)
        frame_len_layout.addWidget(self.frame_len_spin)
        params_layout.addLayout(frame_len_layout)

        hop_len_layout = QVBoxLayout()
        hop_len_layout.addWidget(QLabel("Hop Length:"))
        self.hop_len_spin = QSpinBox()
        self.hop_len_spin.setRange(64, 2048)
        self.hop_len_spin.setValue(512)
        hop_len_layout.addWidget(self.hop_len_spin)
        params_layout.addLayout(hop_len_layout)

        lpc_order_layout = QVBoxLayout()
        lpc_order_layout.addWidget(QLabel("LPC Order:"))
        self.lpc_spin = QSpinBox()
        self.lpc_spin.setRange(8, 32)
        self.lpc_spin.setValue(16)
        lpc_order_layout.addWidget(self.lpc_spin)
        params_layout.addLayout(lpc_order_layout)

        layout.addWidget(params_group)

        controls_layout = QHBoxLayout()
        self.process_btn = QPushButton("Process Audio")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        controls_layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        controls_layout.addWidget(self.progress_bar)
        layout.addLayout(controls_layout)

        self.plot_canvas = SpectralPlot()
        layout.addWidget(self.plot_canvas)

        playback_group = QGroupBox("Playback")
        playback_layout = QHBoxLayout(playback_group)

        self.play_ref_btn = QPushButton("Play Reference")
        self.play_ref_btn.setEnabled(False)
        self.play_ref_btn.clicked.connect(lambda: self.play_audio("reference"))
        playback_layout.addWidget(self.play_ref_btn)
        
        self.play_orig_btn = QPushButton("Play Original TTS")
        self.play_orig_btn.setEnabled(False)
        self.play_orig_btn.clicked.connect(lambda: self.play_audio("original"))
        playback_layout.addWidget(self.play_orig_btn)

        self.play_proc_btn = QPushButton("Play Processed")
        self.play_proc_btn.setEnabled(False)
        self.play_proc_btn.clicked.connect(lambda: self.play_audio("processed"))
        playback_layout.addWidget(self.play_proc_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        playback_layout.addWidget(self.stop_btn)

        layout.addWidget(playback_group)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(100)
        layout.addWidget(self.status_log)

    def log(self, message):
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        self.status_log.append(f"[{now}] {message}")

    def load_reference_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Speaker Audio", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if path:
            self.ref_path = path
            self.ref_label.setText(f"Reference: {os.path.basename(path)}")
            self.log(f"Reference audio selected: {os.path.basename(path)}")
            self.play_ref_btn.setEnabled(True)
            self.denoise_btn.setEnabled(True)
            self.check_ready()
            
    def record_reference_audio(self):
        """Open dialog to record reference voice"""
        dialog = VoiceRecorderDialog(self)
        dialog.recording_complete.connect(self.handle_recording_completed)
        dialog.exec_()
        
    def handle_recording_completed(self, recorded_path):
        """Handle when voice recording is successfully completed"""
        if recorded_path and os.path.exists(recorded_path):
            self.ref_path = recorded_path
            self.ref_label.setText(f"Reference: Recorded Voice")
            self.log(f"Reference voice recorded and loaded")
            self.play_ref_btn.setEnabled(True)
            self.denoise_btn.setEnabled(True)
            
            # Auto-apply noise reduction if checkbox is checked
            if self.noise_reduction_cb.isChecked():
                self.apply_noise_reduction()
            else:
                self.check_ready()

    def load_tts_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TTS Audio", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if path:
            self.log("Converting TTS audio to standard PCM WAV for compatibility...")
            try:
                converted_path = convert_to_pcm_wav(path)
                if self.temp_tts_path and os.path.exists(self.temp_tts_path):
                    try:
                        os.remove(self.temp_tts_path)
                    except Exception:
                        pass
                self.temp_tts_path = converted_path
            except Exception as e:
                QMessageBox.critical(self, "Conversion Error", f"Failed to convert TTS audio:\n{e}")
                return
            self.tts_path = self.temp_tts_path
            self.tts_label.setText(f"TTS: {os.path.basename(path)} (converted)")
            self.log(f"TTS audio converted and loaded: {os.path.basename(path)}")
            self.play_orig_btn.setEnabled(True)
            self.check_ready()

    def apply_noise_reduction(self):
        """Apply noise reduction to reference audio"""
        if not self.ref_path or not os.path.exists(self.ref_path):
            return
            
        self.log("Applying noise reduction to reference audio...")
        
        try:
            # Get selected method
            method = self.noise_method_combo.currentData()
            
            # Create a temporary file for the denoised audio
            from audio_utils import reduce_noise
            
            # Apply noise reduction
            params = {}
            if method == "spectral_subtraction":
                params = {'noise_factor': 1.5}  # Adjust based on testing
            elif method == "median_filter":
                params = {'filter_size': 3}
                
            denoised_path = reduce_noise(
                self.ref_path, 
                method=method,
                **params
            )
            
            # Update reference path
            self.ref_path = denoised_path
            self.log(f"Noise reduction applied successfully using {method}")
            
            # Update UI
            self.ref_label.setText(f"Reference: Denoised Audio")
            
        except Exception as e:
            QMessageBox.critical(self, "Noise Reduction Error", f"Failed to apply noise reduction:\n{e}")
            self.log(f"Noise reduction failed: {e}")
        
        self.check_ready()
            
    def check_ready(self):
        if self.ref_path and self.tts_path:
            self.process_btn.setEnabled(True)
            self.log("Ready to process audio.")

    def start_processing(self):
        self.log("Starting voice conversion processing...")
        self.stop_audio()
        self.player.setMedia(QMediaContent())
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.processor = AudioProcessor(
            self.ref_path,
            self.tts_path,
            self.lpc_spin.value(),
            self.frame_len_spin.value(),
            self.hop_len_spin.value(),
        )
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.finished.connect(self.finished_processing)
        self.processor.error.connect(self.processing_error)
        self.processor.start()

    def finished_processing(self, output_file):
        self.output_path = output_file
        self.log(f"Processing completed. Output saved to: {os.path.basename(output_file)}")
        self.process_btn.setEnabled(True)
        self.play_proc_btn.setEnabled(True)
        try:
            y_ref, sr = librosa.load(self.ref_path, sr=None)
            y_tts, _ = librosa.load(self.tts_path, sr=sr)
            y_proc, _ = librosa.load(self.output_path, sr=sr)
            freq_grid = np.linspace(0, sr / 2, N_FFT // 2 + 1)
            w_ref, env_ref = extract_lpc_env(y_ref, sr, self.lpc_spin.value())
            w_tts, env_tts = extract_lpc_env(y_tts, sr, self.lpc_spin.value())
            w_proc, env_proc = extract_lpc_env(y_proc, sr, self.lpc_spin.value())
            freq_grid_ref = np.clip(freq_grid, w_ref.min(), w_ref.max())
            freq_grid_tts = np.clip(freq_grid, w_tts.min(), w_tts.max())
            freq_grid_proc = np.clip(freq_grid, w_proc.min(), w_proc.max())
            env_ref_interp = np.interp(freq_grid_ref, w_ref, env_ref)
            env_tts_interp = np.interp(freq_grid_tts, w_tts, env_tts)
            env_proc_interp = np.interp(freq_grid_proc, w_proc, env_proc)
            self.plot_canvas.plot_envelopes(freq_grid, env_ref_interp, env_tts_interp, env_proc_interp)
        except Exception as e:
            self.log(f"Plot update failed: {e}")

    def processing_error(self, err):
        self.process_btn.setEnabled(True)
        self.log(f"Processing failed: {err}")
        QMessageBox.critical(self, "Error", f"Voice conversion failed:\n{err}")

    def create_tts_audio(self):
        """Open dialog to create TTS audio from text input"""
        dialog = TTSDialog(self)
        dialog.tts_generated.connect(self.handle_tts_generated)
        dialog.exec_()
        
    def handle_tts_generated(self, tts_path):
        """Handle when TTS is successfully generated"""
        if tts_path and os.path.exists(tts_path):
            # Clean up previous temp file if exists
            if self.temp_tts_path and os.path.exists(self.temp_tts_path):
                try:
                    os.remove(self.temp_tts_path)
                except Exception:
                    pass
                    
            # Set new TTS path
            self.temp_tts_path = tts_path
            self.tts_path = tts_path
            self.tts_label.setText(f"TTS: Generated TTS Audio")
            self.log(f"TTS audio generated and loaded")
            self.play_orig_btn.setEnabled(True)
            self.check_ready()
            
    def play_audio(self, mode):
        if mode == "reference" and self.ref_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.ref_path)))
            self.log("Playing reference audio")
        elif mode == "original" and self.tts_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.tts_path)))
            self.log("Playing original TTS audio")
        elif mode == "processed" and self.output_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.output_path)))
            self.log("Playing converted audio")
        else:
            return
        self.player.play()

    def stop_audio(self):
        self.player.stop()
        self.log("Playback stopped")

    def closeEvent(self, event):
        # Clean up temporary TTS file
        if self.temp_tts_path and os.path.exists(self.temp_tts_path):
            try:
                os.remove(self.temp_tts_path)
            except Exception:
                pass
                
        # Clean up temporary recorded reference file if it's in a temp directory
        if self.ref_path and os.path.exists(self.ref_path) and ("tmp" in self.ref_path.lower() or "temp" in self.ref_path.lower()):
            try:
                os.remove(self.ref_path)
            except Exception:
                pass
                
        event.accept()

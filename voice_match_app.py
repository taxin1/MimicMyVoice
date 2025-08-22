import sys
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QSpinBox,
    QCheckBox, QMessageBox, QDialog, QComboBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import scipy.signal as sg
import pyttsx3

N_FFT = 2048  # For plots and LPC


def extract_lpc_env(y, sr, order, frame_length=1024, hop_length=512, n_fft=N_FFT):
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    envs = []
    for frame in frames:
        wframe = frame * np.hamming(len(frame))
        a = librosa.lpc(wframe, order=order)
        w_freq, h_freq = sg.freqz([1], a, worN=n_fft, fs=sr)
        envs.append(np.abs(h_freq))
    mean_env = np.mean(np.array(envs), axis=0)
    return w_freq, mean_env


def convert_to_pcm_wav(input_path):
    y, sr = librosa.load(input_path, sr=None)
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_wav.name, y, sr, subtype="PCM_16")
    return tmp_wav.name


def extract_lpc_env_single_plot(self, freqs, env_ref, env_tts, env_proc):
    self.ax1.clear()
    self.ax1.plot(freqs, 20 * np.log10(env_ref + 1e-12), "b-", label="Reference", linewidth=2)
    self.ax1.plot(freqs, 20 * np.log10(env_tts + 1e-12), "r-", label="TTS", linewidth=2)
    self.ax1.plot(freqs, 20 * np.log10(env_proc + 1e-12), "m-", label="Processed", linewidth=2)
    self.ax1.set_title("LPC Spectral Envelopes")
    self.ax1.set_ylabel("Magnitude (dB)")
    self.ax1.legend()
    self.ax1.grid(True, alpha=0.3)
    self.draw()


class SpectralPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax1 = self.fig.add_subplot(111)
        self.fig.tight_layout()

    def plot_envelopes(self, freqs, env_ref, env_tts, env_proc):
        self.ax1.clear()
        self.ax1.plot(freqs, 20 * np.log10(env_ref + 1e-12), "b-", label="Reference", linewidth=2)
        self.ax1.plot(freqs, 20 * np.log10(env_tts + 1e-12), "r-", label="TTS", linewidth=2)
        self.ax1.plot(freqs, 20 * np.log10(env_proc + 1e-12), "m-", label="Processed", linewidth=2)
        self.ax1.set_title("LPC Spectral Envelopes")
        self.ax1.set_xlabel("Frequency (Hz)")
        self.ax1.set_ylabel("Magnitude (dB)")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.draw()


def extract_lpc(y_frame, order):
    return librosa.lpc(y_frame, order=order)


def lpc_residual(frame, a):
    return sg.lfilter(a, [1.0], frame)


def resynthesize_from_residual(residual, a):
    return sg.lfilter([1.0], a, residual)


def overlap_add(frames, hop_length):
    n_frames, frame_len = frames.shape
    sig_len = frame_len + hop_length * (n_frames - 1)
    out = np.zeros(sig_len)
    for i in range(n_frames):
        start = i * hop_length
        out[start : start + frame_len] += frames[i]
    return out


class AudioProcessor(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, ref_path, tts_path, lpc_order, frame_length, hop_length):
        super().__init__()
        self.ref_path = ref_path
        self.tts_path = tts_path
        self.lpc_order = lpc_order
        self.frame_length = frame_length
        self.hop_length = hop_length

    def run(self):
        try:
            self.progress.emit(5)
            y_ref, sr = librosa.load(self.ref_path, sr=None)
            y_tts, _ = librosa.load(self.tts_path, sr=sr)
            self.progress.emit(15)

            frames_ref = librosa.util.frame(y_ref, frame_length=self.frame_length, hop_length=self.hop_length).T
            frames_tts = librosa.util.frame(y_tts, frame_length=self.frame_length, hop_length=self.hop_length).T

            n_frames = min(len(frames_ref), len(frames_tts))

            processed_frames = []

            for i in range(n_frames):
                frame_r = frames_ref[i] * np.hamming(self.frame_length)
                frame_t = frames_tts[i] * np.hamming(self.frame_length)

                a_ref = extract_lpc(frame_r, self.lpc_order)
                a_tts = extract_lpc(frame_t, self.lpc_order)

                residual_ref = lpc_residual(frame_r, a_ref)
                synth_frame = resynthesize_from_residual(residual_ref, a_tts)

                processed_frames.append(synth_frame)

                if i % max(1, n_frames // 20) == 0:
                    self.progress.emit(15 + int(70 * i / n_frames))

            self.progress.emit(90)
            processed_frames = np.array(processed_frames)

            y_out = overlap_add(processed_frames, self.hop_length)

            maxv = np.max(np.abs(y_out))
            if maxv > 0:
                y_out = y_out / maxv * 0.95

            self.progress.emit(95)
            output_path = self.tts_path.replace(".wav", "_converted.wav")
            sf.write(output_path, y_out, sr, subtype="PCM_16")
            self.progress.emit(100)
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(str(e))


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

        ref_layout = QHBoxLayout()
        self.ref_label = QLabel("Reference Speaker: Not selected")
        self.ref_btn = QPushButton("Browse Reference Audio")
        self.ref_btn.clicked.connect(self.load_reference_audio)
        ref_layout.addWidget(self.ref_label)
        ref_layout.addWidget(self.ref_btn)
        file_layout.addLayout(ref_layout)

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
        result = dialog.exec_()
        
        if result == QDialog.Accepted and dialog.tts_path:
            # Clean up previous temp file if exists
            if self.temp_tts_path and os.path.exists(self.temp_tts_path):
                try:
                    os.remove(self.temp_tts_path)
                except Exception:
                    pass
                    
            # Set new TTS path
            self.temp_tts_path = dialog.tts_path
            self.tts_path = dialog.tts_path
            self.tts_label.setText("TTS: Generated TTS Audio")
            self.log("TTS audio generated and loaded")
            self.play_orig_btn.setEnabled(True)
            self.check_ready()
            
    def play_audio(self, mode):
        if mode == "original" and self.tts_path:
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
        if self.temp_tts_path and os.path.exists(self.temp_tts_path):
            try:
                os.remove(self.temp_tts_path)
            except Exception:
                pass
        event.accept()


class TTSDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text-to-Speech Generator")
        self.setMinimumSize(600, 400)
        self.tts_path = None
        self.engine = None
        self.setup_ui()
        self.get_available_voices()
        
    def get_available_voices(self):
        """Get available voices from pyttsx3 and organize by gender"""
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            
            # Clear previous items
            self.voice_combo.clear()
            
            # Organize voices by gender
            male_voices = []
            female_voices = []
            
            for i, voice in enumerate(voices):
                name = voice.name
                # Determine gender based on voice properties
                is_male = False
                
                # Check voice gender property if available
                if hasattr(voice, 'gender'):
                    is_male = "male" in str(voice.gender).lower()
                
                # Use name as a fallback for determining gender
                if not is_male:
                    # Common male voice identifiers
                    male_indicators = ["david", "mark", "james", "paul", "george", "mike", 
                                      "male", "man", "boy", "guy", "tom", "sam"]
                    
                    # Check if the voice name contains male indicators
                    name_lower = name.lower()
                    for indicator in male_indicators:
                        if indicator in name_lower:
                            is_male = True
                            break
                
                if is_male:
                    male_voices.append((i, name))
                else:
                    female_voices.append((i, name))
            
            # Add voices to combo box with gender categories
            if male_voices:
                self.voice_combo.addItem("--- Male Voices ---", None)
                for idx, (i, name) in enumerate(male_voices):
                    self.voice_combo.addItem(f"Male: {name}", i)
            
            if female_voices:
                self.voice_combo.addItem("--- Female Voices ---", None)
                for idx, (i, name) in enumerate(female_voices):
                    self.voice_combo.addItem(f"Female: {name}", i)
            
            # Enable the voice selection if voices were found
            has_voices = len(male_voices) > 0 or len(female_voices) > 0
            self.voice_combo.setEnabled(has_voices)
            self.use_system_voices_radio.setEnabled(has_voices)
            
            if not has_voices:
                self.voice_combo.addItem("No system voices available", None)
                self.voice_gender_combo.setEnabled(False)
            else:
                # Select first valid voice
                for i in range(1, self.voice_combo.count()):  # Start from 1 to skip headers
                    if self.voice_combo.itemData(i) is not None:
                        self.voice_combo.setCurrentIndex(i)
                        break
                
        except Exception as e:
            self.voice_combo.addItem("Error loading system voices", None)
            self.voice_combo.setEnabled(False)
            self.use_system_voices_radio.setEnabled(False)
            print(f"Error getting voices: {e}")
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Text input
        layout.addWidget(QLabel("Enter text to synthesize:"))
        self.text_input = QTextEdit()
        layout.addWidget(self.text_input)
        
        # Voice Selection Group
        voice_group = QGroupBox("Voice Selection")
        voice_layout = QVBoxLayout(voice_group)
        
        # Gender selection
        gender_layout = QHBoxLayout()
        gender_layout.addWidget(QLabel("Gender:"))
        self.voice_gender_combo = QComboBox()
        self.voice_gender_combo.addItem("Male")
        self.voice_gender_combo.addItem("Female")
        self.voice_gender_combo.currentIndexChanged.connect(self.filter_voices_by_gender)
        gender_layout.addWidget(self.voice_gender_combo)
        voice_layout.addLayout(gender_layout)
        
        # Voice selection
        voice_selection_layout = QHBoxLayout()
        voice_selection_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        voice_selection_layout.addWidget(self.voice_combo)
        voice_layout.addLayout(voice_selection_layout)
        
        # Voice rate (speed)
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Speed:"))
        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(50, 300)
        self.rate_spin.setValue(200)
        self.rate_spin.setSingleStep(10)
        rate_layout.addWidget(self.rate_spin)
        voice_layout.addLayout(rate_layout)
        
        layout.addWidget(voice_group)
        
        # Keep the Google TTS options group for backwards compatibility
        # but hide it in the UI as we're focusing on system voices
        self.google_options = QGroupBox("Google TTS Options")
        google_layout = QHBoxLayout(self.google_options)
        google_layout.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh-CN': 'Chinese (Simplified)',
        }
        for code, name in languages.items():
            self.lang_combo.addItem(name, code)
        google_layout.addWidget(self.lang_combo)
        self.google_options.setVisible(False)
        layout.addWidget(self.google_options)
        
        # These variables are kept for backwards compatibility
        self.system_options = voice_group
        self.use_google_radio = None
        self.use_system_voices_radio = None
        
        # Set initial state
        self.toggle_tts_options()
        
        # Buttons
        button_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Speech")
        self.generate_btn.clicked.connect(self.generate_speech)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def filter_voices_by_gender(self):
        """Filter voices by the selected gender"""
        if not self.engine:
            return
            
        selected_gender = self.voice_gender_combo.currentText()
        current_index = self.voice_combo.currentIndex()
        current_data = self.voice_combo.currentData()
        
        # Store the current voice selection if possible
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        
        voices = self.engine.getProperty('voices')
        male_voices = []
        female_voices = []
        
        for i, voice in enumerate(voices):
            name = voice.name
            # Determine gender based on voice properties
            is_male = False
            
            # Check voice gender property if available
            if hasattr(voice, 'gender'):
                is_male = "male" in str(voice.gender).lower()
            
            # Use name as a fallback for determining gender
            if not is_male:
                # Common male voice identifiers
                male_indicators = ["david", "mark", "james", "paul", "george", "mike", 
                                  "male", "man", "boy", "guy", "tom", "sam"]
                
                # Check if the voice name contains male indicators
                name_lower = name.lower()
                for indicator in male_indicators:
                    if indicator in name_lower:
                        is_male = True
                        break
            
            if is_male:
                male_voices.append((i, name))
            else:
                female_voices.append((i, name))
        
        # Add only the selected gender voices
        if selected_gender == "Male" and male_voices:
            for i, name in male_voices:
                self.voice_combo.addItem(name, i)
        elif selected_gender == "Female" and female_voices:
            for i, name in female_voices:
                self.voice_combo.addItem(name, i)
        
        # If no voices for the selected gender, add a message
        if self.voice_combo.count() == 0:
            self.voice_combo.addItem(f"No {selected_gender.lower()} voices found", None)
            self.voice_combo.setEnabled(False)
        else:
            self.voice_combo.setEnabled(True)
            # Try to restore the previous selection if it's still valid
            if current_data is not None:
                index = self.voice_combo.findData(current_data)
                if index >= 0:
                    self.voice_combo.setCurrentIndex(index)
        
        self.voice_combo.blockSignals(False)
    
    def generate_speech(self):
        """Generate TTS from the entered text"""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Text", "Please enter some text to synthesize.")
            return
        
        self.generate_btn.setEnabled(False)
        self.status_label.setText("Generating speech... Please wait.")
        
        try:
            # Create a temporary WAV file
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_wav.close()
            
            # Use System voices (pyttsx3)
            if not self.engine:
                self.engine = pyttsx3.init()
            
            # Set voice
            voice_index = self.voice_combo.currentData()
            voices = self.engine.getProperty('voices')
            if voice_index is not None and voices and 0 <= voice_index < len(voices):
                self.engine.setProperty('voice', voices[voice_index].id)
                
                # Set speech rate (default is 200)
                rate = self.rate_spin.value()
                self.engine.setProperty('rate', rate)
                
                # Generate and save speech
                self.engine.save_to_file(text, tmp_wav.name)
                self.engine.runAndWait()
            else:
                QMessageBox.warning(self, "Voice Selection", "Please select a valid voice.")
                self.generate_btn.setEnabled(True)
                return
            
            self.tts_path = tmp_wav.name
            self.accept()  # Close dialog when done
            
        except Exception as e:
            QMessageBox.critical(self, "TTS Generation Error", str(e))
            self.status_label.setText("Error generating speech.")
            
        finally:
            self.generate_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    try:
        import librosa
        import scipy
        import soundfile
        import matplotlib
        import pyttsx3
    except ImportError as e:
        QMessageBox.critical(
            None,
            "Missing Dependency",
            f"Missing package: {e}\nPlease install required packages with:\n  pip install librosa scipy soundfile matplotlib pyttsx3",
        )
        sys.exit(1)

    window = VoiceConversionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import librosa
import librosa.display

class SpectralPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax1 = self.fig.add_subplot(111)
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)
        
        # Store data for different plot types
        self.stored_data = {
            'freqs': None,
            'env_ref': None,
            'env_tts': None,
            'env_proc': None,
            'y_ref': None,
            'y_tts': None,
            'y_proc': None,
            'sr': None
        }
        
        self.current_plot_type = "envelopes"

    def store_audio_data(self, y_ref, y_tts, y_proc, sr):
        """Store audio waveform data for plotting"""
        self.stored_data['y_ref'] = y_ref
        self.stored_data['y_tts'] = y_tts
        self.stored_data['y_proc'] = y_proc
        self.stored_data['sr'] = sr

    def switch_plot_type(self, plot_type):
        """Switch between different plot types"""
        self.current_plot_type = plot_type
        if plot_type == "envelopes":
            self.plot_stored_envelopes()
        elif plot_type == "waveforms":
            self.plot_waveforms()
        elif plot_type == "spectrograms":
            self.plot_spectrograms()

    def plot_envelopes(self, freqs, env_ref, env_tts, env_proc):
        """Original envelope plotting method for compatibility"""
        # Store the data for later use
        self.stored_data['freqs'] = freqs
        self.stored_data['env_ref'] = env_ref
        self.stored_data['env_tts'] = env_tts
        self.stored_data['env_proc'] = env_proc
        
        self.ax1.clear()
        self.ax1.plot(freqs, 20 * np.log10(env_ref + 1e-12), "b-", label="Reference", linewidth=2)
        self.ax1.plot(freqs, 20 * np.log10(env_tts + 1e-12), "r-", label="TTS", linewidth=2)
        self.ax1.plot(freqs, 20 * np.log10(env_proc + 1e-12), "m-", label="Processed", linewidth=2)
        self.ax1.set_title("LPC Spectral Envelopes", fontsize=12, pad=10)
        self.ax1.set_xlabel("Frequency (Hz)")
        self.ax1.set_ylabel("Magnitude (dB)")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)
        self.draw()

    def plot_stored_envelopes(self):
        """Plot envelopes from stored data"""
        if (self.stored_data['freqs'] is None or self.stored_data['env_ref'] is None or 
            self.stored_data['env_tts'] is None or self.stored_data['env_proc'] is None):
            return
            
        # Clear the figure and recreate single subplot
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(111)
        
        freqs = self.stored_data['freqs']
        env_ref = self.stored_data['env_ref']
        env_tts = self.stored_data['env_tts']
        env_proc = self.stored_data['env_proc']
        
        self.ax1.plot(freqs, 20 * np.log10(env_ref + 1e-12), "b-", label="Reference", linewidth=2)
        self.ax1.plot(freqs, 20 * np.log10(env_tts + 1e-12), "r-", label="TTS", linewidth=2)
        self.ax1.plot(freqs, 20 * np.log10(env_proc + 1e-12), "m-", label="Processed", linewidth=2)
        self.ax1.set_title("LPC Spectral Envelopes", fontsize=12, pad=10)
        self.ax1.set_xlabel("Frequency (Hz)")
        self.ax1.set_ylabel("Magnitude (dB)")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)
        self.draw()

    def plot_waveforms(self):
        """Plot time domain waveforms"""
        if (self.stored_data['y_ref'] is None or self.stored_data['y_tts'] is None or 
            self.stored_data['y_proc'] is None or self.stored_data['sr'] is None):
            return
            
        # Clear the figure and recreate single subplot
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(111)
        
        y_ref = self.stored_data['y_ref']
        y_tts = self.stored_data['y_tts']
        y_proc = self.stored_data['y_proc']
        sr = self.stored_data['sr']
        
        # Limit to first 3 seconds for better visualization
        max_samples = min(len(y_ref), len(y_tts), len(y_proc), int(3 * sr))
        time_axis = np.linspace(0, max_samples / sr, max_samples)
        
        # Normalize for better comparison
        y_ref_norm = y_ref[:max_samples] / (np.max(np.abs(y_ref[:max_samples])) + 1e-8)
        y_tts_norm = y_tts[:max_samples] / (np.max(np.abs(y_tts[:max_samples])) + 1e-8)
        y_proc_norm = y_proc[:max_samples] / (np.max(np.abs(y_proc[:max_samples])) + 1e-8)
        
        # Offset for visualization
        self.ax1.plot(time_axis, y_ref_norm + 2, "b-", label="Reference", alpha=0.8, linewidth=1)
        self.ax1.plot(time_axis, y_tts_norm, "r-", label="TTS", alpha=0.8, linewidth=1)
        self.ax1.plot(time_axis, y_proc_norm - 2, "m-", label="Processed", alpha=0.8, linewidth=1)
        
        self.ax1.set_title("Time Domain Waveforms (First 3 seconds)", fontsize=12, pad=10)
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Normalized Amplitude")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(-3.5, 3.5)
        
        # Add horizontal lines to separate waveforms
        self.ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        self.ax1.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
        
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)
        self.draw()

    def plot_spectrograms(self):
        """Plot spectrograms"""
        if (self.stored_data['y_ref'] is None or self.stored_data['y_tts'] is None or 
            self.stored_data['y_proc'] is None or self.stored_data['sr'] is None):
            return
            
        # Clear the figure and create three subplots
        self.fig.clear()
        sr = self.stored_data['sr']
        
        try:
            ax1 = self.fig.add_subplot(311)
            ax2 = self.fig.add_subplot(312)
            ax3 = self.fig.add_subplot(313)
            
            # Parameters for spectrograms
            hop_length = 512
            n_fft = 2048
            
            # Reference spectrogram
            S_ref = librosa.stft(self.stored_data['y_ref'], hop_length=hop_length, n_fft=n_fft)
            S_ref_db = librosa.amplitude_to_db(np.abs(S_ref), ref=np.max)
            librosa.display.specshow(S_ref_db, y_axis='hz', x_axis='time', 
                                   sr=sr, hop_length=hop_length, ax=ax1, cmap='viridis')
            ax1.set_title('Reference Spectrogram', fontsize=10)
            ax1.set_ylabel('Frequency (Hz)')
            
            # TTS spectrogram
            S_tts = librosa.stft(self.stored_data['y_tts'], hop_length=hop_length, n_fft=n_fft)
            S_tts_db = librosa.amplitude_to_db(np.abs(S_tts), ref=np.max)
            librosa.display.specshow(S_tts_db, y_axis='hz', x_axis='time', 
                                   sr=sr, hop_length=hop_length, ax=ax2, cmap='viridis')
            ax2.set_title('TTS Spectrogram', fontsize=10)
            ax2.set_ylabel('Frequency (Hz)')
            
            # Processed spectrogram
            S_proc = librosa.stft(self.stored_data['y_proc'], hop_length=hop_length, n_fft=n_fft)
            S_proc_db = librosa.amplitude_to_db(np.abs(S_proc), ref=np.max)
            librosa.display.specshow(S_proc_db, y_axis='hz', x_axis='time', 
                                   sr=sr, hop_length=hop_length, ax=ax3, cmap='viridis')
            ax3.set_title('Processed Spectrogram', fontsize=10)
            ax3.set_ylabel('Frequency (Hz)')
            ax3.set_xlabel('Time (s)')
            
            self.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.4)
            
            # Reset ax1 to None since we're using multiple subplots
            self.ax1 = None
            self.draw()
            
        except Exception as e:
            # Fallback to single plot if there's an error
            self.fig.clear()
            self.ax1 = self.fig.add_subplot(111)
            self.ax1.text(0.5, 0.5, f'Error generating spectrograms:\n{str(e)}', 
                         transform=self.ax1.transAxes, ha='center', va='center')
            self.ax1.set_title("Spectrogram Error")
            self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)
            self.draw()
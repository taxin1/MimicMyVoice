from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class SpectralPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax1 = self.fig.add_subplot(111)
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)

    def plot_envelopes(self, freqs, env_ref, env_tts, env_proc):
        self.ax1.clear()
        self.ax1.plot(freqs, 20 * np.log10(env_ref + 1e-12), "b-", label="Reference", linewidth=2)
        self.ax1.plot(freqs, 20 * np.log10(env_tts + 1e-12), "r-", label="TTS", linewidth=2)
        self.ax1.plot(freqs, 20 * np.log10(env_proc + 1e-12), "m-", label="Processed", linewidth=2)
        self.ax1.set_title("LPC Spectral Envelopes", fontsize=12, pad=10)
        self.ax1.set_xlabel("Frequency (Hz)")
        self.ax1.set_ylabel("Magnitude (dB)")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        # Ensure proper layout and refresh
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)
        self.draw()

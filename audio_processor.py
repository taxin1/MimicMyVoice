from PyQt5.QtCore import QThread, pyqtSignal
import librosa
import numpy as np
import soundfile as sf
from audio_utils import extract_lpc, lpc_residual, resynthesize_from_residual, overlap_add

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

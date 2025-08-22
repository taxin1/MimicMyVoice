import numpy as np
import librosa
import soundfile as sf
import scipy.signal as sg
import tempfile
from scipy.ndimage import median_filter

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

def reduce_noise_spectral_subtraction(y, sr, n_fft=2048, hop_length=512, noise_factor=1.0):
    """
    Reduce noise using spectral subtraction
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    n_fft : int
        FFT window size
    hop_length : int
        Hop length for STFT
    noise_factor : float
        Factor to multiply the noise profile (higher = more aggressive)
    
    Returns:
    --------
    y_clean : numpy.ndarray
        Noise-reduced audio signal
    """
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # Compute magnitude spectrogram
    mag = np.abs(D)
    
    # Estimate noise profile from the first 0.5 seconds (assumed to be noise)
    noise_length = min(int(0.5 * sr / hop_length), mag.shape[1] // 4)
    noise_profile = np.mean(mag[:, :noise_length], axis=1).reshape(-1, 1)
    
    # Spectral subtraction
    mag_clean = np.maximum(mag - noise_factor * noise_profile, 0)
    
    # Recover phase
    phase = np.angle(D)
    
    # Reconstruct signal
    D_clean = mag_clean * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=hop_length)
    
    return y_clean

def reduce_noise_median_filter(y, sr, filter_size=3):
    """
    Reduce noise using median filtering in the spectral domain
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    filter_size : int
        Size of the median filter
    
    Returns:
    --------
    y_clean : numpy.ndarray
        Noise-reduced audio signal
    """
    # Compute STFT
    D = librosa.stft(y)
    
    # Apply median filter along frequency axis
    mag = np.abs(D)
    phase = np.angle(D)
    
    # Apply median filtering to magnitude
    mag_filtered = median_filter(mag, size=(filter_size, filter_size))
    
    # Reconstruct signal
    D_filtered = mag_filtered * np.exp(1j * phase)
    y_clean = librosa.istft(D_filtered)
    
    return y_clean

def reduce_noise(input_path, output_path=None, method='spectral_subtraction', **kwargs):
    """
    Reduce noise in an audio file and save the result
    
    Parameters:
    -----------
    input_path : str
        Path to input audio file
    output_path : str or None
        Path to save output audio file. If None, creates a temporary file.
    method : str
        Noise reduction method: 'spectral_subtraction' or 'median_filter'
    **kwargs : dict
        Additional parameters for the specific noise reduction method
    
    Returns:
    --------
    output_path : str
        Path to the noise-reduced audio file
    """
    # Load audio
    y, sr = librosa.load(input_path, sr=None)
    
    # Choose noise reduction method
    if method == 'spectral_subtraction':
        y_clean = reduce_noise_spectral_subtraction(y, sr, **kwargs)
    elif method == 'median_filter':
        y_clean = reduce_noise_median_filter(y, sr, **kwargs)
    else:
        raise ValueError(f"Unknown noise reduction method: {method}")
    
    # Create output path if not provided
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    
    # Save the result
    sf.write(output_path, y_clean, sr, subtype="PCM_16")
    
    return output_path

import os
import io
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_audio_file(audio_source):
    """
    Loads audio data from a file path, BytesIO, or uploaded file buffer.
    Returns (numpy_array_mono, sample_rate).
    Supports WAV natively, with fallback processing.
    """
    if isinstance(audio_source, (str, os.PathLike)):
        filepath = audio_source
        try:
            import soundfile as sf
            data, sr = sf.read(filepath)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), sr
        except Exception:
            # Native fallback using Python's standard wave library
            with wave.open(filepath, 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return data, sr

    elif hasattr(audio_source, "read"):
        # File object or UploadedFile
        audio_source.seek(0)
        file_bytes = audio_source.read()
        audio_source.seek(0)
        
        try:
            import soundfile as sf
            bio = io.BytesIO(file_bytes)
            data, sr = sf.read(bio)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), sr
        except Exception:
            # Fallback wave parsing
            try:
                bio = io.BytesIO(file_bytes)
                with wave.open(bio, 'rb') as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    frames = wf.readframes(n_frames)
                    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return data, sr
            except Exception as e:
                raise ValueError(f"Could not decode audio file: {e}")
    else:
        raise TypeError("Unsupported audio source type.")

def generate_cyber_plots(data, sr):
    """
    Renders dual interactive cybersecurity forensic plots:
    1. Acoustic Waveform (Amplitude vs Time) with cyan glowing aesthetics
    2. Spectrogram Heatmap (Frequency vs Time) illustrating vocoder brickwall cutoffs
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={'height_ratios': [1, 1.3]})
    fig.patch.set_facecolor('#0d1117')
    ax1.set_facecolor('#161b22')
    ax2.set_facecolor('#161b22')

    num_samples = len(data)
    duration = num_samples / float(sr)
    time_axis = np.linspace(0, duration, num_samples)

    # 1. Waveform Plot
    ax1.plot(time_axis, data, color='#00f2fe', linewidth=0.8, alpha=0.9, label='Amplitude')
    ax1.fill_between(time_axis, data, 0, color='#00f2fe', alpha=0.15)
    ax1.set_title('Forensic Acoustic Waveform (Amplitude vs. Time)', color='#f0f6fc', fontsize=11, fontweight='bold', pad=8)
    ax1.set_ylabel('Amplitude', color='#8b949e', fontsize=9)
    ax1.set_xlim(0, duration)
    ax1.set_ylim(-1.0, 1.0)
    ax1.grid(True, color='#30363d', linestyle='--', alpha=0.5)
    ax1.tick_params(colors='#8b949e', labelsize=8)

    # 2. Spectrogram Plot
    nperseg = min(512, len(data))
    noverlap = nperseg // 2
    step = nperseg - noverlap
    shape = ((len(data) - nperseg) // step + 1, nperseg)
    strides = (data.strides[0] * step, data.strides[0])
    try:
        frames = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    except Exception:
        frames = np.array([data[i:i+nperseg] for i in range(0, len(data)-nperseg+1, step)])

    window = np.hanning(nperseg)
    windowed_frames = frames * window
    fft_vals = np.abs(np.fft.rfft(windowed_frames, axis=1))
    spectrogram_db = 20 * np.log10(fft_vals + 1e-6)
    
    freq_bins = np.fft.rfftfreq(nperseg, 1.0 / sr)
    spec_duration = (len(frames) * step) / sr
    
    im = ax2.imshow(
        spectrogram_db.T,
        aspect='auto',
        origin='lower',
        extent=[0, spec_duration, freq_bins[0], freq_bins[-1]],
        cmap='magma',
        vmin=np.max(spectrogram_db) - 60,
        vmax=np.max(spectrogram_db)
    )
    
    ax2.set_title('High-Resolution Spectral Intensity & Neural Vocoder Artifact Analysis', color='#f0f6fc', fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlabel('Time (seconds)', color='#8b949e', fontsize=9)
    ax2.set_ylabel('Frequency (Hz)', color='#8b949e', fontsize=9)
    ax2.grid(True, color='#30363d', linestyle=':', alpha=0.4)
    ax2.tick_params(colors='#8b949e', labelsize=8)
    
    if sr >= 16000:
        ax2.axhline(y=7800, color='#ff4b4b', linestyle='--', linewidth=1, label='Vocoder Cutoff Threshold (7.8kHz)')
        ax2.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', fontsize=8, labelcolor='#f0f6fc')

    cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.25, shrink=0.7)
    cbar.set_label('Spectral Density (dB)', color='#8b949e', fontsize=8)
    cbar.ax.tick_params(labelsize=7, colors='#8b949e')

    plt.tight_layout()
    return fig

def extract_acoustic_forensic_metrics(data, sr):
    """
    Extracts acoustic forensic indicators using fast NumPy vectorized operations.
    Handles live microphone recordings and pre-recorded audio clips with high precision.
    """
    if data is None or len(data) == 0:
        return {
            "hf_ratio": 0.05,
            "zcr_std": 0.05,
            "energy_std": 0.1,
            "dynamic_range_db": 20.0,
            "phase_variance": 1.0,
            "spectral_flatness": 0.1
        }

    sr = sr or 16000
    # 1. Vectorized High-Frequency Energy Ratio
    fft_spectrum = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
    
    total_energy = np.sum(fft_spectrum**2) + 1e-9
    high_freq_mask = freqs > (0.45 * sr)
    high_freq_energy = np.sum(fft_spectrum[high_freq_mask]**2)
    hf_ratio = float(high_freq_energy / total_energy)

    # 2. Vectorized Frame ZCR, Energy & Pitch Jitter Extraction
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    num_frames = (len(data) - frame_len) // hop_len

    if num_frames > 1:
        if num_frames > 2000:
            step_stride = num_frames // 2000
            frame_indices = np.arange(0, num_frames, step_stride) * hop_len
        else:
            frame_indices = np.arange(num_frames) * hop_len
        
        frames = np.array([data[idx : idx + frame_len] for idx in frame_indices])
        
        diff_signs = np.abs(np.diff(np.signbit(frames), axis=1))
        zcrs = np.mean(diff_signs, axis=1)
        energies = np.sum(frames**2, axis=1)

        zcr_std = float(np.std(zcrs))
        energy_std = float(np.std(energies))

        non_zero_energies = energies[energies > 1e-7]
        if len(non_zero_energies) > 0:
            min_energy = np.min(non_zero_energies)
            max_energy = np.max(non_zero_energies) + 1e-6
            dynamic_range_db = float(10 * np.log10(max_energy / min_energy))
        else:
            dynamic_range_db = 20.0
    else:
        zcr_std = 0.05
        energy_std = 0.1
        dynamic_range_db = 20.0

    # 3. Phase Variance on representative sub-sample
    sub_data = data[:min(len(data), sr * 10)]
    phase_spectrum = np.angle(np.fft.rfft(sub_data))
    phase_diff = np.diff(phase_spectrum)
    phase_variance = float(np.var(phase_diff))

    # 4. Spectral Flatness
    psd = fft_spectrum**2 + 1e-12
    geo_mean = np.exp(np.mean(np.log(psd)))
    ari_mean = np.mean(psd)
    spectral_flatness = float(geo_mean / ari_mean)

    return {
        "hf_ratio": hf_ratio,
        "zcr_std": zcr_std,
        "energy_std": energy_std,
        "dynamic_range_db": dynamic_range_db,
        "phase_variance": phase_variance,
        "spectral_flatness": spectral_flatness
    }

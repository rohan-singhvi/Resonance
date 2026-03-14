import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import argparse
from scipy.signal import spectrogram
from acoustic_metrics import schroeder_integration, compute_rt60

def load_wav(path):
    try:
        data, sr = sf.read(path)
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data, sr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

def plot_comparison(dry_path, wet_path, ir_path=None):
    print(f"Comparing '{dry_path}' vs '{wet_path}'...")
    
    dry, sr1 = load_wav(dry_path)
    wet, sr2 = load_wav(wet_path)
    
    if dry is None or wet is None: return

    # Ensure lengths match for plotting (trim to shorter)
    min_len = min(len(dry), len(wet))
    dry = dry[:min_len]
    wet = wet[:min_len]
    
    # Time axis
    time = np.linspace(0, min_len / sr1, min_len)

    n_plots = 4 if ir_path else 3
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=False)
    plt.subplots_adjust(hspace=0.4)

    # 1. Waveform Comparison
    axs[0].set_title("Time Domain: Amplitude Decay")
    axs[0].plot(time, wet, label='Wet (Reverb)', color='dodgerblue', alpha=0.7)
    axs[0].plot(time, dry, label='Dry (Source)', color='orange', alpha=0.6, linestyle='--')
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True, alpha=0.3)

    # 2. Wet Spectrogram (Frequency Content)
    axs[1].set_title("Wet Signal Spectrogram (Frequency Decay)")
    Pxx, freqs, bins, im = axs[1].specgram(wet, NFFT=1024, Fs=sr1, noverlap=512, cmap='inferno')
    axs[1].set_ylabel("Frequency (Hz)")
    
    # 3. Energy Decay (dB)
    # Calculate simple envelope
    window_size = int(sr1 * 0.01) # 10ms window
    dry_env = np.convolve(dry**2, np.ones(window_size)/window_size, mode='same')
    wet_env = np.convolve(wet**2, np.ones(window_size)/window_size, mode='same')
    
    # Avoid log(0)
    dry_db = 10 * np.log10(dry_env + 1e-12)
    wet_db = 10 * np.log10(wet_env + 1e-12)
    
    axs[2].set_title("Energy Decay Curve (dB)")
    axs[2].plot(time, wet_db, label='Wet Energy', color='dodgerblue')
    axs[2].plot(time, dry_db, label='Dry Energy', color='orange', linestyle='--')
    axs[2].set_ylabel("Power (dB)")
    axs[2].set_ylim(-60, 0) # Focus on top 60dB
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)
    axs[2].set_xlabel("Time (s)")

    if ir_path:
        ir, ir_sr = load_wav(ir_path)
        if ir is not None:
            curve = schroeder_integration(ir, ir_sr)
            rt60 = compute_rt60(ir, ir_sr)
            ir_time = np.arange(len(ir)) / ir_sr

            axs[3].plot(ir_time, curve, color="seagreen", label="Schroeder curve")
            axs[3].set_ylim(-70, 5)
            axs[3].set_ylabel("Level (dB)")
            axs[3].set_xlabel("Time (s)")
            axs[3].set_title("Schroeder Integration")
            axs[3].grid(True, alpha=0.3)

            if rt60 is not None:
                axs[3].axvline(x=rt60, color="red", linestyle="--", label=f"RT60 = {rt60:.2f} s")
                axs[3].axhline(y=-60, color="gray", linestyle=":", alpha=0.5)

            axs[3].legend(loc="upper right")

    output_img = "acoustic_analysis.png"
    plt.savefig(output_img)
    print(f"Analysis saved to: {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dry", help="Path to dry wav")
    parser.add_argument("wet", help="Path to wet wav")
    parser.add_argument("--ir", help="Path to room impulse response wav (enables Schroeder panel)")
    args = parser.parse_args()

    plot_comparison(args.dry, args.wet, ir_path=args.ir)
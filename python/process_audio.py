import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
from acoustic_metrics import print_metrics

# Check if we are in the Docker Simulator or Real GPU
IS_SIMULATION = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

if IS_SIMULATION:
    import numpy as xp
    from scipy.signal import fftconvolve

    print("CPU MODE: Using Scipy for convolution")
else:
    # On real GPU, we use Cupy's signal library
    import cupy as xp
    from cupyx.scipy.signal import fftconvolve

    print("GPU MODE: Using CUDA FFT for convolution")


def load_audio(filepath):
    """Loads audio and ensures it is float32."""
    try:
        data, samplerate = sf.read(filepath)
        return data.astype(np.float32), samplerate
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def apply_convolution(dry_signal, ir):
    """
    Performs the heavy math: Dry Audio * Room Impulse.
    Handles Single Channel (Mono) or Dual Channel (Stereo).
    """
    # move data to device (GPU) or stay on host (CPU)
    d_dry = xp.asarray(dry_signal)
    d_ir = xp.asarray(ir)

    # Input is Stereo (N, 2) and IR is Mono (N,)
    if d_dry.ndim == 2 and d_ir.ndim == 1:
        print("Input is Stereo. Processing channels independently...")
        left = d_dry[:, 0]
        right = d_dry[:, 1]

        # convolve both channels with the room IR
        # mode='full' gives the full reverb tail
        out_left = fftconvolve(left, d_ir, mode="full")
        out_right = fftconvolve(right, d_ir, mode="full")

        # stack them back together
        d_wet = xp.column_stack((out_left, out_right))

    else:
        # mono processing
        print("Input is Mono.")
        d_wet = fftconvolve(d_dry, d_ir, mode="full")

    # bring result back to CPU (if on GPU)
    if not IS_SIMULATION:
        wet_signal = xp.asnumpy(d_wet)
    else:
        wet_signal = d_wet

    return wet_signal


def main():
    parser = argparse.ArgumentParser(
        description="Apply GPU Acoustic Simulation to any audio file."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to your source audio (wav/flac)"
    )
    parser.add_argument(
        "--ir",
        type=str,
        default="room_impulse.wav",
        help="Path to the generated room impulse",
    )
    parser.add_argument(
        "--mix",
        type=float,
        default=0.4,
        help="Wet/Dry mix (0.0 = Dry, 1.0 = All Reverb)",
    )
    parser.add_argument(
        "--output", type=str, default="processed_output", help="Name of Output File"
    )
    args = parser.parse_args()

    if not os.path.exists(args.ir):
        print("Error: Room Impulse not found. Run 'acoustic_simulator.py' first!")
        return

    print(f"Loading input: {args.input_file}")
    dry_audio, dry_sr = load_audio(args.input_file)

    print(f"Loading Room IR: {args.ir}")
    ir_audio, ir_sr = load_audio(args.ir)

    # resample safety check
    if dry_sr != ir_sr:
        print(
            f"Warning: Sample rate mismatch! Input: {dry_sr}, IR: {ir_sr}. This might sound weird."
        )
        # (in a production app, we would auto-resample here)

    start_time = time.time()
    print("Convolving...")

    wet_audio = apply_convolution(dry_audio, ir_audio)

    process_time = time.time() - start_time
    print(f"Processed in {process_time:.2f} seconds")

    # convolution makes things loud - normalize the wet signal first.
    wet_max = np.max(np.abs(wet_audio))
    if wet_max > 0:
        wet_audio = wet_audio / wet_max
    else:
        print("Warning: Reverb signal is silent!")

    # The 'wet_audio' is longer than the 'dry_audio' because of the reverb tail.
    # We need to pad the dry audio to match lengths to mix them.
    target_len = wet_audio.shape[0]
    padded_dry = np.zeros_like(wet_audio)

    # copy original dry audio into padded buffer
    if dry_audio.ndim == 2:
        padded_dry[: dry_audio.shape[0], :] = dry_audio
    else:
        padded_dry[: dry_audio.shape[0]] = dry_audio

    # blend = (Dry * (1 - mix)) + (Wet * mix)
    final_mix = (padded_dry * (1.0 - args.mix)) + (wet_audio * args.mix)

    # safety clamp to prevent clipping
    max_val = np.max(np.abs(final_mix))
    if max_val > 1.0:
        final_mix /= max_val

    output_filename = f"{args.output}.wav"
    sf.write(output_filename, final_mix, dry_sr)
    print(f"Saved to '{output_filename}'")

    try:
        print_metrics(ir_audio, ir_sr)
    except Exception as e:
        print(f"Metrics computation failed: {e}")


if __name__ == "__main__":
    main()

"""
Short-time Fourier transform: framing, windowing, and overlap-add synthesis.

This is the layer that turns a one-shot transform into something that can
process a whole song. It depends on fft.py and nothing else.
"""

import math

from fft import fft, ifft, is_power_of_two


def hann_window(n):
    # Periodic Hann window of length n.
    # Dividing by n rather than n - 1 is what makes it periodic; the
    # symmetric variant does not satisfy the overlap-add condition exactly.
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / n)) for i in range(n)]


def frame_signal(samples, frame_size, hop_size):
    """Split samples into overlapping frames.

    The final frame is zero-padded if the signal does not divide evenly.

    Returns:
        list of frames, each a list of length frame_size.
    """
    frames = []
    position = 0
    while position < len(samples):
        frame = list(samples[position:position + frame_size])
        frame.extend([0.0] * (frame_size - len(frame)))
        frames.append(frame)
        position += hop_size
    return frames


def overlap_add(frames, hop_size, output_length):
    # Reconstruct a signal by summing overlapping frames back together.
    # Correct reconstruction depends on the window and hop size satisfying
    # the COLA condition. Hann at 50% or 75% overlap does; other hop sizes
    # will produce amplitude modulation.
    output = [0.0] * output_length
    for index, frame in enumerate(frames):
        start = index * hop_size
        for offset, value in enumerate(frame):
            target = start + offset
            if target >= output_length:
                break
            output[target] += value
    return output


def process(samples, sample_rate, gain_curve_fn, frame_size=2048,
            hop_size=None, progress_fn=None):
    """Run the full analysis-modify-synthesis loop over a signal.

    For each frame: window it, FFT it, multiply by the gain curve,
    inverse FFT, and accumulate into the output via overlap-add.

    The window is applied twice, once before the transform and once after.
    Windowing on the way out suppresses the discontinuities that spectral
    modification introduces at frame edges; dividing by the accumulated
    sum of squared windows then restores unity gain. With a flat curve this
    reproduces the input to floating-point precision.

    Args:
        samples: input signal as a list of floats.
        sample_rate: samples per second.
        gain_curve_fn: callable (n, sample_rate) -> list of n multipliers.
        frame_size: FFT size. Power of 2.
        hop_size: samples between consecutive frames. Defaults to 75% overlap.
        progress_fn: optional callable (frames_done, frames_total).

    Returns:
        list of floats, same length as samples.
    """
    if not is_power_of_two(frame_size):
        raise ValueError(f"frame_size must be a power of 2, got {frame_size}")
    if hop_size is None:
        hop_size = frame_size // 4
    if hop_size <= 0 or hop_size > frame_size:
        raise ValueError(f"hop_size must be in 1..{frame_size}, got {hop_size}")

    window = hann_window(frame_size)
    gain_curve = gain_curve_fn(frame_size, sample_rate)

    # Pad both ends so that every original sample sits under a full set of
    # overlapping windows. Without this the first and last frame_size samples
    # would be attenuated by the window's taper.
    original_length = len(samples)
    padded = [0.0] * frame_size + list(samples) + [0.0] * (2 * frame_size)

    accumulated = [0.0] * len(padded)
    window_energy = [0.0] * len(padded)

    positions = range(0, len(padded) - frame_size + 1, hop_size)
    total_frames = len(positions)

    for frame_index, position in enumerate(positions):
        frame = [padded[position + i] * window[i] for i in range(frame_size)]

        spectrum = fft(frame)
        for i in range(frame_size):
            spectrum[i] *= gain_curve[i]
        restored = ifft(spectrum)

        for i in range(frame_size):
            accumulated[position + i] += restored[i].real * window[i]
            window_energy[position + i] += window[i] * window[i]

        if progress_fn is not None and frame_index % 16 == 0:
            progress_fn(frame_index, total_frames)

    if progress_fn is not None:
        progress_fn(total_frames, total_frames)

    output = []
    for i in range(frame_size, frame_size + original_length):
        energy = window_energy[i]
        output.append(accumulated[i] / energy if energy > 1e-8 else 0.0)
    return output


def cola_sum(window, hop_size, num_frames=16):
    # Sum of the squared window shifted by hop_size, sampled in the middle
    # where coverage is complete. Used by the tests to check the COLA
    # condition holds for a given window and hop.
    frame_size = len(window)
    total_length = frame_size * (num_frames + 2)
    accumulator = [0.0] * total_length
    position = 0
    while position + frame_size <= total_length:
        for i in range(frame_size):
            accumulator[position + i] += window[i] * window[i]
        position += hop_size
    return accumulator[frame_size * 2:frame_size * num_frames]

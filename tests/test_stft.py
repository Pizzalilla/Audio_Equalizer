"""
Tests for framing, windowing, and overlap-add reconstruction.
"""

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from filters import graphic_eq_curve
from stft import analyse, cola_sum, hann_window, process, synthesise


def music_like_signal(n, sample_rate=8000, seed=0):
    # A few tones plus noise: enough spectral content to exercise the bands.
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = i / sample_rate
        value = (0.4 * math.sin(2 * math.pi * 110 * t)
                 + 0.3 * math.sin(2 * math.pi * 880 * t)
                 + 0.2 * math.sin(2 * math.pi * 3000 * t)
                 + 0.05 * rng.uniform(-1, 1))
        out.append(value)
    return out


def test_hann_window_shape():
    window = hann_window(8)
    assert window[0] == pytest.approx(0.0)
    assert window[4] == pytest.approx(1.0)
    # Periodic, so the window is symmetric about its peak but does not
    # return to zero at the final sample.
    assert window[1] == pytest.approx(window[7])
    assert window[3] == pytest.approx(window[5])


def test_cola_hann_75_percent():
    # process() windows twice, once before the transform and once after, so
    # the quantity that has to come out constant is the sum of SQUARED
    # windows, not the sum of windows. That distinction matters: the usual
    # "Hann is COLA at 50% overlap" result is about the unsquared window.
    window = hann_window(64)

    at_75 = cola_sum(window, 16)
    assert max(at_75) - min(at_75) < 1e-9
    assert at_75[0] == pytest.approx(1.5)

    # Squared Hann at 50% is not constant, because
    # sin^4 + cos^4 = 1 - sin^2(2x)/2, which swings between 0.5 and 1.
    at_50 = cola_sum(window, 32)
    assert max(at_50) == pytest.approx(1.0)
    assert min(at_50) == pytest.approx(0.5)

    # A hop that does not divide the window evenly is worse still.
    ragged = cola_sum(window, 25)
    assert max(ragged) - min(ragged) > 1e-3


def test_reconstruction_survives_a_non_cola_hop():
    # process() divides by the accumulated window energy rather than relying
    # on the hop being COLA-compliant, so even 50% overlap reconstructs
    # exactly. Without that division it would pulse at the frame rate.
    signal = music_like_signal(4000, seed=7)
    flat = {"bass": 0.0, "mid": 0.0, "treble": 0.0}
    for hop in [128, 64, 32]:
        output = process(signal, 8000,
                         lambda n, sr: graphic_eq_curve(n, sr, flat),
                         frame_size=256, hop_size=hop)
        for original, restored in zip(signal, output):
            assert abs(original - restored) < 1e-9, f"hop {hop}"


def test_analyse_applies_window_and_transform():
    # A frame of constant 1.0 windowed by Hann has DC equal to the sum of
    # the window, which for a periodic Hann is exactly n / 2.
    signal = [1.0] * 128
    window = hann_window(64)
    spectrum = analyse(signal, 0, window)
    assert len(spectrum) == 64
    assert spectrum[0].real == pytest.approx(32.0)
    assert spectrum[0].imag == pytest.approx(0.0)


def test_analyse_reads_from_the_given_offset():
    signal = [0.0] * 64 + [1.0] * 64
    window = hann_window(64)
    assert abs(analyse(signal, 0, window)[0]) == pytest.approx(0.0)
    assert analyse(signal, 64, window)[0].real == pytest.approx(32.0)


def test_synthesise_is_windowed_identity_under_flat_gain():
    # With a flat curve, synthesise undoes analyse except for the second
    # taper, so the result is the input multiplied by the window squared.
    signal = music_like_signal(128)
    window = hann_window(64)
    spectrum = analyse(signal, 0, window)
    frame = synthesise(spectrum, [1.0] * 64, window)
    for i in range(64):
        assert frame[i] == pytest.approx(signal[i] * window[i] ** 2, abs=1e-12)


def test_synthesise_returns_real_values():
    signal = music_like_signal(128, seed=9)
    window = hann_window(64)
    spectrum = analyse(signal, 0, window)
    curve = graphic_eq_curve(64, 8000, {"bass": 6.0, "treble": -6.0})
    assert all(isinstance(v, float) for v in synthesise(spectrum, curve, window))


def test_flat_eq_is_identity():
    # With all gains at 0 dB, output should match input. End-to-end version
    # of the Parseval check: if the whole chain is correct, flat EQ is a no-op.
    signal = music_like_signal(4000)
    flat = {"bass": 0.0, "mid": 0.0, "treble": 0.0}
    output = process(
        signal, 8000,
        lambda n, sr: graphic_eq_curve(n, sr, flat),
        frame_size=256,
    )
    assert len(output) == len(signal)
    for original, restored in zip(signal, output):
        assert abs(original - restored) < 1e-9


def test_energy_is_preserved_under_flat_eq():
    signal = music_like_signal(4000, seed=1)
    flat = {"bass": 0.0, "mid": 0.0, "treble": 0.0}
    output = process(signal, 8000, lambda n, sr: graphic_eq_curve(n, sr, flat),
                     frame_size=256)
    before = sum(v * v for v in signal)
    after = sum(v * v for v in output)
    assert after == pytest.approx(before, rel=1e-9)


def test_boost_and_cut_change_energy_in_the_right_direction():
    signal = music_like_signal(4000, seed=2)
    baseline = sum(v * v for v in signal)

    boosted = process(signal, 8000,
                      lambda n, sr: graphic_eq_curve(n, sr, {"bass": 12.0}),
                      frame_size=256)
    assert sum(v * v for v in boosted) > baseline

    cut = process(signal, 8000,
                  lambda n, sr: graphic_eq_curve(n, sr, {"bass": -12.0}),
                  frame_size=256)
    assert sum(v * v for v in cut) < baseline


def test_output_stays_real():
    # The gain curve is symmetric, so the inverse transform has no
    # meaningful imaginary part to discard.
    signal = music_like_signal(2000, seed=3)
    output = process(signal, 8000,
                     lambda n, sr: graphic_eq_curve(n, sr, {"treble": 9.0}),
                     frame_size=256)
    assert all(isinstance(v, float) for v in output)
    assert all(abs(v) < 10.0 for v in output)


def test_rejects_bad_frame_size():
    with pytest.raises(ValueError):
        process([0.0] * 100, 8000, lambda n, sr: [1.0] * n, frame_size=300)

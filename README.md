# Audio Equalizer

A graphic equaliser and spectrum analyser built on a from-scratch implementation
of the Fast Fourier Transform. No FFT library is used — the transform is written
directly from the Cooley-Tukey recurrence.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

## Tests

```bash
python -m pytest tests/ -v
```

## Layout

```
src/
  fft.py        Cooley-Tukey FFT, inverse FFT, naive DFT reference
  filters.py    Per-bin gain curves for the equaliser bands
  stft.py       Framing, Hann windowing, overlap-add synthesis
  audio_io.py   WAV reading and writing
  plots.py      Waveform, spectrum, and spectrogram figures
app.py          Streamlit interface
tests/          Correctness tests
```

Dependencies point one way only: `fft.py` knows nothing about audio, and the
audio layer knows nothing about the interface.

## How it works

1. Read the WAV file and convert samples to floats
2. Split the signal into overlapping frames and apply a Hann window
3. Transform each frame to the frequency domain with the FFT
4. Multiply each frequency bin by the gain for its band
5. Transform back with the inverse FFT
6. Reassemble the frames with overlap-add
7. Write the result to a new WAV file

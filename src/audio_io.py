"""
WAV reading and writing, using only the standard library `wave` module.

Samples are handled internally as floats in [-1.0, 1.0].
"""

import struct
import wave


def _decode(raw, sample_width, count):
    # Turn packed bytes into signed integers. 8-bit WAV is the odd one out:
    # it is stored unsigned with an offset of 128.
    if sample_width == 1:
        return [b - 128 for b in raw[:count]]
    if sample_width == 2:
        return list(struct.unpack(f"<{count}h", raw[:count * 2]))
    if sample_width == 4:
        return list(struct.unpack(f"<{count}i", raw[:count * 4]))
    if sample_width == 3:
        return [
            int.from_bytes(raw[i:i + 3], "little", signed=True)
            for i in range(0, count * 3, 3)
        ]
    raise ValueError(f"unsupported sample width: {sample_width} bytes")


def _encode(values, sample_width):
    # Inverse of _decode.
    if sample_width == 1:
        return bytes((v + 128) & 0xFF for v in values)
    if sample_width == 2:
        return struct.pack(f"<{len(values)}h", *values)
    if sample_width == 4:
        return struct.pack(f"<{len(values)}i", *values)
    if sample_width == 3:
        out = bytearray()
        for v in values:
            out.extend(v.to_bytes(3, "little", signed=True))
        return bytes(out)
    raise ValueError(f"unsupported sample width: {sample_width} bytes")


def read_wav(source):
    """Read a WAV file.

    Args:
        source: a path, or any file-like object `wave` can open.

    Returns:
        (channels, sample_rate) where channels is a list of channels, each
        a list of floats in [-1, 1].
    """
    with wave.open(source, "rb") as wf:
        channel_count = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    total = frame_count * channel_count
    values = _decode(raw, sample_width, total)
    full_scale = float(1 << (sample_width * 8 - 1))

    channels = [[] for _ in range(channel_count)]
    for index, value in enumerate(values):
        channels[index % channel_count].append(value / full_scale)
    return channels, sample_rate


def write_wav(destination, channels, sample_rate, sample_width=2):
    # Write channels out as a WAV file. Values outside [-1, 1] are clipped.
    if not channels:
        raise ValueError("no channels to write")

    peak = float(1 << (sample_width * 8 - 1))
    limit = int(peak) - 1
    frame_count = len(channels[0])

    interleaved = []
    for frame in range(frame_count):
        for channel in channels:
            scaled = int(max(-1.0, min(1.0, channel[frame])) * peak)
            interleaved.append(max(-limit - 1, min(limit, scaled)))

    with wave.open(destination, "wb") as wf:
        wf.setnchannels(len(channels))
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(_encode(interleaved, sample_width))


def to_mono(channels):
    # Average a multi-channel signal down to a single channel.
    if len(channels) == 1:
        return list(channels[0])
    count = len(channels)
    return [sum(frame) / count for frame in zip(*channels)]

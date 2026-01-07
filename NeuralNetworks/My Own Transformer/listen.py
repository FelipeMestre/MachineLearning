#!/usr/bin/env python3
"""
Listen to a MIDI file using a simple piano-like synth.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pretty_midi


def notes_to_frequencies(notes):
    return 2 ** ((np.array(notes) - 69) / 12) * 440


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    max_val = np.max(np.abs(audio)) if audio.size else 0
    if max_val > 1.0:
        audio = audio / max_val
    return audio


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = _normalize_audio(audio)
    if audio.ndim == 1:
        channels = 1
        data = audio
    elif audio.ndim == 2 and audio.shape[1] in (1, 2):
        channels = audio.shape[1]
        data = audio.reshape(-1)
    else:
        raise ValueError("Unsupported audio shape for WAV export.")

    pcm = (data * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _play_audio(audio: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice as sd

        sd.play(audio, sample_rate)
        sd.wait()
        return
    except Exception:
        pass

    try:
        import simpleaudio as sa

        audio = _normalize_audio(audio)
        if audio.ndim == 1:
            channels = 1
            data = audio
        else:
            channels = audio.shape[1]
            data = audio.reshape(-1)
        pcm = (data * 32767).astype(np.int16)
        sa.play_buffer(pcm, channels, 2, sample_rate).wait_done()
        return
    except Exception:
        pass

    wav_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    _write_wav(wav_path, audio, sample_rate)

    if sys.platform == "darwin":
        cmd = ["afplay", str(wav_path)]
    elif sys.platform.startswith("linux"):
        cmd = ["aplay", str(wav_path)]
    else:
        cmd = ["ffplay", "-autoexit", "-nodisp", str(wav_path)]

    try:
        subprocess.run(cmd, check=True)
    except Exception:
        print(f"No audio backend available. WAV saved to: {wav_path}")


def _synthesize_note(frequency: float, duration: float, velocity: int, sample_rate: int) -> np.ndarray:
    n_samples = max(1, int(duration * sample_rate))
    time = np.linspace(0, duration, n_samples, endpoint=False).astype(np.float32)

    wave = (
        np.sin(2 * np.pi * frequency * time)
        + 0.5 * np.sin(2 * np.pi * 2 * frequency * time)
        + 0.25 * np.sin(2 * np.pi * 3 * frequency * time)
    )

    attack_samples = min(n_samples, max(1, int(0.01 * sample_rate)))
    release_samples = min(n_samples, max(1, int(0.02 * sample_rate)))
    envelope = np.ones(n_samples, dtype=np.float32)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples, dtype=np.float32)
    if release_samples > 1:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples, dtype=np.float32)

    decay = np.exp(-3.0 * time / max(duration, 1e-6)).astype(np.float32)
    wave = wave * envelope * decay

    return wave * (velocity / 127.0)


def midi_to_audio(midi_data: pretty_midi.PrettyMIDI, sample_rate: int = 44100) -> np.ndarray:
    total_samples = int(midi_data.get_end_time() * sample_rate) + 1
    audio = np.zeros(total_samples, dtype=np.float32)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start = int(note.start * sample_rate)
            end = int(note.end * sample_rate)
            if end <= start or start >= total_samples:
                continue
            end = min(end, total_samples)
            duration = (end - start) / sample_rate
            frequency = float(notes_to_frequencies(note.pitch))
            note_wave = _synthesize_note(frequency, duration, note.velocity, sample_rate)
            audio[start:start + note_wave.size] += note_wave[: end - start]

    return audio


def listen_midi(midi_path: str | Path, sample_rate: int = 44100, amplitude: float = 0.2) -> None:
    """
    Load a MIDI file and play it using a simple piano-like synth.
    """
    midi_path = Path(midi_path).expanduser()
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    audio = midi_to_audio(midi_data, sample_rate=sample_rate)
    audio = _normalize_audio(audio) * amplitude
    _play_audio(audio, sample_rate)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen to a MIDI file with a simple piano-like synth.")
    parser.add_argument("midi_path", help="Path to the .midi file")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--amplitude", type=float, default=0.2, help="Playback amplitude (0-1)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    listen_midi(args.midi_path, sample_rate=args.sample_rate, amplitude=args.amplitude)


if __name__ == "__main__":
    main()

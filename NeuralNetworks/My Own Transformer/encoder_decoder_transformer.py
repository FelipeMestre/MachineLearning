#!/usr/bin/env python3
"""
Minimal test harness for MIDI playback.
"""

from pathlib import Path

from listen import listen_midi


def test_listen_first_midi() -> None:
    datasets_dir = Path(__file__).resolve().parent / "datasets" / "maestro-v3.0.0"
    if not datasets_dir.exists():
        print("Dataset not found. Please download it first.")
        return

    midi_files = sorted(datasets_dir.rglob("*.midi"))
    if not midi_files:
        print("No MIDI files found in dataset.")
        return

    listen_midi(midi_files[0])


if __name__ == "__main__":
    test_listen_first_midi()

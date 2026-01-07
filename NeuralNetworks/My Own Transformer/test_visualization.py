#!/usr/bin/env python3
"""
Simple script to test MIDI visualization functionality.
Run this in the NeuralNetworks conda environment.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import from the main script
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import our visualization functions
from midi_visualization import plot_piano_roll_beats, load_midi_file

def main():
    # Dataset path (maintainable and portable)
    datasets_dir = Path(__file__).resolve().parent / "datasets" / "maestro-v3.0.0"

    if not datasets_dir.exists():
        print("Dataset not found. Please run the download script first.")
        return

    # Find the first MIDI file (sorted for determinism)
    midi_files = sorted(datasets_dir.rglob("*.midi"))
    if not midi_files:
        print("No MIDI files found in dataset")
        return

    test_file = midi_files[0]
    midi_data = load_midi_file(test_file)
    if not midi_data:
        print("Failed to load MIDI file")
        return
    try:
        plot_piano_roll_beats(midi_data, title=f"Piano Roll (Beats) - {test_file.name}")
        print("Beat-based visualization created successfully!")
    except Exception as e:
        print(f"Error creating beat visualization: {e}")

if __name__ == "__main__":
    main()
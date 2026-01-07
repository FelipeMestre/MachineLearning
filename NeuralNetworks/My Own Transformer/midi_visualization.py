#!/usr/bin/env python3
"""
MIDI visualization utilities for the MAESTRO dataset.
"""
import pretty_midi

import matplotlib.pyplot as plt
import numpy as np


def plot_piano_roll(midi_data, title="Piano Roll", figsize=(12, 8)):
    """
    Creates a piano roll visualization (note vs time grid) from MIDI data.

    Args:
        midi_data (pretty_midi.PrettyMIDI): The MIDI data object
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
    """
    if midi_data is None:
        print("No MIDI data to plot")
        return

    # Get piano roll (notes over time)
    # Piano roll is a 2D array where rows are MIDI notes (0-127) and columns are time steps
    piano_roll = midi_data.get_piano_roll(fs=50)  # Reducido de 100 a 50 Hz para mejor visualización

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot piano roll using imshow
    # Transpose so time is on x-axis, notes on y-axis
    duration = piano_roll.shape[1] / 50  # duración total en segundos
    im = ax.imshow(piano_roll.T, aspect='auto', origin='lower',
                   cmap='viridis', interpolation='nearest',
                   extent=[0, duration, 0, 127])

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MIDI Note Number')
    ax.set_title(title)
    ax.set_ylim(21, 108)  # Rango de piano estándar

    # Set x-axis ticks to show time in seconds
    if duration > 60:  # Si es mayor a 1 minuto
        time_ticks = np.arange(0, duration + 10, 10)  # cada 10 segundos
    elif duration > 10:  # Si es mayor a 10 segundos
        time_ticks = np.arange(0, duration + 1, 1)  # cada segundo
    else:  # Si es menor a 10 segundos
        time_ticks = np.linspace(0, duration, 11)  # 11 ticks uniformemente distribuidos

    ax.set_xticks(time_ticks)
    ax.set_xticklabels([f'{t:.1f}' for t in time_ticks])

    # Set y-axis ticks for note names (simplified)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_ticks = []
    note_labels = []

    for octave in range(2, 9):  # Octavas 2-8 (rango de piano estándar)
        for i, note in enumerate(note_names):
            midi_note = octave * 12 + i
            if 21 <= midi_note <= 108:
                note_ticks.append(midi_note)
                note_labels.append(f'{note}{octave}')

    # Show only every 12th note to avoid overcrowding
    indices = list(range(0, len(note_ticks), 12))
    ax.set_yticks([note_ticks[i] for i in indices])
    ax.set_yticklabels([note_labels[i] for i in indices])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Velocity')

    plt.tight_layout()
    plt.show()


def plot_piano_roll_beats(midi_data, title="Piano Roll (by Beats)", figsize=(14, 8), trim_silence=True):
    """
    Creates a piano roll visualization (note vs beat grid) from MIDI data.

    Args:
        midi_data (pretty_midi.PrettyMIDI): The MIDI data object
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
    """
    if midi_data is None:
        print("No MIDI data to plot")
        return

    # Obtener información de tempo y beats
    tempi = midi_data.estimate_tempi()
    if len(tempi) == 0:
        print("No tempo information found, falling back to time-based visualization")
        plot_piano_roll(midi_data, title, figsize)
        return

    # Usar el tempo principal (el más común), normalizado a escalar
    if isinstance(tempi, tuple):
        tempo_values = tempi[0]
    else:
        tempo_values = tempi
    main_tempo = float(np.asarray(tempo_values).ravel()[0])  # BPM
    print(f"Main tempo: {main_tempo:.1f} BPM")

    # Obtener las posiciones de los beats
    beats = midi_data.get_beats()
    if len(beats) < 2:
        print("Insufficient beat information, falling back to time-based visualization")
        plot_piano_roll(midi_data, title, figsize)
        return

    # Construir tiempos basados en los beats reales (mejor alineación)
    beat_resolution = 4  # subdivisiones por beat (para más detalle)
    total_beats = len(beats) - 1
    if total_beats <= 0:
        print("Insufficient beat information, falling back to time-based visualization")
        plot_piano_roll(midi_data, title, figsize)
        return

    beat_times = []
    for i in range(total_beats):
        start = beats[i]
        end = beats[i + 1]
        beat_times.extend(np.linspace(start, end, beat_resolution, endpoint=False))
    beat_times = np.asarray(beat_times)

    # Opcional: recortar silencio inicial para alinear el inicio con la primera nota
    if trim_silence:
        first_note_time = None
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if first_note_time is None or note.start < first_note_time:
                    first_note_time = note.start
        if first_note_time is not None:
            beat_times = beat_times[beat_times >= first_note_time]

    if beat_times.size == 0:
        print("No beat grid available after trimming, falling back to time-based visualization")
        plot_piano_roll(midi_data, title, figsize)
        return

    # Generar piano roll directamente en esos tiempos
    piano_roll_beats = midi_data.get_piano_roll(times=beat_times)

    # Recortar columnas sin actividad para que el inicio visible sea el primer evento
    if trim_silence:
        active_cols = np.where(piano_roll_beats.max(axis=0) > 0)[0]
        if active_cols.size > 0:
            first_active = active_cols[0]
            piano_roll_beats = piano_roll_beats[:, first_active:]
        else:
            print("No note activity found, falling back to time-based visualization")
            plot_piano_roll(midi_data, title, figsize)
            return

    total_beats_display = piano_roll_beats.shape[1] / beat_resolution

    # Crear la visualización
    fig, ax = plt.subplots(figsize=figsize)

    # Plot del piano roll
    im = ax.imshow(piano_roll_beats.T,
                   aspect='auto',
                   origin='lower',
                   cmap='plasma',  # Mejor colormap para datos musicales
                   interpolation='nearest',  # Más nítido para distinguir notas
                   vmin=0,
                   vmax=127,
                   extent=[0, total_beats_display, 0, 127])

    # Configurar etiquetas
    ax.set_xlabel('Beats')
    ax.set_ylabel('MIDI Note Number')
    ax.set_title(f"{title}\n({main_tempo:.1f} BPM)")
    ax.set_ylim(21, 108)  # Rango de piano estándar

    # Ticks de beats - mostrar números de beat
    max_tick = int(np.floor(total_beats_display))
    step = max(1, max_tick // 10) if max_tick > 0 else 1
    beat_ticks = np.arange(0, max_tick + 1, step)
    ax.set_xticks(beat_ticks)
    ax.set_xticklabels([f'{int(b)}' for b in beat_ticks])

    # Ticks de notas mejorados
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_ticks = []
    note_labels = []

    for octave in range(2, 9):  # Octavas 2-8
        for i, note in enumerate(note_names):
            midi_note = octave * 12 + i
            if 21 <= midi_note <= 108:
                note_ticks.append(midi_note)
                note_labels.append(f'{note}{octave}')

    # Mostrar solo notas fundamentales para evitar overcrowding
    indices = list(range(0, len(note_ticks), 12))
    ax.set_yticks([note_ticks[i] for i in indices])
    ax.set_yticklabels([note_labels[i] for i in indices])

    # Agregar líneas verticales en cada beat principal
    for beat in range(0, total_beats, 4):  # Línea cada 4 beats (compás)
        ax.axvline(x=beat, color='white', alpha=0.3, linestyle='--', linewidth=1)

    # Colorbar con mejor etiqueta
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Velocity (0-127)')

    plt.tight_layout()
    plt.show()


def load_midi_file(midi_path):
    """
    Loads a MIDI file and returns the pretty_midi object.

    Args:
        midi_path (str or Path): Path to the MIDI file

    Returns:
        pretty_midi.PrettyMIDI: The loaded MIDI object
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        return midi_data
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return None
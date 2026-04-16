import os

import numpy as np
from scipy.io import wavfile


class IEC60601SoundSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

        # Base frequencies in Hz (C4 to G5)
        self.FREQS = {
            "C4": 261.63,
            "D4": 293.66,
            "E4": 329.63,
            "F4": 349.23,
            "G4": 392.00,
            "A4": 440.00,
            "B4": 493.88,
            "C5": 523.25,
            "D5": 587.33,
            "E5": 659.25,
            "F5": 698.46,
            "G5": 783.99,
        }

        # Standard IEC 60601-1-8 melody definitions
        self.MELODIES = {
            "General": ["C4", "C4", "C4", "C4", "C4"],
            "Cardiovascular": ["C4", "E4", "G4", "G4", "C5"],
            "Ventilation": ["C4", "A4", "F4", "A4", "F4"],
            "Oxygen": ["C4", "B4", "A4", "G4", "F4"],
            "Temperature": ["C4", "D4", "E4", "F4", "G4"],
            "DrugDelivery": ["C4", "D4", "G4", "C4", "D4"],
        }

    def _generate_pulse(self, freq, duration_ms, pitch_factor=1.0):
        """
        Generate a single pulse compliant with IEC 60601-1-8 standard.
        Includes fundamental frequency and at least 4 harmonics.
        """
        actual_freq = freq * pitch_factor
        t = np.linspace(
            0, duration_ms / 1000, int(self.sample_rate * duration_ms / 1000), False
        )

        # Harmonic amplitudes: fundamental is strongest, harmonics decrease smoothly
        harmonics = [1.0, 0.6, 0.4, 0.3, 0.2]
        wave = np.zeros_like(t)
        for i, amp in enumerate(harmonics):
            h_freq = actual_freq * (i + 1)
            # Anti-aliasing check
            if h_freq < self.sample_rate / 2:
                wave += amp * np.sin(2 * np.pi * h_freq * t)

        # Apply 15ms rise and fall time envelope
        env_len = int(15 * self.sample_rate / 1000)
        envelope = np.ones_like(t)
        if len(t) > env_len * 2:
            envelope[:env_len] = np.linspace(0, 1, env_len)
            envelope[-env_len:] = np.linspace(1, 0, env_len)

        return wave * envelope

    def create_alarm(self, config):
        """
        Generate full alarm sequence based on configuration.
        config requires: name, type, bpm, urgency, pitch
        """
        sys_type = config.get("type", "General")
        melody_keys = self.MELODIES.get(sys_type, self.MELODIES["General"])
        bpm = config.get("bpm", 120)
        pitch_factor = config.get("pitch", 1.0)
        urgency = config.get("urgency", "high")

        # Timing calculation based on BPM
        beat_ms = 60000 / bpm

        # Adjust pulse duration based on urgency level
        # Low/Medium alarms require longer pulse durations per standard
        if urgency == "low":
            pulse_len = beat_ms * 0.50  # Longest pulse for low urgency
        elif urgency == "medium":
            pulse_len = beat_ms * 0.35  # Medium pulse
        else:
            pulse_len = beat_ms * 0.25  # Short, sharp pulse for high urgency

        gap_ms = beat_ms * 0.15  # Standard gap between consecutive pulses

        segments = []

        if urgency == "high":
            # High priority: 5-note sequence (3+2), repeated twice
            def generate_high_unit():
                unit = []
                # First 3 notes
                for i in range(3):
                    unit.append(
                        self._generate_pulse(
                            self.FREQS[melody_keys[i]], pulse_len, pitch_factor
                        )
                    )
                    unit.append(np.zeros(int(gap_ms * self.sample_rate / 1000)))
                # Half-beat pause between note 3 and 4
                unit.append(np.zeros(int(beat_ms * 0.5 * self.sample_rate / 1000)))
                # Last 2 notes
                for i in range(3, 5):
                    unit.append(
                        self._generate_pulse(
                            self.FREQS[melody_keys[i]], pulse_len, pitch_factor
                        )
                    )
                    unit.append(np.zeros(int(gap_ms * self.sample_rate / 1000)))
                return unit

            unit_sequence = generate_high_unit()
            segments.extend(unit_sequence)
            # Pause between repeats
            segments.append(np.zeros(int(beat_ms * 1.5 * self.sample_rate / 1000)))
            segments.extend(unit_sequence)

        elif urgency == "medium":
            # Medium priority: First 3 notes of the sequence only
            for i in range(3):
                segments.append(
                    self._generate_pulse(
                        self.FREQS[melody_keys[i]], pulse_len, pitch_factor
                    )
                )
                segments.append(np.zeros(int(gap_ms * self.sample_rate / 1000)))

        else:
            # Low priority: 2 long notes (typically using the first note of the melody)
            for i in range(2):
                segments.append(
                    self._generate_pulse(
                        self.FREQS[melody_keys[0]], pulse_len, pitch_factor
                    )
                )
                segments.append(np.zeros(int(gap_ms * 2 * self.sample_rate / 1000)))

        # Concatenate and normalize to 16-bit PCM
        full_audio = np.concatenate(segments)
        full_audio = (full_audio / np.max(np.abs(full_audio)) * 32767).astype(np.int16)
        wavfile.write(config["name"], self.sample_rate, full_audio)
        print(f"Exported: {config['name']} | Type: {sys_type} | Urgency: {urgency}")


# --- Batch Generation Process ---
synth = IEC60601SoundSynthesizer()

# Define output directory
output_dir = "iec_alarms"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

systems = [
    "General",
    "Cardiovascular",
    "Ventilation",
    "Oxygen",
    "Temperature",
    "DrugDelivery",
]
urgencies = ["high", "medium", "low"]

# Standard parameters
# Higher urgency uses slightly higher pitch factor and BPM
params = {
    "high": {"bpm": 120, "pitch": 2.0},
    "medium": {"bpm": 100, "pitch": 1.5},
    "low": {"bpm": 90, "pitch": 1.5},
}

tasks = []
for sys in systems:
    for urg in urgencies:
        filename = f"{sys.lower()}_{urg}.wav"
        filepath = os.path.join(output_dir, filename)
        tasks.append(
            {
                "name": filepath,
                "type": sys,
                "urgency": urg,
                "bpm": params[urg]["bpm"],
                "pitch": params[urg]["pitch"],
            }
        )

# Execute generation
for task in tasks:
    synth.create_alarm(task)

print(f"All audio files generated in directory: {output_dir}")

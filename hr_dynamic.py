import os

import numpy as np
from scipy.io import wavfile


class HeartbeatSoundSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def generate_single_beat(self, spo2, hr):
        """
        Generates a single heartbeat monitor blip including the subsequent silence.

        :param spo2: Blood oxygen saturation percentage (e.g., 100). Determines pitch.
        :param hr: Heart rate in beats per minute (BPM). Determines total duration (gap).
        :return: Numpy array of the audio wave.
        """
        # 1. Pitch Mapping based on SpO2
        # Base frequency for 100% SpO2 is typically around 800Hz.
        # Frequency drops by 15Hz for every 1% drop in SpO2.
        max_freq = 800
        freq_drop_per_percent = 15
        freq = max_freq - (100 - spo2) * freq_drop_per_percent

        # Ensure frequency does not drop below a reasonable audible threshold (e.g., 200Hz)
        freq = max(200, freq)

        # 2. Generate the Pure Tone Pulse (Fixed duration: 60ms)
        pulse_duration = 0.06
        t_pulse = np.linspace(
            0, pulse_duration, int(self.sample_rate * pulse_duration), False
        )
        wave = np.sin(2 * np.pi * freq * t_pulse)

        # Apply 10ms linear fade-in/fade-out to prevent mechanical clicking
        fade_samples = int(0.01 * self.sample_rate)
        if len(wave) > fade_samples * 2:
            wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # 3. Calculate and append the silence gap based on Heart Rate
        # Total interval between beats in seconds = 60 / HR
        beat_interval = 60.0 / hr
        gap_duration = beat_interval - pulse_duration

        # Fallback if HR is impossibly high (e.g., > 1000 BPM)
        if gap_duration < 0:
            gap_duration = 0

        gap = np.zeros(int(self.sample_rate * gap_duration))

        # Combine the beep and the silence gap
        return np.concatenate([wave, gap])

    def export_audio(self, wave_data, filename):
        """Normalizes and exports the numpy array to a 16-bit PCM WAV file."""
        normalized = wave_data / np.max(np.abs(wave_data))
        pcm_16bit = (normalized * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, pcm_16bit)
        print(f"Exported: {filename}")


# --- Execution Process ---
if __name__ == "__main__":
    synth = HeartbeatSoundSynthesizer()

    # Define output directory
    output_dir = "monitor_sounds"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Simulate a deteriorating patient scenario for 10 beats:
    # SpO2 gradually drops from 100% to 82%
    # HR gradually increases from 60 BPM to 140 BPM
    patient_vitals = [
        {"spo2": 100, "hr": 60},  # Beat 1
        {"spo2": 100, "hr": 62},  # Beat 2
        {"spo2": 98, "hr": 65},  # Beat 3
        {"spo2": 95, "hr": 75},  # Beat 4
        {"spo2": 92, "hr": 85},  # Beat 5
        {"spo2": 90, "hr": 95},  # Beat 6
        {"spo2": 88, "hr": 105},  # Beat 7
        {"spo2": 85, "hr": 115},  # Beat 8
        {"spo2": 83, "hr": 125},  # Beat 9
        {"spo2": 82, "hr": 140},  # Beat 10
    ]

    sequence_segments = []

    for i, vitals in enumerate(patient_vitals):
        # Generate individual beat with its corresponding gap
        beat_audio = synth.generate_single_beat(spo2=vitals["spo2"], hr=vitals["hr"])
        sequence_segments.append(beat_audio)
        print(f"Generated beat {i + 1}: SpO2={vitals['spo2']}%, HR={vitals['hr']} BPM")

    # Concatenate all 10 beats into a single audio file
    full_sequence = np.concatenate(sequence_segments)

    filepath = os.path.join(output_dir, "spo2_hr_dynamic_10beats.wav")
    synth.export_audio(full_sequence, filepath)

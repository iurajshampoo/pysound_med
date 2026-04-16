import os

import numpy as np
from scipy.io import wavfile


class DefibrillatorSoundSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def _generate_standard_pulse(self, freq, duration_ms):
        """
        Generates a single pulse compliant with the IEC harmonic and envelope style.
        Used to maintain acoustic consistency with the main patient monitor.
        """
        t = np.linspace(
            0, duration_ms / 1000, int(self.sample_rate * duration_ms / 1000), False
        )

        # Consistent harmonic profile
        harmonics = [1.0, 0.6, 0.4, 0.3, 0.2]
        wave = np.zeros_like(t)
        for i, amp in enumerate(harmonics):
            h_freq = freq * (i + 1)
            if h_freq < self.sample_rate / 2:
                wave += amp * np.sin(2 * np.pi * h_freq * t)

        # 15ms rise/fall envelope
        env_len = int(15 * self.sample_rate / 1000)
        envelope = np.ones_like(t)
        if len(t) > env_len * 2:
            envelope[:env_len] = np.linspace(0, 1, env_len)
            envelope[-env_len:] = np.linspace(1, 0, env_len)

        return wave * envelope

    def generate_charging(self, duration_sec=4.0, start_freq=300, end_freq=1200):
        """
        Generates capacitor charging sound.
        The frequency sweeps up linearly to a lower target frequency.
        """
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), False)

        slope = (end_freq - start_freq) / duration_sec
        phase = 2 * np.pi * (start_freq * t + 0.5 * slope * t**2)

        # Applied mild harmonics to match the system tone quality
        wave = np.sin(phase) + 0.6 * np.sin(2 * phase) + 0.4 * np.sin(3 * phase)

        envelope = np.ones_like(t)
        fade_in_samples = int(0.2 * self.sample_rate)
        envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)

        return wave * envelope

    def generate_charge_complete(self, duration_sec=1.5, freq=1200):
        """
        Generates the 'charging complete' continuous ready tone.
        Matches the end frequency of the charging phase.
        """
        # Uses the standard pulse method to keep the harmonic texture identical
        wave = self._generate_standard_pulse(freq, duration_sec * 1000)
        return wave

    def generate_stand_clear(self, repeats=8, freq=1046.5):
        """
        Generates 'Stand Clear' warning alarm.
        Changed to a single-tone, rapid pulse pattern to match IEC standard monitor style.
        Default frequency is C6 (1046.5 Hz) for high urgency penetration.
        """
        pulse_duration = 100  # 100ms tone
        gap_duration = 100  # 100ms silence

        tone = self._generate_standard_pulse(freq, pulse_duration)
        gap = np.zeros(int(self.sample_rate * gap_duration / 1000))

        segments = []
        for _ in range(repeats):
            segments.extend([tone, gap])

        return np.concatenate(segments)

    def export_audio(self, wave_data, filename):
        """Normalizes and exports the numpy array to a 16-bit PCM WAV file."""
        normalized = wave_data / np.max(np.abs(wave_data))
        pcm_16bit = (normalized * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, pcm_16bit)
        print(f"Exported: {filename}")


# --- Execution Process ---
if __name__ == "__main__":
    synth = DefibrillatorSoundSynthesizer()

    # Define output directory
    output_dir = "defib_sounds"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Synthesize charging sound (4 seconds sweep, lower end frequency)
    charging = synth.generate_charging(duration_sec=4.0, end_freq=1200)
    synth.export_audio(charging, os.path.join(output_dir, "defib_charge.wav"))

    # 2. Synthesize charging complete tone (Continuous tone at end_freq)
    charge_complete = synth.generate_charge_complete(duration_sec=1.5, freq=1200)
    synth.export_audio(
        charge_complete, os.path.join(output_dir, "defib_charge_complete.wav")
    )

    # 3. Synthesize 'Stand Clear' warning (rapid single-tone pulse)
    stand_clear = synth.generate_stand_clear(repeats=8, freq=1046.5)
    synth.export_audio(stand_clear, os.path.join(output_dir, "defib_stand_clear.wav"))

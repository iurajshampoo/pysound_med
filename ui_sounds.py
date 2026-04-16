import os

import numpy as np
from scipy.io import wavfile


class BasicUISoundSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def _generate_pure_tone(self, freq, duration_sec, fade_ms=5):
        """
        Generates a basic pure sine wave with a simple linear fade.
        No complex harmonics or exponential decay.
        """
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), False)
        wave = np.sin(2 * np.pi * freq * t)

        # Simple linear fade to prevent hard clicking, keeping it basic
        fade_samples = int(fade_ms * self.sample_rate / 1000)
        envelope = np.ones_like(wave)
        if len(wave) > fade_samples * 2:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        return wave * envelope

    def generate_startup(self):
        """
        Generates a classic, basic startup sequence.
        Sequential non-overlapping tones (C4 - E4 - G4).
        """
        t1 = self._generate_pure_tone(261.63, 0.2)
        gap = np.zeros(int(0.075 * self.sample_rate))
        t2 = self._generate_pure_tone(329.63, 0.2)
        t3 = self._generate_pure_tone(392.00, 0.45)
        return np.concatenate([t1, gap, t2, gap, t3])

    def generate_click_normal(self):
        """
        Generates a standard UI button blip.
        Low pitch (400Hz), very short duration.
        """
        return self._generate_pure_tone(400, 0.03, fade_ms=2)

    def generate_click_success(self):
        """
        Generates a basic success confirmation beep.
        Two rapid, distinct pure tones stepping up.
        """
        tone1 = self._generate_pure_tone(800, 0.08)
        gap = np.zeros(int(0.04 * self.sample_rate))
        tone2 = self._generate_pure_tone(1000, 0.12)
        return np.concatenate([tone1, gap, tone2])

    def generate_click_error(self):
        """
        Generates a generic error buzzer sound.
        Uses wave clipping to simulate a basic square-wave buzzer.
        """
        duration = 0.2
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # Low frequency (150Hz) sine wave
        wave = np.sin(2 * np.pi * 150 * t)
        # Clip the wave to add harshness (similar to a cheap buzzer)
        wave = np.clip(wave * 1.5, -1, 1)

        # Linear fade
        fade = int(10 * self.sample_rate / 1000)
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)

        return wave

    def generate_system_notification(self):
        """
        Generates a simple system prompt.
        Alternating pure tones like an old terminal or pager.
        """
        t1 = self._generate_pure_tone(600, 0.2)
        t2 = self._generate_pure_tone(800, 0.4)
        return np.concatenate([t1, t2])

    def export_audio(self, wave_data, filename):
        """Normalizes and exports the numpy array to a 16-bit PCM WAV file."""
        normalized = wave_data / np.max(np.abs(wave_data))
        pcm_16bit = (normalized * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, pcm_16bit)
        print(f"Exported: {filename}")


# --- Execution Process ---
if __name__ == "__main__":
    synth = BasicUISoundSynthesizer()

    # Define output directory
    output_dir = "ui_sounds"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate and export sounds
    tasks = {
        "ui_basic_startup.wav": synth.generate_startup(),
        "ui_basic_click_normal.wav": synth.generate_click_normal(),
        "ui_basic_click_success.wav": synth.generate_click_success(),
        "ui_basic_click_error.wav": synth.generate_click_error(),
        "ui_basic_notification.wav": synth.generate_system_notification(),
    }

    for filename, wave_data in tasks.items():
        filepath = os.path.join(output_dir, filename)
        synth.export_audio(wave_data, filepath)

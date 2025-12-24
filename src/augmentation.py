import librosa
import numpy as np


class AudioAugmentation:
    def time_stretch(self, audio):
        rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y=audio, rate=rate)

    def pitch_shift(self, audio, sr):
        steps = np.random.randint(-2, 3)
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)

    def add_noise(self, audio):
        noise = np.random.randn(len(audio))
        return audio + 0.005 * noise

    def apply(self, audio, sr):
        aug_audio = audio.copy()

        if np.random.rand() < 0.5:
            aug_audio = self.time_stretch(aug_audio)

        if np.random.rand() < 0.5:
            aug_audio = self.pitch_shift(aug_audio, sr)

        if np.random.rand() < 0.5:
            aug_audio = self.add_noise(aug_audio)

        return aug_audio

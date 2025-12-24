import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa.display

class AudioProcessor:
    def __init__(self, sr=22050, duration=10):
        self.sr = sr
        self.duration = duration

    def add_noise(self, signal, noise_factor=0.005):
        noise = np.random.randn(len(signal))
        augmented = signal + noise_factor * noise
        return augmented.astype(type(signal[0]))

    def load_audio(self, path):
        audio, sr = librosa.load(path, sr=self.sr)
        audio = audio[:sr * self.duration]
        return audio

    def generate_mel_spectrogram(self, audio, save_path=None):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=128
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if save_path:
            plt.figure(figsize=(4, 3))
            librosa.display.specshow(mel_db, sr=self.sr)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return mel_db

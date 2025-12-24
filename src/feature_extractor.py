import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, n_mels=128, max_len=128):
        self.n_mels = n_mels
        self.max_len = max_len

    def extract_mel(self, audio, sr):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            fmax=8000
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = librosa.util.fix_length(mel_db, size=self.max_len, axis=1)

        mel_db = mel_db[..., np.newaxis] 
        return mel_db

import os
import numpy as np
import librosa
import soundfile as sf

from src.model import MelSpectrogramExtractor
from src.augmentation import AudioAugmentation


class DataLoader:
    def __init__(
        self,
        data_dir,
        genres,
        sr=22050,
        duration=3
    ):
        self.data_dir = data_dir
        self.genres = genres
        self.sr = sr
        self.duration = duration

        self.extractor = MelSpectrogramExtractor(max_len=130)
        self.augmentor = AudioAugmentation()

    def safe_load_audio(self, path):
        try:
            audio, sr = sf.read(path)

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            if sr != self.sr:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=self.sr
                )

            max_len = self.sr * self.duration
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            else:
                audio = audio[:max_len]

            return audio

        except Exception as e:
            print(f"Skip file lỗi: {path}")
            print(e)
            return None

    def load_dataset(self):
        X, y = [], []

        for label, genre in enumerate(self.genres):
            genre_path = os.path.join(self.data_dir, genre)

            if not os.path.exists(genre_path):
                print(f"Không tìm thấy thư mục: {genre_path}")
                continue

            for file in os.listdir(genre_path):
                if not file.lower().endswith(".wav"):
                    continue

                path = os.path.join(genre_path, file)

                audio = self.safe_load_audio(path)
                if audio is None:
                    continue

                mel = self.extractor.extract(audio, self.sr)
                X.append(mel[..., np.newaxis])
                y.append(label)

                aug_audio = self.augmentor.apply(audio, self.sr)
                mel_aug = self.extractor.extract(aug_audio, self.sr)
                X.append(mel_aug[..., np.newaxis])
                y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        print("Dataset loaded")
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        return X, y

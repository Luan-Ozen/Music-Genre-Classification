import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MELS = 128
FIXED_LENGTH = 128
SEGMENT_SECONDS = 3


class MusicInference:
    def __init__(self, model, genres):
        self.model = model
        self.genres = genres

    def _split_audio(self, audio, sr):
        samples = int(SEGMENT_SECONDS * sr)
        return [
            audio[i:i + samples]
            for i in range(0, len(audio) - samples, samples)
        ]

    def _extract_mel(self, audio, sr):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=N_MELS
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        if mel.shape[1] < FIXED_LENGTH:
            mel = np.pad(mel, ((0, 0), (0, FIXED_LENGTH - mel.shape[1])))
        else:
            mel = mel[:, :FIXED_LENGTH]

        return mel[..., np.newaxis]

    def predict_file(self, file):
        audio, sr = librosa.load(file, sr=SAMPLE_RATE)

        segments = self._split_audio(audio, sr)
        preds = []

        for seg in segments:
            mel = self._extract_mel(seg, sr)
            mel = np.expand_dims(mel, axis=0)
            preds.append(self.model.predict(mel, verbose=0)[0])

        return np.mean(preds, axis=0)

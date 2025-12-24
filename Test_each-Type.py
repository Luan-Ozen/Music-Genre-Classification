import os
import numpy as np
import librosa
from collections import defaultdict
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore")

# =============================
# CONFIG
# =============================
DATA_DIR = "data/GTZAN"
MODEL_PATH = "models/music_genre_model.keras"

GENRES = ["blues", "disco", "hiphop", "jazz", "pop"]

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3


# =============================
# MEL SPECTROGRAM EXTRACTOR
# =============================
class MelSpectrogramExtractor:
    def __init__(self):
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512

    def extract(self, audio, sr):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Fix shape (128, 130)
        if mel_db.shape[1] < 130:
            mel_db = np.pad(mel_db, ((0, 0), (0, 130 - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :130]

        return mel_db


# =============================
# UTILS
# =============================
def split_audio(audio, sr, seconds):
    segment_len = sr * seconds
    return [
        audio[i:i + segment_len]
        for i in range(0, len(audio) - segment_len, segment_len)
    ]


# =============================
# MAIN ANALYSIS
# =============================
def main():
    model = load_model(MODEL_PATH)
    extractor = MelSpectrogramExtractor()

    stats = {
        true_genre: defaultdict(int)
        for true_genre in GENRES
    }

    skipped_files = 0

    for true_genre in GENRES:
        genre_path = os.path.join(DATA_DIR, true_genre)
        print(f"\n🔍 Processing genre: {true_genre}")

        for file in os.listdir(genre_path):
            if not file.endswith(".wav"):
                continue

            path = os.path.join(genre_path, file)

            # =============================
            # SAFE AUDIO LOADING
            # =============================
            try:
                audio, sr = librosa.load(path, sr=SAMPLE_RATE)
            except Exception as e:
                skipped_files += 1
                print(f"⚠️ Skipped corrupted file: {file}")
                continue

            segments = split_audio(audio, sr, SEGMENT_SECONDS)
            if len(segments) == 0:
                continue

            predictions = []

            for seg in segments:
                mel = extractor.extract(seg, sr)
                mel = mel[np.newaxis, ..., np.newaxis]
                pred = model.predict(mel, verbose=0)[0]
                predictions.append(pred)

            mean_pred = np.mean(predictions, axis=0)
            predicted_genre = GENRES[np.argmax(mean_pred)]

            stats[true_genre][predicted_genre] += 1

    # =============================
    # PRINT RESULTS
    # =============================
    print("\n📊 PREDICTION DISTRIBUTION TABLE\n")

    header = "True \\ Pred".ljust(12)
    for g in GENRES:
        header += g.ljust(10)
    print(header)
    print("-" * len(header))

    for true_genre in GENRES:
        row = true_genre.ljust(12)
        for pred_genre in GENRES:
            row += str(stats[true_genre][pred_genre]).ljust(10)
        print(row)

    print(f"\n⚠️ Total skipped corrupted files: {skipped_files}")


if __name__ == "__main__":
    main()

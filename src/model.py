import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

class MelSpectrogramExtractor:
    def __init__(self, n_mels=128, max_len=130):
        self.n_mels = n_mels
        self.max_len = max_len

    def extract(self, audio, sr):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        if mel.shape[1] < self.max_len:
            pad_width = self.max_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mel = mel[:, :self.max_len]

        return mel


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

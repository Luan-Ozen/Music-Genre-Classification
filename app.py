import os
import numpy as np
import streamlit as st
import librosa
import tempfile
import matplotlib.pyplot as plt
import librosa.display

from tensorflow.keras.models import load_model

from src.trainer import Trainer
from src.model import MelSpectrogramExtractor

DATA_DIR = "data/GTZAN"
MODEL_PATH = "models/music_genre_model.keras"

GENRES = ["blues", "disco", "hiphop", "jazz", "pop"]

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
N_MELS = 128

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model chưa tồn tại. Đang tiến hành train model...")
        trainer = Trainer(
            data_dir=DATA_DIR,
            model_path=MODEL_PATH,
            genres=GENRES
        )
        trainer.run()
        st.success("Train model hoàn tất!")


@st.cache_resource
def load_trained_model():
    ensure_model()
    return load_model(MODEL_PATH)

def split_audio(audio, sr, seconds):
    segment_len = sr * seconds
    segments = []

    for i in range(0, len(audio) - segment_len, segment_len):
        segments.append(audio[i:i + segment_len])

    return segments


def plot_mel_spectrogram(mel, sr):
    fig, ax = plt.subplots(figsize=(8, 4))

    img = librosa.display.specshow(
        mel,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax
    )

    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    st.pyplot(fig)
    plt.close(fig)


st.set_page_config(page_title="Music Genre Classification", layout="centered")

st.title("Music Genre Classification")

uploaded_file = st.file_uploader("Upload file WAV", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    if len(audio) == 0:
        st.error("File audio rỗng hoặc không hợp lệ")
        st.stop()

    segments = split_audio(audio, sr, SEGMENT_SECONDS)

    extractor = MelSpectrogramExtractor()
    model = load_trained_model()

    all_predictions = []

    st.subheader("Mel Spectrogram")

    mel_example = extractor.extract(segments[0], sr)
    plot_mel_spectrogram(mel_example, sr)

    for seg in segments:
        mel = extractor.extract(seg, sr)
        mel = mel[np.newaxis, ..., np.newaxis]
        pred = model.predict(mel, verbose=0)[0]
        all_predictions.append(pred)

    mean_prediction = np.mean(all_predictions, axis=0)

    st.subheader("Xác suất dự đoán")

    for genre, prob in zip(GENRES, mean_prediction):
        st.write(f"**{genre}**: {prob * 100:.2f}%")

    final_genre = GENRES[np.argmax(mean_prediction)]
    confidence = np.max(mean_prediction) * 100

    st.success(f"Kết luận: **{final_genre.upper()}** ({confidence:.2f}%)")

import os
from audio_processor import AudioProcessor

class DatasetBuilder:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processor = AudioProcessor()

    def build(self):
        genres = os.listdir(self.data_dir)

        for genre in genres:
            genre_path = os.path.join(self.data_dir, genre)
            output_genre = os.path.join(self.output_dir, genre)
            os.makedirs(output_genre, exist_ok=True)

            for file in os.listdir(genre_path):
                if not file.endswith(".wav"):
                    continue

                audio_path = os.path.join(genre_path, file)
                audio = self.processor.load_audio(audio_path)
                audio = self.processor.add_noise(audio)

                save_img = os.path.join(
                    output_genre, file.replace(".wav", ".png")
                )
                self.processor.generate_mel_spectrogram(audio, save_img)

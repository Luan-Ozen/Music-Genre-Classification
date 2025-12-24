import os
from src.data_loader import DataLoader
from src.model import build_model


class Trainer:
    def __init__(self, data_dir, model_path, genres):
        self.data_dir = data_dir
        self.model_path = model_path
        self.genres = genres

    def run(self):
        loader = DataLoader(
            data_dir=self.data_dir,
            genres=self.genres
        )

        X, y = loader.load_dataset()

        model = build_model(
            input_shape=(128, 130, 1),
            num_classes=len(self.genres)
        )

        model.fit(
            X, y,
            epochs=25,
            batch_size=32,
            validation_split=0.2
        )

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save(self.model_path)

        print("MODEL SAVED:", self.model_path)

import os
from src.trainer import Trainer

MODEL_PATH = "models/music_genre_model.keras"

class Pipeline:
    def run(self):
        if not os.path.exists(MODEL_PATH):
            trainer = Trainer()
            trainer.run()

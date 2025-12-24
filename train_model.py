import os
from src.trainer import Trainer
from src.model import MusicGenreCNN

os.makedirs("models", exist_ok=True)

model_builder = MusicGenreCNN(
    input_shape=(128, 128, 1),
    num_classes=5
)
model = model_builder.build()


trainer = Trainer(
    img_dir="processed/spectrograms",
    model=model
)

trainer.train(epochs=30)

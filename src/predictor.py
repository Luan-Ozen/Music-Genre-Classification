from dataset_builder import DatasetBuilder
from model import MusicGenreModel
from trainer import Trainer

builder = DatasetBuilder("data/GTZAN", "processed/spectrograms")
builder.build()

model = MusicGenreModel((128,128,1), 5).build()
trainer = Trainer("processed/spectrograms", model)
trainer.train()


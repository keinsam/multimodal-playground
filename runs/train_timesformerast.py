import sys
import os
import yaml
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets.dummy_dataset import DummyMultimodalDataset
from models.timesformerast import TimeSformerAST

# Load hyperparameters
with open('configs/hparams.yaml', 'r') as f:
    hparams = yaml.safe_load(f)
NB_SAMPLES = hparams["nb_samples"]
NB_CLASSES = hparams['nb_classes']
BATCH_SIZE = hparams['batch_size']
EPOCHS = hparams['epochs']
LR = hparams['lr']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TimeSformerAST(nb_classes=NB_CLASSES, hf_token=HF_TOKEN).to(DEVICE)
dataset = DummyMultimodalDataset(nb_samples=NB_SAMPLES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    for audio_inputs, video_inputs, labels in dataloader:
        audio_inputs = {k: v.to(DEVICE) for k, v in audio_inputs.items()}
        video_inputs = {k: v.to(DEVICE) for k, v in video_inputs.items()}
        labels = labels.to(DEVICE)

        logits = model(audio_inputs, video_inputs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

print("Training complete.")

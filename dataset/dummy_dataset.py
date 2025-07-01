import torch
from torch.utils.data import Dataset

class DummyMultimodalDataset(Dataset):
    def __init__(self, nb_samples=10):
        self.nb_samples = nb_samples
        self.audio_shape = (1, 128, 1024)  # (C, Freq, Time)  
        self.video_shape = (3, 8, 224, 224)  # (C, T, H, W)

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        audio_inputs = {'input_values': torch.randn(128, 1024)}
        video_inputs = {'pixel_values': torch.randn(*self.video_shape)}
        label = torch.randint(0, 5, ()).item()
        return audio_inputs, video_inputs, label
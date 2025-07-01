import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchaudio

class FridgeMultimodalDataset(Dataset):
    def __init__(self, root_dir, transform_video=None, transform_audio=None, audio_sample_rate=16000):
        self.samples = []  # [(video_path, label)]
        self.label_to_idx = {}
        self.transform_video = transform_video
        self.transform_audio = transform_audio
        self.audio_sample_rate = audio_sample_rate

        classes = sorted(os.listdir(root_dir))
        for label, cls in enumerate(classes):
            self.label_to_idx[cls] = label
            class_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(class_dir):
                if fname.endswith(".mp4"):
                    video_path = os.path.join(class_dir, fname)
                    self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # Load video [T, H, W, C]
        video, _, _ = read_video(video_path, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        video = video.float() / 255.0  # Convert to float and normalize to [0, 1]

        if self.transform_video:
            video = self.transform_video(video)

        # Load audio
        audio_path = video_path.replace(".mp4", ".wav")
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.audio_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.audio_sample_rate)

        if self.transform_audio:
            waveform = self.transform_audio(waveform)  # MelSpectrogram
            # AST expects (freq, time) not (channels, freq, time)
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)

        return {'pixel_values': video, 'input_values': waveform}, label
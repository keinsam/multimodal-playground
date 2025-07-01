import torch
from torch import nn
from transformers import AutoModel

class TimeSformerAST(nn.Module):
    def __init__(self, nb_classes, hf_token=None):
        super().__init__()
        self.nb_classes = nb_classes
        
        self.ast = AutoModel.from_pretrained(
            'MIT/ast-finetuned-audioset-10-10-0.4593',
            token=hf_token
        )
        for param in self.ast.parameters():
            param.requires_grad = False
        self.timesformer = AutoModel.from_pretrained(
            'facebook/timesformer-base-finetuned-k400',
            token=hf_token
        )
        for param in self.timesformer.parameters():
            param.requires_grad = False
            
        ast_dim = self.ast.config.hidden_size
        timesformer_dim = self.timesformer.config.hidden_size
        self.classifier = nn.Linear(ast_dim + timesformer_dim, nb_classes)

    def forward(self, audio_inputs, video_inputs):
        input_values = audio_inputs['input_values']  # [B, freq, time] after batching
        pixel_values = video_inputs['pixel_values']  # [B, T, C, H, W] after batching
        
        ast_out = self.ast(input_values=input_values).last_hidden_state.mean(dim=1)
        
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        timesformer_out = self.timesformer(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
        
        fused = torch.cat([ast_out, timesformer_out], dim=1)
        logits = self.classifier(fused)
        return logits
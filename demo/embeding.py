# wavlm_feature_extractor.py
import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

class WavLMEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.model.eval()
    
    def extract(self, waveform):
        if isinstance(waveform, np.ndarray):
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = self.processor(
                waveform.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs).last_hidden_state  # [B, T, 768]

        embeddings = out.mean(dim=1)  # [B, 768]
        return embeddings

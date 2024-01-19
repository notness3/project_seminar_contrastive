import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEmbedder(nn.Module):
    def __init__(self,
                 model_path: str,
                 embedding_size: int = 512,
                 freeze: bool = False,
                 device: str = 'cpu',
                 normalize: bool = False):
        super().__init__()

        self.base_model = torch.load(model_path, map_location=torch.device(device))

        self.internal_embedding_size = self.base_model.classifier[0].in_features
        self.base_model.classifier = nn.Linear(in_features=self.internal_embedding_size, out_features=embedding_size)
        self.normalize = normalize

        if freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            for param in self.base_model.parameters():
                param.requires_grad = True
        self.base_model.classifier.requires_grad_(True)

        self.base_model.to(device)

    def embed_image(self, image):
        return self.base_model(image)

    def save(self, model_path):
        torch.save(self.base_model.state_dict(), os.path.join(model_path, 'model.pth'))

    @torch.no_grad()
    def get_embeddings(self, image):

        out = self.base_model.forward(image)

        if self.normalize:
            out = F.normalize(out, dim=1)

        return out.cpu().numpy()

    def forward(self, x):
        embedding = self.embed_image(x)

        if self.normalize:
            embedding = F.normalize(embedding, dim=-1)

        return embedding

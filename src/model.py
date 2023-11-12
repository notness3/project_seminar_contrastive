import os

import torch
import torch.nn as nn


class ImageEmbedder(nn.Module):
    def __init__(self,
                 model_path: str,
                 embedding_size: int = 512,
                 freeze: bool = False,
                 device: str = 'cpu'):
        super().__init__()

        self.base_model = torch.load(model_path, map_location=torch.device(device))

        self.internal_embedding_size = self.base_model.classifier[0].in_features
        self.base_model.classifier = nn.Linear(in_features=self.internal_embedding_size, out_features=embedding_size)

        if freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.classifier.requires_grad_(True)

        self.base_model.to(device)

    def embed_image(self, image):
        return self.base_model(image)

    def save(self, model_path):
        torch.save(self.base_model.state_dict(), os.path.join(model_path, 'model.pth'))

    def forward(self, x):
        embedding = self.embed_image(x)

        return embedding


if __name__ == '__main__':
    model = ImageEmbedder('/Users/notness/contrastive_visual_embed/model/enet_b0_8_best_vgaf.pt')


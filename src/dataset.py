import random
import pandas as pd
from torch.utils.data import Dataset

import os

import cv2
from tqdm import tqdm
from torchvision import transforms


class EmotionsDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size=(224, 224), emb_mode=False, model=None, sample_mode='triplet'):
        super(EmotionsDataset, self).__init__()

        self.dataset_root = dataset_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.emb_mode = emb_mode
        self.model = model
        self.embeddings = list()
        self.sample_mode = sample_mode

        self.images, self.names, self.labels, self.sample_nums = self._load_list(self.dataset_root)
        self.classes = {'Neutral': 0, 'Anger': 1, 'Disgust': 2, 'Fear': 3, 'Happiness': 4, 'Sadness': 5, 'Surprise': 6, 'Other': 7}

    def _load_list(self, list_root):
        samples, sample_names, sample_labels, frame_nums = list(), list(), list(), list()
        files_list = os.listdir(list_root)

        for file in tqdm(files_list):
            if file.endswith('.jpeg'):
                # path, frame, label
                path = os.path.join(list_root, file)

                sample, name, label, frame_num = self._load_samples_with_labels(path)

                samples.append(sample)
                sample_names.append(name)
                sample_labels.append(label)
                frame_nums.append(frame_num)

        return samples, sample_names, sample_labels, frame_nums

    def _load_samples_with_labels(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        name, class_label, frame_num = path.split('/')[-1].replace('.jpeg', '').split('_')

        if self.transform is not None:
            image = self.transform(image)

        if self.emb_mode:
            emb = self.model.get_embeddings(image.unsqueeze(0))

            self.embeddings.append(emb[0])

        return image, name, class_label, frame_num

    def extract_embeddings(self, model):
        print(f'Extracting embeddings')
        self.embeddings = model.get_embeddings(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.emb_mode:
            return self.embeddings[idx], int(self.labels[idx]), self.names[idx], int(self.sample_nums[idx])

        if self.sample_mode == 'triplet':
            anchor, anchor_label, anchor_name = self.images[idx], int(self.labels[idx]), self.names[idx]

            positive_list = [self.images[i] for i in range(len(self.images)) if
                             i != idx and int(self.labels[i]) == anchor_label]
            negative_list = [self.images[i] for i in range(len(self.images)) if
                             self.names[i] != anchor_name and int(self.labels[i]) != anchor_label]

            positive = random.choice(positive_list)
            negative = random.choice(negative_list)

            return anchor, positive, negative, anchor_label

        return self.images[idx], int(self.labels[idx]), self.names[idx], int(self.sample_nums[idx])


if __name__ == '__main__':
    # from src.model import ImageEmbedder
    #
    # model = ImageEmbedder('/Users/notness/contrastive_visual_embed/model/enet_b0_8_best_vgaf.pt')
    # eval = EmotionsDataset('/Users/notness/contrastive_visual_embed/dataset/val_prepare', emb_mode=True, model=model)
    #
    # pd.DataFrame({'label': eval.labels, 'video_name': eval.names, 'embeddings': eval.embeddings}).to_parquet('eval_1.parquet')

    dataset = EmotionsDataset('/Users/notness/contrastive_visual_embed/dataset/val_prepare')

    print(dataset[0])

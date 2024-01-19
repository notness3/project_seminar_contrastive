import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

import os

import cv2
from tqdm import tqdm
from torchvision import transforms


class EmotionsDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 img_size=(224, 224),
                 emb_mode=False,
                 model=None,
                 sample_mode: str = 'triplet',
                 count_negatives: int = 2,
                 protocol_path='/content/drive/MyDrive',
                 prefix='train'
                 ):
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
        self.n_count = count_negatives  # /val_balanced_protocol.csv

        self.protocol_path = os.path.join(protocol_path, f'{prefix}_balanced_protocol.csv')
        self.protocol = pd.read_csv(self.protocol_path)

        self.images, self.names, self.labels, self.sample_nums = self._load_list()
        self.classes = {'Neutral': 0, 'Anger': 1, 'Disgust': 2, 'Fear': 3, 'Happiness': 4, 'Sadness': 5, 'Surprise': 6,
                        'Other': 7}

    def _load_list(self):
        samples, sample_names, sample_labels, frame_nums = list(), list(), list(), list()

        for path in self.protocol['path'].tolist():
            # path, frame, label

            sample, name, label, frame_num = self._load_samples_with_labels(path)

            samples.append(sample)
            sample_names.append(name)
            sample_labels.append(int(label))
            frame_nums.append(frame_num)

        return samples, sample_names, sample_labels, frame_nums

    def _load_samples_with_labels(self, path):
        name, class_label, frame_num = path.split('/')[-1].replace('.jpeg', '').split('_')

        if self.emb_mode:
            image = cv2.imread(path)
            image = self.transform(image)

            emb = self.model.get_embeddings(image.unsqueeze(0))

            self.embeddings.append(emb[0])

        return path, name, class_label, frame_num

    def extract_embeddings(self, model):
        print(f'Extracting embeddings')
        self.embeddings = model.get_embeddings(self.images)

    def load_image(self, path):
        if type(path) != str:
            final_list = list()
            for p in path:
                image = cv2.imread(p)
                image = self.transform(image)

                final_list.append(image)

            return final_list

        image = cv2.imread(path)
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.emb_mode:
            return self.embeddings[idx], self.labels[idx], self.names[idx], self.sample_nums[idx]

        if self.sample_mode == 'triplet':
            anchor, anchor_label, anchor_name = self.load_image(self.images[idx]), self.labels[idx], self.names[idx]
            negative_list = list()

            for label in self.classes.values():
                if label != anchor_label:
                    negative_cand = [self.images[i] for i in range(len(self.images)) if
                                     (self.labels[i] == label)]

                    negative_list.extend(np.random.choice(negative_cand, self.n_count, replace=False).tolist())

            positive_list = [self.images[i] for i in range(len(self.images)) if
                             (anchor_name != self.names[i]) and (self.labels[i] == anchor_label)]

            positive = self.load_image(np.random.choice(positive_list, 3, replace=False).tolist())
            negative = self.load_image(negative_list)

            return anchor, positive, negative, anchor_label

        if self.sample_mode == 'arcface':
            anchor, anchor_label, anchor_name = self.load_image(self.images[idx]), int(self.labels[idx]), self.names[
                idx]

            return anchor, anchor_label

        return self.images[idx], int(self.labels[idx]), self.names[idx], int(self.sample_nums[idx])

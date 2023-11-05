from torch.utils.data import Dataset

import os

import cv2
from tqdm import tqdm
from torchvision import transforms


class EmotionsDataset(Dataset):
    def __init__(self, dataset_dir: str, interval: int = 4, img_size=(224, 224)):
        super(EmotionsDataset, self).__init__()

        self.dataset_root = dataset_dir
        self.interval = interval
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=img_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.samples, self.sample_labels = self._load_list(self.dataset_root)

    def _load_list(self, list_root):
        samples, sample_labels = list(), list()
        files_list = os.listdir(list_root)[:10]

        for file in tqdm(files_list):
            if file.endswith('.mp4'):
                # path, frame, label
                path = os.path.join(list_root, file)
                sample_list = self._load_samples(path)
                sample_classes = [file.split('-')[2]] * len(sample_list)

                samples.extend(sample_list)
                sample_labels.extend(sample_classes)

        return samples, sample_labels

    def _load_samples(self, path):
        capture = cv2.VideoCapture(path)
        samples = list()

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, frame_count, self.interval):
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = capture.read()

            if not ret:
                break

            if self.transform is not None:
                frame = self.transform(frame)

            samples.append(frame)

        capture.release()

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx], int(self.sample_labels[idx]) - 1


if __name__ == '__main__':
    train = EmotionsDataset('/Users/notness/contrastive_visual_embed/dataset/test')

    print(train[1])

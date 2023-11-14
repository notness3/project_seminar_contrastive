from torch.utils.data import Dataset

import os

import cv2
from tqdm import tqdm
from torchvision import transforms


class EmotionsDataset(Dataset):
    def __init__(self, dataset_dir: str, interval: int = 4, img_size=(224, 224), mode='afew'):
        super(EmotionsDataset, self).__init__()

        self.dataset_root = dataset_dir
        self.interval = interval
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mode = mode

        self.samples, self.sample_labels = self._load_list(self.dataset_root)

        self.classes = None
# TODO: normal class map
# TODO: think about similar classes
    def _load_list(self, list_root):
        samples, sample_labels = list(), list()
        files_list = os.listdir(list_root)[:10]

        if self.mode == 'abaw':
            for file in tqdm(files_list):
                if file.endswith('.mp4'):
                    # path, frame, label
                    path = os.path.join(list_root, file)
                    if file.replace('.mp4', '.txt') in os.listdir(f'{self.dataset_root}_txt'):
                        label_list = self._read_annotations(file)

                        sample_list, sample_classes = self._load_samples_with_labels(path, label_list)

                        samples.extend(sample_list)
                        sample_labels.extend(sample_classes)
        else:
            self.classes = {label: idx for idx, label in enumerate(files_list)}

            for class_name in tqdm(files_list):
                path_to_class = os.path.join(list_root, class_name)
                class_files_list = os.listdir(path_to_class)[:10]

                for file in tqdm(class_files_list):
                    if file.endswith('avi'):
                        # path, frame, label
                        path = os.path.join(path_to_class, file)

                        sample_list = self._load_samples(path)
                        sample_classes = [self.classes[class_name] for _ in sample_list]

                        samples.extend(sample_list)
                        sample_labels.extend(sample_classes)

        return samples, sample_labels

    def _read_annotations(self, filename):
        name = filename.split('.')[0]
        path = os.path.join(f'{self.dataset_root}_txt', f'{name}.txt')

        labels = [line.strip() for line in open(path, 'r')]
        self.classes = {label: idx for idx, label in enumerate(labels[0].split(','))}

        frame_labels = [int(lab) for lab in labels[1:]]

        return frame_labels

    def _load_samples_with_labels(self, path, list_labels):
        capture = cv2.VideoCapture(path)
        samples, sample_classes = list(), list()

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, frame_count, self.interval):
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = capture.read()

            if not ret:
                break

            if self.transform is not None:
                frame = self.transform(frame)

            class_label = list_labels[i]

            if class_label != -1:
                samples.append(frame)
                sample_classes.append(class_label)

        capture.release()

        return samples, sample_classes

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

        return self.samples[idx], int(self.sample_labels[idx])


if __name__ == '__main__':
    train = EmotionsDataset('/Users/notness/contrastive_visual_embed/dataset/test')

    print(train[1])

import os

import cv2
from tqdm import tqdm

from face_detector import extract_faces

classes = {'Neutral': 0, 'Anger': 1, 'Disgust': 2, 'Fear': 3, 'Happiness': 4, 'Sadness': 5, 'Surprise': 6, 'Other': 7}


def prepare_abaw(start_dir: str, save_dir: str) -> None:
    print(start_dir)
    os.makedirs(save_dir, exist_ok=True)
    files_list = os.listdir(start_dir)[131:]

    for file in tqdm(files_list):
        if file.endswith('.mp4'):
            path = os.path.join(start_dir, file)

            if file.replace('.mp4', '.txt') in os.listdir(f'{start_dir}_txt'):
                label_list = _read_annotations(start_dir, file)
                print(save_dir)
                _load_and_save_abaw(path, label_list, save_dir)


def _read_annotations(start_dir: str, filename: str):
    name = filename.split('.')[0]
    path = os.path.join(f'{start_dir}_txt', f'{name}.txt')

    labels = [line.strip() for line in open(path, 'r')]

    frame_labels = [int(lab) for lab in labels[1:]]

    return frame_labels


def _load_and_save_abaw(path, list_labels, save_dir, interval=10):
    capture = cv2.VideoCapture(path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, frame_count, interval):
        capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = capture.read()

        if not ret:
            break

        class_label = list_labels[i]

        file = path.split('/')[-1]
        filename = os.path.join(save_dir, f'{file}_{class_label}_{i}.jpeg')

        if class_label != -1:
            mask = extract_faces(frame)
            if mask:
                mask = mask[0][1]
                crop = frame[mask['y']:mask['y'] + mask['h'], mask['x']:mask['x'] + mask['w']]

                cv2.imwrite(filename, crop)

    capture.release()


if __name__ == '__main__':
    prepare_abaw('/Users/notness/contrastive_visual_embed/dataset/train', '/Users/notness/contrastive_visual_embed/dataset/train1_prepare')
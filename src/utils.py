import os
import random
import shutil

import numpy as np
import torch

from operator import itemgetter
from random import choices


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_tens(x):
    for i in range(len(x[0])):
        yield torch.stack(list(map(itemgetter(i), x)))


def parce_dataset(source_path: str, dest_path: str, val_size: float = 0.2) -> None:
    source_paths = [path for path in os.listdir(source_path) if path not in ['test', 'train', 'val', '.DS_Store']]

    for directory in source_paths:
        directory_path = os.path.join(source_path, directory)

        for file in os.listdir(directory_path):
            part = 'val' if choices([0, 1], [1 - val_size, val_size])[0] else 'train'
            destination_dir = os.path.join(dest_path, part)

            src_path, move_path = os.path.join(directory_path, file), os.path.join(destination_dir, file)

            shutil.move(src_path, move_path)


def collate_fn(batch):
    anchor = torch.stack(list(map(lambda x: x[0], batch)))
    positive = list(to_tens(list(map(lambda x: x[1], batch))))
    negative = list(to_tens(list(map(lambda x: x[2], batch))))
    target = torch.tensor(list(map(lambda x: x[3], batch)))

    return [anchor, positive, negative, target]


def collate_fn_arcface(batch):
    anchor = torch.stack(list(map(lambda x: x[0], batch)))
    target = torch.tensor(list(map(lambda x: x[1], batch)))

    return [anchor, target]


def cosine_similarity(emb1: np.ndarray, emb2: np.array) -> np.ndarray:
    """
    Batched cosine similarity for normalized vectors.
    :param emb1: (n, dim)
    :param emb2: (n, dim)
    :return: (n)
    """

    return np.sum(emb1 * emb2, axis=1)



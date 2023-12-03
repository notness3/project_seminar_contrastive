import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import EmotionsDataset
# must be here
from src.model import ImageEmbedder
from src.loss import ArcFaceLoss, TripletLoss
from src.trainer import EmotionsTrainer
from src.utils import collate_fn, set_all_seeds

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', default='/Users/notness/contrastive_visual_embed/dataset/train1_prepare')
    parser.add_argument('--val-dataset-dir', default='/Users/notness/contrastive_visual_embed/dataset/val_prepare')
    parser.add_argument('--test-dataset-dir', default='/Users/notness/contrastive_visual_embed/dataset/test')
    parser.add_argument('--checkpoint-dir', default='/Users/notness/contrastive_visual_embed/experiments', help='Checkpoint directory')
    parser.add_argument('--model-path', default='/Users/notness/contrastive_visual_embed/model/enet_b0_8_best_vgaf.pt',
                        help='Model directory')
    parser.add_argument('--model-class', default='ImageEmbedder', help='')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=64, help='Number of examples for each iteration')
    parser.add_argument('--accumulate-batches', type=int, default=1, help='Number of batches to accumulate')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optim-betas', type=list, nargs='+', default=[0.9, 0.999], help='Optimizer betas')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Optimizer weight decay')
    parser.add_argument('--threshold_chooser', type=str, default='accuracy',
                        help='Could be either max_f1, eer, accuracy')

    parser.add_argument('--loss', type=str, default='TripletLoss', help='Could be either ArcFaceLoss or TripletLoss')

    parser.add_argument('--seed', type=int, default=1004, help='Random seed value')
    parser.add_argument('--checkpoint-iter', type=int, default=5000, help='Eval and checkpoint frequency.')
    parser.add_argument('--scale-scores', type=bool, default=True,
                        help='Scale cosine similarity to [0, 1] for a better score interpretability')
    parser.add_argument('--device', default='cpu', help='Device to use for training: cpu or cuda')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        set_all_seeds(args.seed)

    model = eval(args.model_class)(
        model_path=args.model_path,
        device=args.device
    )

    train_dataset = EmotionsDataset(
        dataset_dir=args.dataset_dir
    )
    dev_dataset = EmotionsDataset(
        dataset_dir=args.val_dataset_dir,
    )
    test_dataset = EmotionsDataset(
        dataset_dir=args.test_dataset_dir,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6,
                                  drop_last=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6,
                                 collate_fn=collate_fn)

    #Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)

    loss = eval(args.loss)(
        emb_size=512,
        num_classes=8,
        device=args.device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=args.optim_betas,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        len(train_dataset) * args.epochs / args.batch_size,
        eta_min=1e-6
    )

    trainer = EmotionsTrainer(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        device=args.device
    )

    trainer.train(args.epochs)


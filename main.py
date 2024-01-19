import argparse

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from src.dataset import EmotionsDataset
# must be here
from src.model import ImageEmbedder
from src.loss import ArcFaceLoss, TripletLoss, ContrastiveCrossEntropy
from src.trainer import EmotionsTrainer
from src.utils import collate_fn, collate_fn_arcface, set_all_seeds

import warnings
warnings.filterwarnings("ignore")


def parse_argus():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', default='/content/dataset/train')
    parser.add_argument('--val-dataset-dir', default='/content/dataset/val')
    parser.add_argument('--test-dataset-dir', default='/content/dataset/test')
    parser.add_argument('--checkpoint-dir', default='/content/drive/MyDrive/experiments', help='Checkpoint directory')
    parser.add_argument('--model-path', default='/content/drive/MyDrive/enet_b0_8_best_vgaf.pt',
                        help='Model directory')
    parser.add_argument('--model-class', default='ImageEmbedder', help='')

    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=8, help='Number of examples for each iteration')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optim-betas', type=list, nargs='+', default=[0.9, 0.999], help='Optimizer betas')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Optimizer weight decay')

    parser.add_argument('--loss', type=str, default='TripletLoss', help='Could be either ArcFaceLoss or TripletLoss')

    parser.add_argument('--seed', type=int, default=1004, help='Random seed value')
    parser.add_argument('--device', default='cuda', help='Device to use for training: cpu or cuda')

    return parser.parse_args(args=[])


if __name__ == '__main__':
    args = parse_argus()

    wandb.init(
        # set the wandb project where this run will be logged
        project="Emotions_Recognition",

        # track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "architecture": args.model_path,
            "loss": args.loss
        }
    )

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

    # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)

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

    if args.loss == 'ArcFaceLoss':
        optimizer_arc = torch.optim.AdamW(
            loss.parameters(),
            lr=args.learning_rate,
            betas=args.optim_betas,
            weight_decay=args.weight_decay
        )
    else:
        optimizer_arc = None

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
        optimizer_arc=optimizer_arc,
        scheduler=scheduler,
        loss=loss,
        device=args.device
    )

    trainer.train(args.epochs)

    wandb.finish()


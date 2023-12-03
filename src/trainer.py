import os
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm


class EmotionsTrainer:
    def __init__(
            self, model, checkpoint_dir,
            train_dataloader, dev_dataloader, test_dataloader,
            optimizer, scheduler, loss,
            device='cuda', save_best=True
    ):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_best = save_best

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

        self.experiment_dir = os.path.join(self.checkpoint_dir,
                                           self.model.__class__.__name__+ f'_{datetime.today().strftime("%Y_%m_%d")}')
        self.model_dir = os.path.join(self.experiment_dir, 'model')

        self.eval_step = 0

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir), exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()

        epoch_loss = 0
        with tqdm(total=len(self.train_dataloader)) as pbar:
            for i, (anchor, positive, negative, class_label) in enumerate(self.train_dataloader):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)

                loss = self.loss(anchor_emb, positive_emb, negative_emb).sum()

                loss.backward(loss)
                self._optimizer_step()

                pbar.set_description('Epoch {} - current loss: {:.4f}'.format(epoch, loss.sum().item()))
                pbar.update(1)

        return epoch_loss / len(self.train_dataloader)

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()

        val_loss = 0
        for i, (anchor, positive, negative, class_label) in enumerate(self.dev_dataloader):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)

            loss = self.loss(anchor_emb, positive_emb, negative_emb).sum()

            val_loss += loss.item()

        return val_loss / len(self.dev_dataloader)

    def train(self, num_epochs):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            loss = self.train_epoch(epoch)
            print(f'Epoch {epoch} - loss {loss}')

            val_loss = self.val_epoch()
            print(f'Epoch {epoch} - validation loss {val_loss}')

#            self.eval()
            self._write_checkpoint(val_loss, best_val_loss)

    def _optimizer_step(self):
        self.optimizer.step()
        self.scheduler.step()

        self.optimizer.zero_grad()

    def _write_checkpoint(self, val_loss, best_val_loss):
        if self.save_best:
            if val_loss < best_val_loss:
                self.model.save(self.model_dir)

        else:
            save_dir = os.path.join(self.experiment_dir, f'loss_{val_loss}')
            os.makedirs(save_dir)
            self.model.save(save_dir)

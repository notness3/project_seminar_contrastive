import os
from datetime import datetime

import torch
import torch.nn.functional as F
import wandb

from tqdm import tqdm


class EmotionsTrainer:
    def __init__(
            self, model, checkpoint_dir,
            train_dataloader, dev_dataloader, test_dataloader,
            optimizer, optimizer_arc, scheduler, loss,
            device='cuda', save_best=False, sample_mode='triplet'
    ):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_best = save_best

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.optimizer_arc = optimizer_arc
        self.scheduler = scheduler
        self.loss = loss

        self.experiment_dir = os.path.join(self.checkpoint_dir,
                                           self.model.__class__.__name__+ f'_{datetime.today().strftime("%Y_%m_%d")}')
        self.model_dir = os.path.join(self.experiment_dir, 'model')

        self.eval_step = 0
        self.sample_mode = sample_mode

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir), exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()

        epoch_loss = 0
        with tqdm(total=len(self.train_dataloader)) as pbar:
            if self.sample_mode == 'triplet':
                for i, (anchor, positive, negative, class_label) in enumerate(self.train_dataloader):
                    anchor = anchor.to(self.device)
                    anchor_emb = self.model(anchor)

                    loss = torch.tensor(0.0).to(self.device)

                    for neg in negative:
                        neg = neg.to(self.device)
                        negative_emb = self.model(neg)
                        for pos in positive:
                          pos = pos.to(self.device)
                          positive_emb = self.model(pos)

                          loss += self.loss(anchor_emb, positive_emb, negative_emb, "train")

                    loss.backward(loss)

                    self._optimizer_step()

                    wandb.log({"train_loss": loss.item()})

                    pbar.set_description('Epoch {} - current loss: {:.4f}'.format(epoch, loss.item()))
                    pbar.update(1)

            if self.sample_mode == 'arcface':
                for i, (anchor, class_label) in enumerate(self.train_dataloader):
                    anchor = anchor.to(self.device)
                    class_label = class_label.to(self.device)

                    anchor_emb = self.model(anchor)

                    loss = self.loss(anchor_emb, class_label).sum()

                    loss.backward(loss)

                    self._optimizer_step()
                    self.optimizer_arc.step()
                    self.optimizer_arc.zero_grad()

                    wandb.log({"train_loss": loss.sum().item()})

                    pbar.set_description('Epoch {} - current loss: {:.4f}'.format(epoch, loss.sum().item()))
                    pbar.update(1)

        return epoch_loss / len(self.train_dataloader)

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()

        val_loss = 0
        with tqdm(total=len(self.dev_dataloader)) as pbar:
            if self.sample_mode == 'triplet':
                for i, (anchor, positive, negative, class_label) in enumerate(self.dev_dataloader):
                    anchor = anchor.to(self.device)
                    anchor_emb = self.model(anchor)

                    loss = torch.tensor(0.0).to(self.device)
                    for neg in negative:
                        neg = neg.to(self.device)
                        negative_emb = self.model(neg)
                        for pos in positive:
                          pos = pos.to(self.device)
                          positive_emb = self.model(pos)

                          loss += self.loss(anchor_emb, positive_emb, negative_emb, "eval")

                    val_loss += loss.item()

                    pbar.set_description('Epoch {} - current loss: {:.4f}'.format(epoch, loss.sum().item()))
                    pbar.update(1)

            if self.sample_mode == 'arcface':
                for i, (anchor, class_label) in enumerate(self.dev_dataloader):
                    anchor = anchor.to(self.device)
                    class_label = class_label.to(self.device)

                    anchor_emb = self.model(anchor)

                    loss = self.loss(anchor_emb, class_label).sum()

                    val_loss += loss.item()

                    pbar.set_description('Epoch {} - current loss: {:.4f}'.format(epoch, loss.sum().item()))
                    pbar.update(1)

        return val_loss / len(self.dev_dataloader)

    def train(self, num_epochs):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            loss = self.train_epoch(epoch)
            print(f'Epoch {epoch} - loss {loss}')
            wandb.log({"train_epoch_loss": loss})

            val_loss = self.val_epoch()
            print(f'Epoch {epoch} - validation loss {val_loss}')
            wandb.log({"val_loss": val_loss})
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


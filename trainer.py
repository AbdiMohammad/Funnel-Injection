import os
from datetime import datetime
import warnings

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        save_every: int = 50,
        output_dir: str = "run",
        es_patience: int = 20,
        device: torch.device = None,
        graceful_restart: bool = False
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.output_dir = output_dir
        self.es_patience = es_patience
        self.graceful_restart = graceful_restart

        self.epochs_run = 0
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_val_acc = -1
        self.last_val_loss = float('inf')
        self.es_cnt = 0
        self.datetime_now = datetime.now()

        os.makedirs(self.output_dir, exist_ok=True)
        if self.graceful_restart and os.path.exists(os.path.join(self.output_dir, "snapshot.pt")):
            print("Loading snapshot")
            self._load_snapshot()

    def _load_snapshot(self):
        snapshot = torch.load(os.path.join(self.output_dir, "snapshot.pt"), map_location=self.device)
        self.model.load_state_dict(snapshot["model_state"])
        self.epochs_run = snapshot["epoch"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        size = len(self.train_loader.dataset)
        print(f"lr: {self.optimizer.param_groups[0]['lr']} | Batchsize: {len(next(iter(self.train_loader))[0])} | nBatches: {len(self.train_loader)}")
        running_loss = 0.0
        self.model.train()
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            loss = self._run_batch(X, y)

            running_loss += loss
            if batch % 100 == 99:
                current = (batch + 1) * len(X)
                # print(f"avg loss: {running_loss / (batch + 1):>7f} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return running_loss / len(self.train_loader)

    def _validate(self):
        size = len(self.val_loader.dataset)
        num_batches = len(self.val_loader)
        self.model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        val_loss /= num_batches
        val_acc /= size
        print(f"Validation Error: \n Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        return val_loss, val_acc

    def _save_snapshot(self, epoch):
        snapshot = {
            "model_state": self.model.state_dict(),
            "epoch": epoch,
        }
        torch.save(snapshot, os.path.join(self.output_dir, "snapshot.pt"))
        print(f"Epoch {epoch} | Training snapshot saved at {os.path.join(self.output_dir, 'snapshot.pt')}")

    def _save_best(self, epoch):
        filename = f"best_{self.datetime_now.strftime('%Y%m%d_%H%M%S')}.pt"
        best = {
            "model": self.model,
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "val_loss": self.best_val_loss,
            "val_acc": self.best_val_acc,
            "train_loss": self.best_train_loss
        }
        torch.save(best, f"{os.path.join(self.output_dir, filename)}")

    def train(self, total_epochs: int):
        for epoch in range(self.epochs_run, total_epochs):
            self._run_epoch(epoch)
            self.lr_scheduler.step()
            if self.graceful_restart and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

    def train_and_validate(self, total_epochs: int, target_acc=None):
        for epoch in range(self.epochs_run, total_epochs):
            print(f"Epoch {epoch}\n-------------------------------")
            train_loss = self._run_epoch(epoch)

            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.modules.conv._ConvNd):
                    print(f'{name}\t{torch.sum(torch.norm(module.weight, p=2, dim=(1, 2, 3)) == 0) / module.weight.shape[0]}')

            val_loss, val_acc = self._validate()
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_loss)
            else:
                self.lr_scheduler.step()
            if self.graceful_restart and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            if val_loss < self.best_val_loss:
                if self.best_val_loss != float('inf'):
                    self._save_best(epoch)
                self.best_val_loss = val_loss
            
            if target_acc is not None and val_acc > target_acc:
                print("Target accuracy reached")
                break

            if self.last_val_loss < val_loss:
                self.es_cnt += 1
            else:
                self.es_cnt = 0
            if self.es_cnt > self.es_patience:
                print(f"Early Stopping: Validation loss doesn't decrease after {self.es_patience} epochs")
                break
            self.last_val_loss = val_loss

        print(f"Best Results:\nAcc: {self.best_val_acc}\tVal Loss: {self.best_val_loss}\tTrain Loss: {self.best_train_loss}")
        return epoch + 1
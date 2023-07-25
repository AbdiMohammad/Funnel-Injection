import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        save_every: int,
        output_dir: str,
        es_patience: int
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.epochs_run = 0
        self.output_dir = output_dir
        self.es_patience = es_patience

        self.best_val_loss = float('inf')
        self.last_val_loss = float('inf')
        self.es_cnt = 0
        self.datetime_now = datetime.now()
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.output_dir, "snapshot.pt")):
            print("Loading snapshot")
            self._load_snapshot()

        print(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(os.path.join(self.output_dir, "snapshot.pt"), map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
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
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {len(next(iter(self.train_loader))[0])} | nBatches: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        avg_loss = 0.0
        self.model.train()
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.local_rank), y.to(self.local_rank)

            loss = self._run_batch(X, y)

            running_loss += loss
            if batch % 100 == 99:
                avg_loss, current = running_loss / 100.0, (batch + 1) * len(X)
                if self.local_rank == 0:
                    print(f"avg loss: {avg_loss:>7f} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                running_loss = 0.0
        return avg_loss

    def _validate(self):
        size = len(self.val_loader.dataset)
        num_batches = len(self.val_loader)
        self.model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.local_rank), y.to(self.local_rank)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        val_loss /= num_batches
        val_acc /= size
        if self.local_rank == 0:
            print(f"Validation Error: \n Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        return val_loss, val_acc

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, os.path.join(self.output_dir, "snapshot.pt"))
        print(f"Epoch {epoch} | Training snapshot saved at {os.path.join(self.output_dir, 'snapshot.pt')}")

    def _save_best(self):
        filename = f"best_{self.datetime_now.strftime('%Y%m%d_%H%M%S')}.pt"
        filename_state = f"best_{self.datetime_now.strftime('%Y%m%d_%H%M%S')}_state.pt"
        torch.save(self.model.module, f"{os.path.join(self.output_dir, filename)}")
        torch.save(self.model.module.state_dict(), f"{os.path.join(self.output_dir, filename_state)}")

    def train(self, total_epochs: int):
        for epoch in range(self.epochs_run, total_epochs):
            self._run_epoch(epoch)
            self.lr_scheduler.step()
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

    def train_and_validate(self, total_epochs: int):
        for epoch in range(self.epochs_run, total_epochs):
            if self.local_rank == 0:
                print(f"Epoch {epoch+1}\n-------------------------------")
            self._run_epoch(epoch)
            val_loss, val_acc = self._validate()
            self.lr_scheduler.step(val_loss)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            if self.local_rank == 0 and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_best()
            if self.last_val_loss < val_loss:
                self.es_cnt += 1
            else:
                self.es_cnt = 0
            if self.es_cnt > self.es_patience:
                if self.local_rank == 0:
                    print(f"Early Stopping: Validation loss doesn't decrease after {self.es_patience} epochs")
                break
            self.last_val_loss = val_loss
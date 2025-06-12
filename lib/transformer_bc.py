# =============================================================================
# TransformerBCExp: Experiment Tracker for Transformer on Binary Classification
# -----------------------------------------------------------------------------
# Summary: Implements an experiment tracker for conducting binary
#          classification using a simplified transformer.
# Author: Q.WU
# =============================================================================

import os
import torch
import numpy as np
from typing import cast, Sized
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from infra.exp_tracker import ExpTracker
from model.tbc import TransformerForBC


class TransformerBCExp(ExpTracker):
    def __init__(self, epoch=10):
        super().__init__()
        # init general config
        self.cfg = {
            "seed": 2025,
            "d": 32,
            "L": 10,
            "num_heads": 4,
            "num_layers": 2,
            "num_classes": 2,
            "train_samples": 1000,
            "test_samples": 200,
            "batch_size": 32,
            "epochs": epoch,
            "lr": 1e-3,
        }
        self.cfg["exp_code"] = "ml_3"  # tmp project codename
        self.cfg["tmp_dir"] = f".tmp/{self.cfg['exp_code']}"
        os.makedirs(self.cfg["tmp_dir"], exist_ok=True)
        # init stats
        #   [epochID, 0]: training loss
        #   [epochID, 1]: testing loss
        #   [epochID, 2]: testing acc
        self.stats = np.zeros((self.cfg["epochs"], 3))
        # init exp dependencies
        torch.manual_seed(self.cfg["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.prep_data()
        self.model = TransformerForBC(
            d=self.cfg["d"],
            L=self.cfg["L"],
            num_heads=self.cfg["num_heads"],
            num_layers=self.cfg["num_layers"],
            num_classes=self.cfg["num_classes"],
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg["lr"])

    def prep_data(self, saveData=True):
        class TBCDataset(Dataset):
            def __init__(self, config, num_samples=1000):
                self.d = config["d"]
                self.L = config["L"]
                self.data = torch.randn(num_samples, self.L, self.d)
                self.labels = (self.data.sum(dim=(1, 2)) > 0).long()

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        train_dataset = TBCDataset(self.cfg, self.cfg["train_samples"])
        test_dataset = TBCDataset(self.cfg, self.cfg["test_samples"])
        self.trainloader = DataLoader(
            train_dataset, batch_size=self.cfg["batch_size"], shuffle=True
        )
        self.testloader = DataLoader(
            test_dataset, batch_size=self.cfg["batch_size"], shuffle=False
        )

        if saveData:
            datapath = os.path.join(self.cfg["tmp_dir"], "data.pt")
            torch.save(
                {
                    "x_train": train_dataset.data.mean(dim=1),
                    "y_train": train_dataset.labels,
                    "x_test": test_dataset.data.mean(dim=1),
                    "y_test": test_dataset.labels,
                },
                datapath,
            )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()

        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss / len(cast(Sized, self.trainloader.dataset))
        avg_acc = correct / len(cast(Sized, self.trainloader.dataset))
        return avg_loss, avg_acc

    @torch.no_grad()
    def test(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()

        for x, y in self.testloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss / len(cast(Sized, self.testloader.dataset))
        avg_acc = correct / len(cast(Sized, self.testloader.dataset))
        return avg_loss, avg_acc

    def exec(self, verbose=True, saveStats=True):
        for epoch in range(self.cfg["epochs"]):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test()
            self.stats[epoch, :] = [train_loss, test_loss, test_acc]
            if verbose:
                print(
                    f"Epoch {epoch + 1:02d}: "
                    f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} | "
                    f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}"
                )
        if saveStats:
            filename = os.path.join(
                self.cfg["tmp_dir"],
                "stats.npz",
            )
            np.savez(filename, stats=self.stats)


if __name__ == "__main__":
    exp = TransformerBCExp()
    exp.exec()

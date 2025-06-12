# =============================================================================
# ViTFCTracker: Experiment Tracker for ViT on Image Classification
# -----------------------------------------------------------------------------
# Summary: Implements an experiment tracker for conducting image
#          classification on cifar10 data using a vision transformer.
# Author: Q.WU
# =============================================================================

import os
import torch
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from typing import cast, Sized
import numpy as np
from infra.exp_tracker import ExpTracker
from model.vitfc import ViTForClassfication


class ViTFCExp(ExpTracker):
    def __init__(self, epoch=100):
        super().__init__()
        # init general config
        self.cfg["num_workers"] = 4  # set to num. of cpu core
        self.cfg["seed"] = 2025
        self.cfg["batch_sz"] = 256
        self.cfg["epochs"] = epoch
        self.cfg["lr"] = 1e-2
        self.cfg["weight_decay"] = 1e-2
        self.cfg["exp_code"] = "ml_4"  # tmp project codename
        self.cfg["tmp_dir"] = f".tmp/{self.cfg['exp_code']}"
        os.makedirs(self.cfg["tmp_dir"], exist_ok=True)
        # init vitfc config
        self.cfg_vitfc = {
            "patch_size": 4,
            "hidden_size": 48,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4 * 48,  # 4 * hidden_size
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "initializer_range": 0.02,
            "image_size": 32,
            "num_classes": 10,  # num_classes of CIFAR10
            "num_channels": 3,
            "qkv_bias": True,
            "use_faster_attention": True,
        }
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
        self.model = ViTForClassfication(self.cfg_vitfc).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        # validation
        assert (
            self.cfg_vitfc["hidden_size"] % self.cfg_vitfc["num_attention_heads"] == 0
        )
        assert self.cfg_vitfc["image_size"] % self.cfg_vitfc["patch_size"] == 0

    def prep_data(self, use_local=False):
        # we hardcode cifar-10 as the toy example
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(
                    (32, 32),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if not use_local:
            datapath = os.path.join(self.cfg["tmp_dir"], "data")
            trainset = torchvision.datasets.CIFAR10(
                root=datapath,
                train=True,
                download=True,
                transform=train_transform,
            )
            testset = torchvision.datasets.CIFAR10(
                root=datapath,
                train=False,
                download=True,
                transform=test_transform,
            )
        else:
            raise Exception

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.cfg["batch_sz"],
            num_workers=self.cfg["num_workers"],
        )
        self.testloader = torch.utils.data.DataLoader(
            testset,
            shuffle=False,
            batch_size=self.cfg["batch_sz"],
            num_workers=self.cfg["num_workers"],
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            self.optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(self.model(images)[0], labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(images)
        return total_loss / len(cast(Sized, self.trainloader.dataset))

    def test(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in self.testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                logits, _ = self.model(images)
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item() * len(images)

                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(cast(Sized, self.testloader.dataset))
        avg_loss = total_loss / len(cast(Sized, self.testloader.dataset))
        return avg_loss, accuracy

    def exec(self, verbose=True, saveStats=True):
        for i in range(self.cfg["epochs"]):
            self.stats[i, 0] = self.train_epoch()
            self.stats[i, 1], self.stats[i, 2] = self.test()
            if verbose:
                print(
                    f"Epoch: {i + 1}, Train loss: {self.stats[i, 0]:.4f}, Test loss: {self.stats[i, 1]:.4f}, Accuracy: {self.stats[i, 2]:.4f}"
                )
        if saveStats:
            filename = os.path.join(
                self.cfg["tmp_dir"],
                "stats.npz",
            )
            np.savez(filename, stats=self.stats)


if __name__ == "__main__":
    exp = ViTFCExp()
    exp.exec()

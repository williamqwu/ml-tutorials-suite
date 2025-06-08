from abc import ABC, abstractmethod
from typing import Any
import torch
import numpy as np


class ExpTracker(ABC):
    ######### Configs
    cfg: dict[Any, Any]

    ######### Running variables
    device: torch.device
    trainloader: torch.utils.data.DataLoader
    testloader: torch.utils.data.DataLoader
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    stats: np.ndarray

    def __init__(self):
        self.cfg = {}

    @abstractmethod
    def prep_data(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def exec(self):
        pass

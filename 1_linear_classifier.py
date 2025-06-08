# Building and Understanding Linear Classifiers with Perceptron
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from infra.exp_tracker import ExpTracker
from model.perceptron import Perceptron


class LinearClassifierExp(ExpTracker):
    def __init__(self, case_study=1):
        super().__init__()
        # init general config
        self.cfg["epochs"] = 50
        self.cfg["case"] = case_study
        self.cfg["input_dim"] = 2
        self.cfg["n_train"] = 100
        self.cfg["n_test"] = 100
        self.cfg["exp_code"] = "ml_1"  # tmp project codename
        self.cfg["tmp_dir"] = f".tmp/{self.cfg['exp_code']}"
        # init stats
        #   [epochID, 0]: training loss
        #   [epochID, 1]: training acc
        #   [epochID, 2]: testing loss
        #   [epochID, 3]: testing acc
        self.stats = np.zeros((self.cfg["epochs"], 3))
        # init exp dependencies
        self.device = torch.device("cpu")
        self.prep_data()
        self.model = Perceptron(self.cfg["input_dim"])

    def prep_data(self):
        def __gen1(n, dim):
            x = torch.randn(n, dim)
            y = torch.sign(x[:, 0] + x[:, 1])
            return x, y

        def __gen2(n, dim):
            x = torch.randn(n, dim)
            y = torch.sign(x[:, 0] * x[:, 1])
            return x, y

        if self.cfg["case"] == 1:
            x_train, y_train = __gen1(self.cfg["n_train"], self.cfg["input_dim"])
            x_test, y_test = __gen1(self.cfg["n_test"], self.cfg["input_dim"])
        elif self.cfg["case"] == 2:
            x_train, y_train = __gen2(self.cfg["n_train"], self.cfg["input_dim"])
            x_test, y_test = __gen2(self.cfg["n_test"], self.cfg["input_dim"])
        else:
            raise Exception(f"Unknown case type: cfg['case']={self.cfg['case']}.")

        self.trainloader = DataLoader(TensorDataset(x_train, y_train), batch_size=1, shuffle=False)
        self.testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False)

    def train_epoch(self):
        correct = 0
        total = 0
        loss_sum = 0.0
        for x, y in self.trainloader:
            x, y = x.squeeze(0), y.item()
            y_pred = self.model(x).item()
            if y_pred != y:
                self.model.weights.data += y * x
                loss_sum += 1
            else:
                correct += 1
            total += 1
        return loss_sum, correct / total

    def test(self):
        correct = 0
        total = 0
        loss_sum = 0.0
        for x, y in self.testloader:
            y_pred = self.model(x).squeeze()
            loss_sum += torch.sum(y_pred != y).item()
            correct += torch.sum(y_pred == y).item()
            total += y.size(0)
        return loss_sum, correct / total

    def exec(self, verbose=True, saveStats=True):
        for epoch in range(self.cfg["epochs"]):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test()
            self.stats[epoch, :] = [train_loss, train_acc, test_loss, test_acc]
            if verbose:
                print(
                    f"Epoch {epoch + 1:02d}: "
                    f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} | "
                    f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}"
                )
        if saveStats:
            np.savez(self.cfg["tmp_dir"], stats=self.stats)


if __name__ == "__main__":
    print("Running Easy Case:")
    easy = LinearClassifierExp(case_study=1)
    easy.exec()

    print("\nRunning Hard Case:")
    hard = LinearClassifierExp(case_study=2)
    hard.exec()
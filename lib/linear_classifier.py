# Building and Understanding Linear Classifiers with Perceptron
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from infra.exp_tracker import ExpTracker
from model.perceptron import Perceptron


class LinearClassifierExp(ExpTracker):
    def __init__(self, case_study=1):
        super().__init__()
        # init general config
        self.cfg["case"] = case_study
        if case_study != 0:
            self.cfg["epochs"] = 25
        else:
            self.cfg["epochs"] = 6
        self.cfg["input_dim"] = 2
        self.cfg["n_train"] = 100
        self.cfg["n_test"] = 100
        self.cfg["seed"] = 2025
        self.cfg["exp_code"] = "ml_1"  # tmp project codename
        self.cfg["tmp_dir"] = f".tmp/{self.cfg['exp_code']}"
        os.makedirs(self.cfg["tmp_dir"], exist_ok=True)
        # init stats
        #   [epochID, 0]: training err
        #   [epochID, 1]: cum. training acc
        #   [epochID, 2]: testing err
        #   [epochID, 3]: testing acc
        self.stats = np.zeros((self.cfg["epochs"], 4))
        # init tracker for weight history
        self.w_hist = np.zeros((self.cfg["epochs"] + 1, self.cfg["input_dim"]))
        # init exp dependencies
        self.device = torch.device("cpu")
        self.prep_data()
        self.model = Perceptron(self.cfg["input_dim"])
        self.epochID = 0
        torch.manual_seed(self.cfg["seed"])

    def prep_data(self, saveData=True):
        def __gen0():
            x = torch.tensor(
                [
                    [-1.0, 2.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [-1.0, 0.0],
                    [-1.0, -2.0],
                    [1.0, -1.0],
                ]
            )
            y = torch.sign(x[:, 0])
            return x, y

        def __genk(n, dim, case):
            if case == 1:
                margin = 0.3
                x = torch.randn(n, dim)
                y = torch.sign(x[:, 0] + x[:, 1])
                # ensure no label is exactly zero
                y[y == 0] = 1
                # shift positively labeled points away from boundary
                shift_vec = torch.tensor([1.0, 1.0])
                for i in range(n):
                    x[i] += margin * y[i] * shift_vec / shift_vec.norm()
                return x, y
            elif case == 2:
                x = torch.randn(n, dim)
                y = torch.sign(x[:, 0] + x[:, 1])
                return x, y
            elif case == 3:
                x = torch.randn(n, dim)
                y = torch.sign(x[:, 0] * x[:, 1])
                return x, y
            else:
                raise Exception(f"Unknown case type: cfg['case']={self.cfg['case']}.")

        if self.cfg["case"] == 0:
            x_train, y_train = __gen0()
            x_test, y_test = __gen0()
        else:
            x_train, y_train = __genk(
                self.cfg["n_train"], self.cfg["input_dim"], self.cfg["case"]
            )
            x_test, y_test = __genk(
                self.cfg["n_train"], self.cfg["input_dim"], self.cfg["case"]
            )

        self.trainloader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=1, shuffle=False
        )
        self.testloader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=32, shuffle=False
        )

        if saveData:
            datapath = os.path.join(
                self.cfg["tmp_dir"], f"data_case{self.cfg['case']}.pt"
            )
            torch.save(
                {
                    "x_train": x_train,
                    "y_train": y_train,
                    "x_test": x_test,
                    "y_test": y_test,
                },
                datapath,
            )

    def train_epoch(self):
        dataset = list(self.trainloader)
        if self.epochID >= len(dataset):
            raise IndexError(
                f"Epoch {self.epochID} exceeds training data length {len(dataset)}"
            )

        x, y = dataset[self.epochID]
        x, y = x.squeeze(0), y.item()
        y_pred = self.model(x).item()

        is_correct = False
        if y_pred != y:
            self.model.weights.data += y * x
        else:
            is_correct = True

        return is_correct

    def test(self):
        correct = 0
        total = 0
        err_sum = 0.0
        for x, y in self.testloader:
            y_pred = self.model(x).squeeze()
            err_sum += torch.sum(y_pred != y).item()
            correct += torch.sum(y_pred == y).item()
            total += y.size(0)
        return err_sum, correct / total

    def exec(self, verbose=True, saveStats=True):
        train_err = 0
        train_correct = 0
        assert isinstance(self.model.weights, torch.Tensor)
        for self.epochID in range(self.cfg["epochs"]):
            self.w_hist[self.epochID, :] = self.model.weights.numpy()
            is_correct = self.train_epoch()
            if is_correct:
                train_correct += 1
            else:
                train_err += 1
            train_acc = train_correct / (self.epochID + 1)
            test_err, test_acc = self.test()
            self.stats[self.epochID, :] = [train_err, train_acc, test_err, test_acc]
            if verbose:
                print(
                    f"Phase {self.epochID + 1:02d}: "
                    f"Err = {is_correct}, Cum. Train Acc = {train_acc:.4f} | "
                    f"Test Err = {test_err:.0f}, Test Acc = {test_acc:.4f}"
                )
        self.w_hist[self.epochID + 1, :] = self.model.weights.numpy()
        if saveStats:
            filename = os.path.join(
                self.cfg["tmp_dir"], f"stats_case{self.cfg['case']}.npz"
            )
            np.savez(filename, stats=self.stats, w_hist=self.w_hist)


if __name__ == "__main__":
    print("Running demo case from slides:")
    easy = LinearClassifierExp(case_study=0)
    easy.exec()

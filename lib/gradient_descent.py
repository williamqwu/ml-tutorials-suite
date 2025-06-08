import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from infra.exp_tracker import ExpTracker


class LinearGDExp(ExpTracker):
    def __init__(self, case_study=1, lr=0.1, mode="gd"):
        super().__init__()
        # init general config
        self.cfg["case"] = case_study
        self.cfg["tr_mode"] = mode
        self.cfg["epochs"] = 25
        self.cfg["input_dim"] = 2
        self.cfg["lr"] = lr
        self.cfg["n_train"] = 200
        self.cfg["seed"] = 2025
        self.cfg["exp_code"] = "ml_2"  # tmp project codename
        self.cfg["tmp_dir"] = f".tmp/{self.cfg['exp_code']}"
        os.makedirs(self.cfg["tmp_dir"], exist_ok=True)
        # init stats
        #   [epochID, 0]: training loss
        #   [epochID, 1]: training acc
        self.stats = np.zeros((self.cfg["epochs"], 2))
        # init tracker for weight history
        self.w_hist = np.zeros((self.cfg["epochs"] + 1, self.cfg["input_dim"] + 1))
        # init exp dependencies
        self.device = torch.device("cpu")
        self.prep_data()
        self.model = torch.nn.Linear(self.cfg["input_dim"], 1, bias=True)
        self.epochID = 0
        torch.manual_seed(self.cfg["seed"])

    def prep_data(self, saveData=True):
        def __genk(n, dim, case):
            X = torch.randn(n, dim)
            if case == 1:
                noise_ratio = 0.0
            elif case == 2:
                noise_ratio = 0.1
            elif case == 3:
                noise_ratio = 0.3
            else:
                raise Exception(f"Unknown case type: cfg['case']={self.cfg['case']}.")
            # create a linearly separable dataset first
            y = (X.sum(dim=1) > 0).float()
            # optionally, add noise
            if noise_ratio > 0:
                num_noisy = int(noise_ratio * n)
                noise_indices = torch.randperm(n)[:num_noisy]
                # shift feat. vec. across decision boundary
                shift_vector = torch.tensor([1.0, 1.0])
                for idx in noise_indices:
                    # shift it towards the opposite class
                    direction = -1 if y[idx] == 1 else 1
                    X[idx] += direction * 0.5 * shift_vector / shift_vector.norm()
            return X, y.unsqueeze(1)

        x_train, y_train = __genk(
            self.cfg["n_train"], self.cfg["input_dim"], self.cfg["case"]
        )
        batch_sz = 1 if self.cfg["tr_mode"] == "sgd" else self.cfg["n_train"]
        self.trainloader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=batch_sz, shuffle=False
        )

        if saveData:
            datapath = os.path.join(
                self.cfg["tmp_dir"], f"data_case{self.cfg['case']}.pt"
            )
            torch.save(
                {
                    "x_train": x_train,
                    "y_train": y_train,
                },
                datapath,
            )

    def train_epoch(self):
        epoch_losses = []
        correct = 0
        total = 0

        for X_batch, y_batch in self.trainloader:
            logits = self.model(X_batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_batch)
            self.model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.cfg["lr"] * param.grad
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
            epoch_losses.append(loss.item())
        accuracy = correct / total
        return np.mean(epoch_losses), accuracy

    def test(self):
        pass

    def exec(self, verbose=True, saveStats=True):
        assert isinstance(self.model.weight, torch.Tensor)
        assert isinstance(self.model.bias, torch.Tensor)

        for self.epochID in range(self.cfg["epochs"]):
            self.w_hist[self.epochID, : self.cfg["input_dim"]] = (
                self.model.weight.detach().numpy()
            )
            self.w_hist[self.epochID, -1] = self.model.bias.detach().numpy()
            train_loss, train_acc = self.train_epoch()
            self.stats[self.epochID, :] = [train_loss, train_acc]
            if verbose:
                print(
                    f"Epoch {self.epochID + 1:02d}: "
                    f"Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}"
                )

        self.w_hist[self.epochID + 1, : self.cfg["input_dim"]] = (
            self.model.weight.detach().numpy()
        )
        self.w_hist[self.epochID + 1, -1] = self.model.bias.detach().numpy()
        if saveStats:
            filename = os.path.join(
                self.cfg["tmp_dir"], f"stats_case{self.cfg['case']}.npz"
            )
            np.savez(filename, stats=self.stats, w_hist=self.w_hist)


if __name__ == "__main__":
    exp = LinearGDExp(case_study=1)
    exp.exec()

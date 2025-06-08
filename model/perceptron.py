import torch


class Perceptron(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(input_dim), requires_grad=False)

    def forward(self, x):
        return torch.sign(x @ self.weights)

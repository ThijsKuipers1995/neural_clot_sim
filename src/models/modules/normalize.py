import torch
from torch import Tensor, nn


class Normalizer(nn.Module):
    __constants__ = ["input_size", "max_accumulations", "std_epsilon"]

    def __init__(
        self, input_size: int, max_accumulations: int = 10**6, std_epsilon: float = 1e-8
    ):
        super().__init__()

        self.input_size = input_size
        self.max_accumulations = max_accumulations
        self.std_epsilon = std_epsilon

        # how many tensors we accumulated, e.g., a batch of N tensors increases count by N
        self.register_buffer("accumulation_count", torch.zeros(1))

        self.register_buffer("accumulated_sum", torch.zeros(self.input_size))
        self.register_buffer("accumulated_sum_sq", torch.zeros(self.input_size))

    def mean(self) -> Tensor:
        return self.accumulated_sum / self.accumulation_count

    def std(self, mean: Tensor) -> Tensor:
        return (
            torch.sqrt(self.accumulated_sum_sq / self.accumulation_count - mean**2)
            + self.std_epsilon
        )

    def accumulate(self, x: Tensor):
        self.accumulation_count += x.shape[0]
        self.accumulated_sum += x.sum(0)
        self.accumulated_sum_sq += (x**2).sum(0)

    @torch.no_grad()
    def forward(self, x: Tensor, accumulate: bool) -> Tensor:
        B, N, D = x.shape
        x = x.reshape(-1, D)
        if accumulate and self.accumulation_count < self.max_accumulations:
            self.accumulate(x)

        mean = self.mean()
        std = self.std(mean)

        x = (x - mean) / std

        return x.view(B, N, D)

    def inverse(self, x: Tensor) -> Tensor:
        mean = self.mean()
        std = self.std(mean)

        return (x * std) + mean

    def __repr__(self):
        mean = self.mean()
        std = self.std(mean)
        return super().__repr__() + f"({mean=}, {std=})"

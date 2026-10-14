import torch
import torch.nn as nn


class SpikeFunction(torch.autograd.Function):
    """Surrogate-gradient spike function for backpropagation through a threshold."""

    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor):
        ctx.save_for_backward(membrane_potential)
        return (membrane_potential >= 0).to(dtype=membrane_potential.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane_potential,) = ctx.saved_tensors
        grad = grad_output * 0.5 * (1.0 - torch.tanh(membrane_potential).pow(2))
        return grad


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with membrane potential decay and spike emission."""

    def __init__(self, threshold: float = 1.0, alpha: float = 0.9, reset_value: float = 0.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.reset_value = reset_value
        self.dtype = dtype

    def forward(self, inputs: torch.Tensor, membrane_potential: torch.Tensor | None = None):
        """
        Apply one discrete-time LIF step.

        Args:
            inputs: Input current tensor of shape [batch, features] or [batch, features, time_steps].
            membrane_potential: Optional existing membrane potential tensor.

        Returns:
            Tuple of (spikes, new_membrane_potential)
        """
        if inputs.dim() < 2:
            raise ValueError("inputs must have at least 2 dimensions")

        if membrane_potential is None:
            membrane_potential = torch.zeros_like(inputs, dtype=self.dtype)
        else:
            membrane_potential = membrane_potential.to(dtype=self.dtype, device=inputs.device)

        membrane_potential = self.alpha * membrane_potential + inputs.to(dtype=self.dtype, device=inputs.device)
        membrane_potential = membrane_potential - self.threshold
        spikes = SpikeFunction.apply(membrane_potential)

        membrane_potential = torch.where(
            spikes > 0,
            torch.full_like(membrane_potential, self.reset_value),
            membrane_potential + self.threshold,
        )

        return spikes, membrane_potential


if __name__ == "__main__":
    neuron = LIFNeuron(threshold=1.0, alpha=0.8)
    sample_inputs = torch.tensor([[0.4, 0.7], [0.9, 0.2]], dtype=torch.float32)
    spikes, membrane = neuron(sample_inputs)
    print("Spikes:", spikes)
    print("Membrane potential:", membrane)

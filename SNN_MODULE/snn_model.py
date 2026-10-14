import torch
import torch.nn as nn

from lif_neuron import LIFNeuron


class SpikingClassifier(nn.Module):
    """A simple feedforward spiking neural network for binary classification."""

    def __init__(self, input_size: int, hidden_size: int = 16, output_size: int = 1):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.lif_hidden = LIFNeuron(threshold=1.0, alpha=0.9)
        self.lif_output = LIFNeuron(threshold=1.0, alpha=0.9)

    def forward(self, spike_trains: torch.Tensor):
        """
        Process spike trains through the network.

        Args:
            spike_trains: Tensor of shape [batch_size, features, time_steps]

        Returns:
            Tuple of (output_logits, hidden_spikes)
        """
        if spike_trains.dim() != 3:
            raise ValueError("spike_trains must have shape [batch_size, features, time_steps]")

        batch_size, input_size, time_steps = spike_trains.shape
        hidden_membrane = None
        output_membrane = None
        hidden_spikes = []
        output_logits = []

        for step in range(time_steps):
            step_inputs = spike_trains[:, :, step]
            current = self.input_layer(step_inputs)
            current = torch.relu(current)

            hidden_spikes_step, hidden_membrane = self.lif_hidden(current, hidden_membrane)
            hidden_spikes.append(hidden_spikes_step)

            output_current = self.hidden_layer(hidden_spikes_step)
            output_current = torch.relu(output_current)
            output_spikes, output_membrane = self.lif_output(output_current, output_membrane)
            output_logits.append(output_spikes)

        hidden_spikes_tensor = torch.stack(hidden_spikes, dim=1)
        output_logits_tensor = torch.stack(output_logits, dim=1)

        return output_logits_tensor, hidden_spikes_tensor


if __name__ == "__main__":
    model = SpikingClassifier(input_size=4, hidden_size=8, output_size=1)
    sample_spikes = torch.rand(2, 4, 5)
    outputs, hidden = model(sample_spikes)
    print("Output spikes shape:", outputs.shape)
    print("Hidden spikes shape:", hidden.shape)

import torch.nn as nn
import torch.nn.utils.rnn


class LSTMEncoder(nn.Module):
    def __init__(
        self, encoder_in_size: int, encoder_out_size: int, num_layers: int = 1, **_
    ):
        """
        Initialize an encoder that applies an LSTM to the inputs.
        :param encoder_in_size: Input dimensionality.
        :param encoder_out_size: Output dimensionality.
        :param num_layers: Number of LSTM layers.
        """
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=encoder_in_size,
            hidden_size=encoder_out_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, idxs: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        idxs_packed = torch.nn.utils.rnn.pack_padded_sequence(
            idxs, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        # Output has zeros for states outside the sequence length
        # h_n automatically finds the last nonzero state
        _, (h_n, _) = self.lstm(idxs_packed)
        # (n_layers=1, bsize, hidden_size)
        # Get the final layer hidden states
        out = h_n[-1]
        # (bsize, encoder_out_size)
        return out

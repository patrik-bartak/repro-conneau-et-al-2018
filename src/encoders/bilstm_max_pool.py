import torch
import torch.nn as nn


class BiLSTMMaxPoolEncoder(nn.Module):
    def __init__(
        self, encoder_in_size: int, encoder_out_size: int, num_layers: int = 1, **_
    ):
        """
        Initialize an encoder that applies a bidirectional LSTM to the inputs and max-pools the outputs.
        :param encoder_in_size: Input dimensionality.
        :param encoder_out_size: Output dimensionality.
        :param num_layers: Number of LSTM layers.
        """
        super(BiLSTMMaxPoolEncoder, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=encoder_in_size,
            hidden_size=encoder_out_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, idxs: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        idxs_packed = torch.nn.utils.rnn.pack_padded_sequence(
            idxs, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        # Output has zeros for states outside the sequence length
        # h_n automatically finds the last nonzero state
        out, (_, _) = self.bilstm(idxs_packed)
        # (N, L, D * H_out)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # Pooling features
        pooled, _ = torch.max(unpacked, dim=1)
        return pooled

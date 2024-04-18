import torch
import torch.nn as nn


class BiLSTMMaxPoolEncoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size, num_layers=1, **_):
        super(BiLSTMMaxPoolEncoder, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=encoder_in_size,
            hidden_size=encoder_out_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, idxs, lens):
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

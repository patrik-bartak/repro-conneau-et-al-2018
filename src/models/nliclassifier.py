import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR


def get_mlp_layers_from_dims(mlp_dims):
    layers = []
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i < len(mlp_dims) - 2:
            layers.append(nn.ReLU())
    return layers


class NLIClassifier(pl.LightningModule):
    def __init__(self, mlp_dims, encoder_type, embedding_mat, **kwargs):
        super(NLIClassifier, self).__init__()
        self.encoder = encoder_type(**kwargs)
        self.embedding = nn.Embedding.from_pretrained(embedding_mat, freeze=True)
        self.mlp = nn.Sequential(*get_mlp_layers_from_dims(mlp_dims))
        self.lr = kwargs["lr"]
        self.lr_decay = kwargs["lr_decay"]
        self.save_hyperparameters(ignore=["embedding_mat"])

    def forward(self, u, v, l_u, l_v):
        # Embed token idxs
        u = self.embedding(u)
        v = self.embedding(v)
        # Create sentence representations
        u = self.encoder(u, l_u)
        v = self.encoder(v, l_v)
        # Concatenate representations
        h = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        # Pass through MLP with output dim 3
        out = self.mlp(h)
        return out

    def common_step(self, batch):
        u, v, l_u, l_v, t = batch
        out = self.forward(u, v, l_u, l_v)
        return F.cross_entropy(out, t)

    def training_step(self, batch, _):
        loss = self.common_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        assert not loss.isnan().any(), "Some loss values are nan"
        return loss

    def validation_step(self, batch, _):
        loss = self.common_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_acc", loss, on_epoch=True, prog_bar=True, logger=True)
        assert not loss.isnan().any(), "Some loss values are nan"
        return loss

    def test_step(self, batch, _):
        loss = self.common_step(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_acc", loss, on_epoch=True, prog_bar=True, logger=True)
        assert not loss.isnan().any(), "Some loss values are nan"
        return loss

    def on_save_checkpoint(self, checkpoint):
        # Do not save glove embeddings to reduce checkpoint size
        del checkpoint["state_dict"]["embedding.weight"]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # scheduler_1 = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=1)
        scheduler_2 = ExponentialLR(optimizer, gamma=self.lr_decay)
        return {"optimizer": optimizer, "lr_scheduler": scheduler_2}
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler_2,
        #         "monitor": "val_loss",
        #         "frequency": 1,
        #     },
        # }

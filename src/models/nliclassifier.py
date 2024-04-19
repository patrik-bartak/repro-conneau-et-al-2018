import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau


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
        # To step both schedulers easily in the validation epoch
        self.automatic_optimization = False
        self.exp_scheduler = None
        self.rlr_scheduler = None
        self.step_val_accs = []

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

    def common_step(self, batch, compute_acc=False):
        u, v, l_u, l_v, t = batch
        preds = self.forward(u, v, l_u, l_v)
        loss = F.cross_entropy(preds, t)
        if compute_acc:
            acc = (torch.argmax(preds, dim=1) == t).float().mean()
            return loss, acc
        else:
            return loss

    def training_step(self, batch, _):
        opt = self.optimizers()
        opt.zero_grad()

        loss = self.common_step(batch)

        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        assert not loss.isnan().any(), "Some loss values are nan"
        return loss

    def validation_step(self, batch, _):
        loss, acc = self.common_step(batch, compute_acc=True)
        # Save acc of validation step to compute epoch val acc later for early stopping scheduler
        self.step_val_accs.append(acc)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        assert not loss.isnan().any(), "Some loss values are nan"
        return loss

    def on_validation_epoch_end(self):
        self.exp_scheduler.step()
        # compute epoch val acc later for early stopping scheduler
        epoch_val_acc = torch.stack(self.step_val_accs).mean()
        self.rlr_scheduler.step(epoch_val_acc)
        self.step_val_accs.clear()

    def test_step(self, batch, _):
        loss, acc = self.common_step(batch, compute_acc=True)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        assert not loss.isnan().any(), "Some loss values are nan"
        return loss

    def on_save_checkpoint(self, checkpoint):
        # Do not save glove embeddings to reduce checkpoint size
        del checkpoint["state_dict"]["embedding.weight"]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.exp_scheduler = ExponentialLR(optimizer, gamma=self.lr_decay)
        self.rlr_scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.2, patience=1)
        return optimizer

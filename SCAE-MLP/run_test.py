import os

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch_scae.general_utils import Invert
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, ImageFolder
from torchvision import transforms
from torch_scae_experiments.wdata.train import get_mdl_obj


class TestModel(pl.LightningModule):

    def __init__(self, scae_model):
        super(TestModel, self).__init__()
        self.scae = scae_model

    def forward(self, x):
        res = self.scae(x)
        return res.pred_out

    def training_step(self, batch, batch_nb):
        pass
        return {}

    def calculate_acc(self, pred, labels):
        preds = torch.argmax(pred, axis=1)
        return (preds==labels).float().mean()

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)

        accuracy = self.calculate_acc(y_hat, y)
        return {'val_loss': F.cross_entropy(y_hat, y), 'accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):

        transforms = torchvision.transforms.Compose([
              torchvision.transforms.Grayscale(),
            Invert(),
            torchvision.transforms.Resize(60),
            torchvision.transforms.ToTensor()                                       
        ])

        data_train = ImageFolder("dataset/train", transform=transforms)
        data_val = ImageFolder("dataset/test", transform=transforms)
        data_test = ImageFolder("dataset/test", transform=transforms)

        self.train_dataset = data_train
        self.val_dataset = data_val
        self.test_dataset = data_test

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        accuracy = self.calculate_acc(y_hat, y)
        return {'test_loss': F.cross_entropy(y_hat, y), 'accuracy': accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss.item(), 'test_accuracy': avg_acc.item()}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=32,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=32,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=32,
                          num_workers=4)


if __name__ == "__main__":
    scae_model = get_mdl_obj()

    scae_model.load_state_dict(torch.load("scae-mle.pth"))
    mdl = TestModel(scae_model)
    tester = pl.Trainer()  
    tester.test(mdl)

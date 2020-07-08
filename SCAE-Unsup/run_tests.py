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




class SCAEUnsupTest(pl.LightningModule):

    def __init__(self, scae):
        super(SCAEUnsupTest, self).__init__()
        self.scae = scae
        self.scae.eval()
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )
        
        # self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        
        res = self.scae(x)
        out = self.mlp(res.capsule_probs)
        return out

    def training_step(self, batch, batch_nb):
        # REQUIRED
        pass
        return {}

    def calculate_acc(self, pred, labels):
        preds = torch.argmax(pred, axis=1)
        return (preds==labels).float().mean()

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)

        accuracy = self.calculate_acc(y_hat, y)
        return {'val_loss': F.cross_entropy(y_hat, y), 'accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
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
        # REQUIRED
        return DataLoader(self.train_dataset,
                          batch_size=32,
                          num_workers=4)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset,
                          batch_size=32,
                          num_workers=4)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.test_dataset,
                          batch_size=32,
                          num_workers=4)
   

if __name__ == "__main__":
    scae_model = get_mdl_obj()

    model_unsup_test = SCAEUnsupTest(scae_model)
    model_unsup_test.load_state_dict(torch.load("scae-unsup.pth", map_location=torch.device('cpu')))

    tester = pl.Trainer()  
    tester.test(model_unsup_test)

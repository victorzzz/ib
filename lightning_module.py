import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import constants as cnts
import lightning_datamodule as ldm

class PyTorchTradingModel(torch.nn.Module):
    def __init__(self, num_features = ldm.FEATURES, num_classes = ldm.CLASSES, hidden_sizes:list[int] = [512, 256, 128, 64]):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, hidden_sizes[0]),
            torch.nn.BatchNorm1d(hidden_sizes[0]),
            torch.nn.ReLU(),
            
            # 2nd hidden layer
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.BatchNorm1d(hidden_sizes[1]),
            torch.nn.ReLU(),
            
            # 3d hidden layer
            torch.nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            torch.nn.BatchNorm1d(hidden_sizes[2]),
            torch.nn.ReLU(),            
            
            # 4th hidden layer
            torch.nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            torch.nn.BatchNorm1d(hidden_sizes[3]),
            torch.nn.ReLU(),               
            
            # output layer
            torch.nn.Linear(hidden_sizes[3], num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits
    
class LightningTradingModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        
        self.example_input_array = torch.Tensor(5, ldm.FEATURES) # example batch

        self.learning_rate = learning_rate
        self.model = model

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=ldm.CLASSES)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=ldm.CLASSES)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=ldm.CLASSES)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return opt
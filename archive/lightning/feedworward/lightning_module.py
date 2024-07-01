import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import constants as cnts
import lightning_datamodule as ldm
import logging

class PyTorchTradingModel(torch.nn.Module):
    def __init__(self, num_features = ldm.FEATURES, num_classes = ldm.CLASSES, hidden_sizes:list[int] = [1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128]):
        super().__init__()

        self.all_layers = torch.nn.Sequential()
        
        for i in range(0, len(hidden_sizes)):
            if i == 0:
                self.all_layers.append(torch.nn.Linear(num_features, hidden_sizes[i]))
            else:
                self.all_layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.all_layers.append(torch.nn.BatchNorm1d(hidden_sizes[i]))
            self.all_layers.append(torch.nn.ReLU())
        
        self.all_layers.append(torch.nn.Linear(hidden_sizes[-1], num_classes))
        
        self.loss_weight = torch.tensor([0.2, 5.0, 5.0]).to("cuda")
        
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
        
        self.train_acc_non_avg = torchmetrics.Accuracy(task="multiclass", num_classes=ldm.CLASSES, average=None)
        self.val_acc_non_avg = torchmetrics.Accuracy(task="multiclass", num_classes=ldm.CLASSES, average=None)
        self.test_acc_non_avg = torchmetrics.Accuracy(task="multiclass", num_classes=ldm.CLASSES, average=None)
        
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=ldm.CLASSES)


    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels, weight=self.model.loss_weight)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log("ta", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)

        train_acc_non_avg_result = self.train_acc_non_avg(predicted_labels, true_labels)
        self.log("ta0", train_acc_non_avg_result[0], prog_bar=True, on_epoch=True, on_step=False)
        self.log("ta1", train_acc_non_avg_result[1], prog_bar=True, on_epoch=True, on_step=False)
        self.log("ta2", train_acc_non_avg_result[2], prog_bar=True, on_epoch=True, on_step=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
                
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

        val_acc_non_avg_result = self.val_acc_non_avg(predicted_labels, true_labels)
        self.log("va0", val_acc_non_avg_result[0], prog_bar=True)
        self.log("va1", val_acc_non_avg_result[1], prog_bar=True)
        self.log("va2", val_acc_non_avg_result[2], prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

        self.confmat.update(predicted_labels, true_labels)

        test_acc_non_avg_result = self.test_acc_non_avg(predicted_labels, true_labels)
        self.log("test_a0", test_acc_non_avg_result[0])
        self.log("test_a1", test_acc_non_avg_result[1])
        self.log("test_a2", test_acc_non_avg_result[2])

    def on_test_end(self):
        # Log the confusion matrix at the end of testing
        conf_matrix = self.confmat.compute()
        logging.info(f"Confusion Matrix: {conf_matrix}")
        self.confmat.reset()


    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt
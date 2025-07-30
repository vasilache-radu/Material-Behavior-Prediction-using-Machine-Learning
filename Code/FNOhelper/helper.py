from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from fsspec import Callback
import json
from sklearn.metrics import r2_score
from neuralop.losses import LpLoss, H1Loss  # Import LpLoss from neuralop.losses

class FNODataset(Dataset):
    def __init__(self, inputs, outputs, device='cpu'):
        self.inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2).to(device)
        self.outputs = torch.tensor(outputs, dtype=torch.float32, requires_grad= True).permute(0, 3, 1, 2).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'x': self.inputs[idx], 'y': self.outputs[idx]}

class FNODataset2(Dataset):
    def __init__(self, inputs, outputs, device='cpu', normalizerInput=None, normalizerOutput=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True).to(device)
        self.outputs = torch.tensor(outputs, dtype=torch.float32, requires_grad= True).to(device)

        if normalizerInput is not None and normalizerOutput is not None:
            self.inputs = normalizerInput.encode(self.inputs)
            self.outputs = normalizerOutput.encode(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'x': self.inputs[idx], 'y': self.outputs[idx]}
    
# BEGIN: Initialize parameters with Xavier initialization
def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# END: Initialize parameters with Xavier initialization

class BehaviourCallback(Callback):
    def __init__(self, model, val_loader, data_processor, save_path, device='cpu'):
        super().__init__()
        self.model = model
        self.val_loader = val_loader
        self.data_processor = data_processor
        self.save_path = save_path
        self.device = device

        self.train_losses = []
        self.val_losses = []
        self.mean_lplosses = []  
        self.mean_lplosses_x = []
        self.mean_lplosses_y = []
        
        self.best_val_loss = np.inf
        self.patience = 20
        
        # Define the loss function. This thing must be investigated further
        self.loss_fn = LpLoss(d=3, p=2, reductions='sum', reduce_dims=[0])

    def on_epoch_end(self, epoch, train_err, avg_loss):
        print(f"Epoch {epoch}: Train Error: {train_err}")
        self.train_losses.append(train_err)
        if(epoch==199):        
            metrics = {
                "last_train_loss": self.train_losses[-1],
                "last_val_loss": self.val_losses[-1],
                "last_mean_lploss": (self.mean_lplosses[-1]).item(),
                "last_mean_lploss_x": (self.mean_lplosses_x[-1]).item(),
                "last_mean_lploss_y": (self.mean_lplosses_y[-1]).item()
            }
            with open(f'{self.save_path}/metrics.json', 'w') as f:
                json.dump(metrics, f)
        
    def on_val_epoch_end(self, errors, sample, out):
        val_loss_key = next(key for key in errors.keys() if 'val' in key)  # Just for the first key with 'val' in it
        current_val_loss = errors[val_loss_key]
        self.val_losses.append(current_val_loss)

        self.compute_accuracy()
        # self.checkImprovement(current_val_loss)
        
    def checkImprovement(self, current_val_loss):
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.counter = 0
        else:
            if self.counter == 0:
                torch.save(self.model.state_dict(), f'{self.save_path}/best_model.pth')  # Save the best model for minimal validation loss  

            self.counter += 1
            print(f"Warning: Model not improving for {self.counter} epochs.")
            print(f"Best Validation Loss: {self.best_val_loss}")
                
            if self.counter >= self.patience:
                metrics = {
                "last_train_loss": self.train_losses[-1],
                "last_val_loss": self.val_losses[-1],
                "last_mean_lploss": (self.mean_lplosses[-1]).item(),
                "last_mean_lploss_x": (self.mean_lplosses_x[-1]).item(),
                "last_mean_lploss_y": (self.mean_lplosses_y[-1]).item()
                }
                with open(f'{self.save_path}/metrics.json', 'w') as f:
                    json.dump(metrics, f)
                raise Exception("Early stopping.")
    
    def compute_accuracy(self):
        rel_error = 0
        rel_error_x = 0
        rel_error_y = 0
        nr_samples = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch['x'], batch['y']
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                out, y = out.to('cpu'), y.to('cpu')
                rel_error += self.loss_fn(out, y)
                rel_error_x += self.loss_fn(out[:, 0, :, :].unsqueeze(1), y[:, 0, :, :].unsqueeze(1))
                rel_error_y += self.loss_fn(out[:, 1, :, :].unsqueeze(1), y[:, 1, :, :].unsqueeze(1))
                nr_samples += x.shape[0]
                
            rel_error /= nr_samples
            rel_error_x /= nr_samples
            rel_error_y /= nr_samples
        self.mean_lplosses.append(rel_error)
        self.mean_lplosses_x.append(rel_error_x)
        self.mean_lplosses_y.append(rel_error_y)

def plot(save_path, file_name, behaviour_callback):
    train_losses, val_losses = behaviour_callback.train_losses, behaviour_callback.val_losses
    mean_lplosses = behaviour_callback.mean_lplosses
    mean_lplosses_x = behaviour_callback.mean_lplosses_x
    mean_lplosses_y = behaviour_callback.mean_lplosses_y

    print("Mean Relative Error: ", mean_lplosses[-1])
    print("Mean Relative Error x: ", mean_lplosses_x[-1])
    print("Mean Relative Error y: ", mean_lplosses_y[-1])

    # Plot Training Error
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss per batch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/{file_name}/train_loss.png')
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses, label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss per sample')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/{file_name}/val_loss.png')
    plt.show()

    # Plot Relative Errors using LpLoss
    plt.figure(figsize=(10, 5))
    plt.plot(mean_lplosses, label='Mean LpLoss', color='blue')
    plt.plot(mean_lplosses_x, label='Mean LpLoss x', color='green')
    plt.plot(mean_lplosses_y, label='Mean LpLoss y', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    plt.title('Overall, x and y Relative Errors')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/{file_name}/rel_errors.png')
    plt.show()

# Does not work yet
def plot_samples(save_path, test_loaders, model, data_processor, nr_samples=5, device='cpu'):
    test_samples = test_loaders["val"].dataset
    fig = plt.figure(figsize=(7, 7))
    
    for index in range(nr_samples):
        data = test_samples[index]
        #data = data_processor.preprocess(data)
        x = data['x']
        y = data['y']
        print(x.shape)
        model = model.to(device)
        out = model(x)
        #out, data = data_processor.postprocess(out, data)
        print('5')
        
        print('b')
        x = data['x']
        y = data['y']
        print('c')
        x = x.cpu()
        y = y.cpu()
        out = out.cpu()
        print('2')
        relative_error = torch.sum((y - out) ** 2) / torch.sum(y ** 2)
        relative_accuracy = 1 - relative_error
        # l2loss = LpLoss(d=2, p=2, reductions='sum', reduce_dims=[0, 1])
        # diff = l2loss(out, y)
        print('3')
        info = {
            'index': index,
            # 'diff': diff.item(),
            'relative_accuracy': relative_accuracy.item()
        }
        with open(f'{save_path}/sample_info.json', 'w') as f:
            json.dump(info, f)
        
        print('4')
        x = x.detach().numpy()
        out = out.detach().numpy()
        y = y.detach().numpy()

        ax = fig.add_subplot(nr_samples, 5, index * 5 + 1)
        ax.imshow(x[0], cmap='gray')
        if index == 0: 
            ax.set_title('x')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(nr_samples, 5, index * 5 + 2)
        ax.imshow(y[0])
        if index == 0: 
            ax.set_title('y[0]')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(nr_samples, 5, index * 5 + 3)
        ax.imshow(out[0, 0])
        if index == 0: 
            ax.set_title('y_pred[0]')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(nr_samples, 5, index * 5 + 4)
        ax.imshow(y[1])
        if index == 0: 
            ax.set_title('y[1]')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(nr_samples, 5, index * 5 + 5)
        ax.imshow(out[0, 1])
        if index == 0: 
            ax.set_title('y_pred[1]')
        plt.xticks([], [])
        plt.yticks([], [])
        print('5')
        
    fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{save_path}/samples.png')
    plt.show()
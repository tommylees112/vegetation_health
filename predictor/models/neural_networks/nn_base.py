import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
from tqdm import tqdm
import numpy as np

from ..base import ModelBase, DataTuple


class NNBase(ModelBase):
    """The base for neural networks models
    """

    def __init__(self, model, data=Path('data'), arrays=Path('data/processed/arrays'),
                 hide_vegetation=False):
        super().__init__(data, arrays, hide_vegetation)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # for reproducability
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model

    def train(self, num_epochs=100, patience=3, batch_size=32, learning_rate=1e-3):
        """Train the neural network

        Parameters
        ----------
        num_epochs: int
            The maximum number of epochs to train the model for
        patience: int
            If no improvement is seen in the validation set for `patience` epochs,
            training is stopped to prevent overfitting
        batch_size: int
            The batch size to use
        learning_rate: float
            The learning rate to use when updating model parameters
        """
        train_data = self.load_tensors(mode='train')

        # split the data into a training and validation set
        total_size = train_data.x.shape[0]
        val_size = total_size // 10  # 10 % for validation
        train_size = total_size - val_size
        print(f'After split, training on {train_size} examples, '
              f'validating on {val_size} examples')
        train_dataset, val_dataset = random_split(TensorDataset(train_data.x, train_data.y.unsqueeze(1)),
                                                  (train_size, val_size))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam([pam for pam in self.model.parameters()],
                                     lr=learning_rate)

        epochs_without_improvement = 0
        best_loss = np.inf

        for epoch in range(num_epochs):
            self.model.train()

            epoch_train_loss = 0
            num_train_batches = 0

            epoch_val_loss = 0
            num_val_batches = 0

            for train_x, train_y in tqdm(train_dataloader):
                optimizer.zero_grad()
                pred_y = self.model(train_x)

                loss = F.smooth_l1_loss(pred_y, train_y)
                loss.backward()
                optimizer.step()

                num_train_batches += 1
                epoch_train_loss += loss.sqrt().item()

            self.model.eval()
            with torch.no_grad():
                for val_x, val_y in tqdm(val_dataloader):
                    val_pred_y = self.model(val_x)
                    val_loss = F.mse_loss(val_pred_y, val_y)

                    num_val_batches += 1
                    epoch_val_loss += val_loss.sqrt().item()

            epoch_train_loss /= num_train_batches
            epoch_val_loss /= num_val_batches

            print(f'Epoch {epoch} - Training RMSE: {epoch_train_loss}, '
                  f'Validation RMSE: {epoch_val_loss}')

            if epoch_val_loss < best_loss:
                best_state = self.model.state_dict()
                best_loss = epoch_val_loss

                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == patience:
                    self.model.load_state_dict(best_state)
                    print('Early stopping!')
                    return
        self.model.load_state_dict(best_state)

    def predict(self, batch_size=64):
        test_data = self.load_tensors(mode='test')

        test_dataloader = DataLoader(TensorDataset(test_data.x, test_data.y),
                                     batch_size=batch_size)

        output_preds, output_true = [], []

        self.model.eval()
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                output_preds.append(self.model(test_x).squeeze(1).cpu().numpy())
                output_true.append(test_y.cpu().numpy())
        return np.concatenate(output_true), np.concatenate(output_preds)

    def load_tensors(self, mode='train'):
        data = self.load_arrays(mode)

        return DataTuple(
                latlon=data.latlon,
                years=data.years,
                x=torch.as_tensor(data.x, device=self.device).float(),
                y=torch.as_tensor(data.y, device=self.device).float())

    def save_model(self, name="model.pt"):
        """save the model's state_dict"""
        savedir = self.data_path / self.model_name
        if not savedir.exists(): savedir.mkdir()

        # save with .pt extension
        if not '.pt' in name: name += '.pt'
        print(f'Saving predictions to {savedir / name}')
        torch.save(self.model, savedir / name)

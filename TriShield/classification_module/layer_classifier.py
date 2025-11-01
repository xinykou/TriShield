import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


import torch
import torch.nn.functional as F


class LayerClassifierTorch(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)

    def forward(self, x):
        return self.sigmoid(self.linear(x))

    def train_model(self, pos_tensor, neg_tensor, n_epoch=100, batch_size=64, lr=0.01):
        # 构造数据
        X = torch.vstack([pos_tensor, neg_tensor]).float()
        y = torch.cat(
            [torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0))]
        ).float()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in tqdm(range(n_epoch)):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(
                    self.device
                ).unsqueeze(1)
                optimizer.zero_grad()
                output = self.forward(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, tensor):
        self.eval()
        with torch.no_grad():
            tensor = tensor.float().to(self.device)
            return (self.forward(tensor) > 0.5).float().cpu()

    def predict_logit(self, tensor):
        self.eval()
        with torch.no_grad():
            tensor = tensor.float().to(self.device)
            return self.forward(tensor)

    def evaluate_testacc(self, pos_tensor=None, neg_tensor=None):

        if pos_tensor is None:
            X = neg_tensor
            y_true = torch.zeros(neg_tensor.size(0))
        elif neg_tensor is None:
            X = pos_tensor
            y_true = torch.ones(pos_tensor.size(0))
        else:
            X = torch.vstack([pos_tensor, neg_tensor])
            y_true = torch.cat(
                [torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0))]
            )
        y_pred = self.predict(X)
        return (y_pred.squeeze() == y_true).float().mean().item()

    def get_weights_bias(self):
        return self.linear.weight.data.clone().to(
            self.device
        ), self.linear.bias.data.clone().to(self.device)

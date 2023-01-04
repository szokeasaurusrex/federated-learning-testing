import torch
from torch.utils.data import DataLoader

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ClientUpdate:
    def __init__(self, updated_local_state, global_state, local_dataset_size):
        self.updated_state = updated_local_state
        self.local_dataset_size = local_dataset_size
        self.update = updated_local_state.copy()
        for layer in self.update:
            self.update[layer] -= global_state[layer]

class FederatedClient:
    def __init__(self, local_data, local_epochs, local_batches, device):
        self.dataset_size = len(local_data)
        self.data_loader = DataLoader(local_data, batch_size=len(local_data) // local_batches)
        self.epochs = local_epochs
        self.model = NeuralNetwork().to(device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-1)
        self.device = device
    
    def train(self, training_loss_fn=None):
        training_loss_fn = training_loss_fn if training_loss_fn is not None else self.loss_fn
        self.model.train()
        for (X, y) in self.data_loader:
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = training_loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def client_update(self, state_dict, *training_args):
        self.model.load_state_dict(state_dict)
        for _ in range(self.epochs):
            self.train(*training_args)
        return ClientUpdate(self.model.state_dict(), state_dict, self.dataset_size)


class LabelFlipMaliciousClient:
    def __init__(self, client: FederatedClient):
        self.underlying_client = client
    
    def loss_fn(self, predicted, actual):
        return self.underlying_client.loss_fn(predicted, torch.randint(10, size=actual.size()).to(self.underlying_client.device))
    
    def train(self):
        self.underlying_client.train(self.loss_fn)
    
    def client_update(self, state_dict):
        return self.underlying_client.client_update(state_dict, self.loss_fn)
from federated_client import FederatedClient, ClientUpdate
import torch

class RandomClient(FederatedClient):
    def __init__(self, device):
        self.device = device

    def train(self):
        pass
    
    def client_update(self, state_dict):
        new_state = state_dict.copy()
        for layer, weights in state_dict.items():
            # # Randomize new state to uniform values between -0.2 and 0.2.
            # new_state[layer] = torch.rand(weights.size()).to(self.device) * 20.0 - 10.0

            # Randomize weights to Gaussian noise with mean, standard deviation matching the layer
            new_state[layer] = torch.randn(weights.size()).to(self.device) * torch.std(weights) + torch.mean(weights)
        return ClientUpdate(new_state, state_dict, 100)  # TODO: Allow local dataset size to be configured
import constants
import federated_client
import random
import torch
import math

class FedAvgAggregator:
    def aggregate(self, global_model_state, client_updates):
        new_global_state = global_model_state.copy()
        
        global_dataset_size = sum(client_update.local_dataset_size for client_update in client_updates)

        for client_update in client_updates:
            for layer in new_global_state:
                new_global_state[layer] += client_update.local_dataset_size * client_update.update[layer] / global_dataset_size
        
        return new_global_state

class StaticNormClipAggregator:
    def __init__(self, clip_threshold):
        self.clip_threshold = clip_threshold
    
    def update_norm(self, update):
        return math.sqrt(sum(torch.linalg.norm(update[layer]) ** 2 for layer in update))
    
    def aggregate(self, global_model_state, client_updates):
        new_global_state = global_model_state.copy()        
        global_dataset_size = sum(client_update.local_dataset_size for client_update in client_updates)

        for client_update in client_updates:
            norm = self.update_norm(client_update.update)
            norm_clipping_factor = min(1, self.clip_threshold / norm)
            for layer in new_global_state:
                new_global_state[layer] += norm_clipping_factor * client_update.local_dataset_size * client_update.update[layer] / global_dataset_size
        
        return new_global_state

class FederatedServer:
    def __init__(self, clients, client_fraction, aggregator):
        self.clients = clients
        self.aggregator = aggregator
        self.clients_per_epoch = int(len(clients) * client_fraction)
        self.global_model = federated_client.NeuralNetwork().to(constants.device)
        self.global_model_state = self.global_model.state_dict()
        self.epoch = 0        

    def train(self, epochs):
        """Train given number of epochs"""
        for _ in range(epochs):
            client_updates = [client.client_update(self.global_model_state) for client in random.sample(self.clients, self.clients_per_epoch)]
            self.global_model_state = self.aggregator.aggregate(self.global_model_state, client_updates)
            self.global_model.load_state_dict(self.global_model_state)
            self.epoch += 1

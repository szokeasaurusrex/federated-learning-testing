import constants
import federated_client
import random
import torch
import math


def update_norm(update):
    return math.sqrt(sum(torch.linalg.norm(update[layer]) ** 2 for layer in update))

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
       
    def aggregate(self, global_model_state, client_updates):
        new_global_state = global_model_state.copy()        
        global_dataset_size = sum(client_update.local_dataset_size for client_update in client_updates)

        for client_update in client_updates:
            norm = update_norm(client_update.update)
            if math.isnan(norm):
                continue

            norm_clipping_factor = min(1, self.clip_threshold / norm)
            for layer in new_global_state:
                new_global_state[layer] += norm_clipping_factor * client_update.local_dataset_size * client_update.update[layer] / global_dataset_size
        
        return new_global_state

class DynamicNormClipAggregator:
    def __init__(self, initial_clip_threshold):
        self.static_aggregator = StaticNormClipAggregator(initial_clip_threshold)
        self.current_threshold = initial_clip_threshold
    
    def aggregate(self, global_model_state, client_updates):
        update_norms = torch.tensor([update_norm(update.update) for update in client_updates])
        dynamic_threshold = torch.quantile(update_norms, constants.dynamic_clip_probability_threshold)
        if dynamic_threshold < self.current_threshold:
            self.static_aggregator = StaticNormClipAggregator(dynamic_threshold)
            self.current_threshold = dynamic_threshold
        
        return self.static_aggregator.aggregate(global_model_state, client_updates)

class FederatedServer:
    @staticmethod
    def constant_lr_scheduler(_):
        return constants.learning_rate
    
    @staticmethod
    def one_over_n_lr_scheduler(epoch):
        return constants.learning_rate / (epoch + 1)
    
    @staticmethod
    def exponential_lr_scheduler_generator(multiplier):
        return lambda epoch: constants.learning_rate * (multiplier ** epoch)

    def __init__(self, clients, client_fraction, aggregator, lr_scheduler):
        self.clients = clients
        self.aggregator = aggregator
        self.clients_per_epoch = int(len(clients) * client_fraction)
        self.global_model = federated_client.NeuralNetwork().to(constants.device)
        self.global_model_state = self.global_model.state_dict()
        self.epoch = 0
        self.scheduler = lr_scheduler

    def train(self, epochs):
        """Train given number of epochs"""
        for _ in range(epochs):
            lr = self.scheduler(self.epoch)
            client_updates = [client.client_update(self.global_model_state, lr) for client in random.sample(self.clients, self.clients_per_epoch)]
            self.global_model_state = self.aggregator.aggregate(self.global_model_state, client_updates)
            self.global_model.load_state_dict(self.global_model_state)
            self.epoch += 1

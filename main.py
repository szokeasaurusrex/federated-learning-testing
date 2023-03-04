import federated_client, federated_server
import random_client
import random
import torch
import torch.backends.mps
import torchvision
import random
import itertools
import constants
import json
import csv
import datetime
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, random_split

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

def iid_clients(training_data, num_clients, device):
    clients = []
    for client_data in random_split(training_data, [len(training_data) // num_clients] * num_clients):
        clients.append(federated_client.FederatedClient(client_data, 1, 60, device))
    return clients

def non_iid_clients(training_data, num_clients, device):
    clients = []
    sorted_data = sorted(training_data, key=lambda example: example[1])
    num_shards = num_clients * 2
    shard_size = len(sorted_data) // num_shards
    shards = [sorted_data[i:i + len(sorted_data) // (num_clients * 2)] for i in range(0, len(sorted_data), shard_size)]
    random.shuffle(shards)
    for shards in zip(shards[0::2], shards[1::2]):
        X = torch.stack([data[0] for data in itertools.chain(*shards)])
        y = torch.tensor([data[1] for data in itertools.chain(*shards)])
        client_data = TensorDataset(X, y)
        clients.append(federated_client.FederatedClient(client_data, 1, 60, device))
    return clients

def make_shards(data, num_shards):
    sorted_data = sorted(data, key=lambda example: example[1])
    shard_size = len(sorted_data) // num_shards
    shards = [sorted_data[i:i + len(sorted_data) // num_shards] for i in range(0, len(sorted_data), shard_size)]
    random.shuffle(shards)

    return shards

def collusive_attack_clients(shards, num_clients, adversarial_fraction, device):
    clients = []

    # Shuffle data for adversarial clients
    num_adversarial_clients = int(adversarial_fraction * num_clients)
    if num_adversarial_clients > 0:
        adversarial_data = list(itertools.chain(*shards[:2 * num_adversarial_clients]))
        adversarial_X = torch.stack([data[0] for data in adversarial_data])
        adversarial_y = torch.tensor([data[1] for data in adversarial_data])
        adversarial_dataset = TensorDataset(adversarial_X, adversarial_y)
        for client_data in random_split(adversarial_dataset, [len(adversarial_dataset) // num_adversarial_clients] * num_adversarial_clients):
            clients.append(federated_client.GradientAscentMaliciousClient(federated_client.FederatedClient(client_data, 1, 60, device)))

    # Remaining benign clients get non-iid shards
    for shard in zip(shards[2 * num_adversarial_clients::2], shards[2 * num_adversarial_clients + 1::2]):
        X = torch.stack([data[0] for data in itertools.chain(*shard)])
        y = torch.tensor([data[1] for data in itertools.chain(*shard)])
        client_data = TensorDataset(X, y)
        clients.append(federated_client.FederatedClient(client_data, 1, 60, device))
    return clients

def random_iid_clients(training_data, num_clients, num_random_clients, device):
    clients = iid_clients(training_data, num_clients, device)
    for i in range(num_random_clients):
        clients[i] = random_client.RandomClient(device)
    return clients

def run_experiments_with_aggregator(shards, aggregator_getter, test_data, val_data):
    num_clients = constants.num_clients
    adversarial_fraction = constants.adversarial_fraction
    device = constants.device

    benign_clients = collusive_attack_clients(shards, num_clients, 0,  device)

    non_collusive_clients = collusive_attack_clients(shards, num_clients, 0, device)
    for i, client in enumerate(non_collusive_clients[:int(num_clients * adversarial_fraction)]):
        non_collusive_clients[i] = federated_client.GradientAscentMaliciousClient(client)
    
    random_attack_clients = collusive_attack_clients(shards, num_clients, 0, device)
    for i, client in enumerate(random_attack_clients[:int(num_clients * adversarial_fraction)]):
        random_attack_clients[i] = random_client.RandomClient(device)
    
    collusive_clients = collusive_attack_clients(shards, num_clients, adversarial_fraction, device)

    batch_size = 64

    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    runs = {
        'collusive': collusive_clients,
        'random': random_attack_clients,
        'non-collusive': non_collusive_clients,
        'benign': benign_clients,
    }

    results = dict()

    for run_title, clients in runs.items():
        aggregator = aggregator_getter()
        results[run_title] = dict()
        print(f'+++++++ {run_title} ++++++++')
        server = federated_server.FederatedServer(clients, 0.1, aggregator)
        for _ in range(40):
            server.train(20)
            print(f'Epoch {server.epoch}')
            results[run_title][server.epoch] = test(val_dataloader, server.global_model, constants.loss_fn, constants.device)
    return results

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # Attempt to use Metal hardware acceleration
    print(f'Using {device} device')

    training_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    training_data, val_data = random_split(training_data, [0.8, 0.2])

    num_clients = 100
    adversarial_fraction = 0.25

    shards = make_shards(training_data, num_clients * 2)
    
    aggregator_getters = {
        'fed_avg': lambda: federated_server.FedAvgAggregator(),
        'static_norm_clip': lambda: federated_server.StaticNormClipAggregator(1),
        'dynamic_norm_clip': lambda: federated_server.DynamicNormClipAggregator(1),
    }

    results = dict()

    for aggregator_title, aggregator_getter in aggregator_getters.items():
        print(f'======================{aggregator_title}=======================')
        results[aggregator_title] = run_experiments_with_aggregator(shards, aggregator_getter, test_data, val_data)
    
    timestamp = datetime.datetime.utcnow().isoformat()

    with open(f'output_{timestamp}.json', 'w') as f:
        json.dump(results, f)
    
    # with open(f'output_{timestamp}.csv') as f:
    #     csv_writer = csv.DictWriter() # TODO: Continue here

    # Probably can use state_dict to average models
    print('done')

if __name__ == '__main__':
    main()
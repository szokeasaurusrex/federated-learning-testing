import federated_client, federated_server
import random_client
import random
import torch
import torch.backends.mps
import torchvision
import random
import itertools
import constants
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

def random_iid_clients(training_data, num_clients, num_random_clients, device):
    clients = iid_clients(training_data, num_clients, device)
    for i in range(num_random_clients):
        clients[i] = random_client.RandomClient(device)
    return clients

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
    clients = non_iid_clients(training_data, num_clients, device)

    for i in range(num_clients // 5):
        clients[i] = random_client.RandomClient(device)
    
    # for i in range(num_clients // 3):
    #     clients.append(random_client.RandomClient(device))

    batch_size = 64

    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    

    loss_fn = torch.nn.CrossEntropyLoss()

    server = federated_server.FederatedServer(clients, 0.1, federated_server.StaticNormClipAggregator(1))

    for _ in range(20):
        server.train(10)
        print(f'Epoch {server.epoch}')
        test(val_dataloader, server.global_model, constants.loss_fn, constants.device)
    
    # Probably can use state_dict to average models
    print('done')

if __name__ == '__main__':
    main()
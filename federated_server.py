from http import client
import federated_client
import random
import torch
import torch.backends.mps
import torchvision
from torch.utils.data import DataLoader, random_split

def federated_server(test_dataloader, rounds, loss_fn, clients, client_fraction, device):
    global_model = federated_client.NeuralNetwork().to(device)
    global_state = global_model.state_dict()

    for i in range(rounds):
        new_global_state = global_state.copy()
        for layer in new_global_state:
            new_global_state[layer] = 0
        num_clients = int(len(clients) * client_fraction)
        clients_for_round = random.sample(clients, num_clients)
        for client in clients_for_round:
            update = client.client_update(global_state)
            for layer in new_global_state:
                new_global_state[layer] += update[layer]
        
        for layer in new_global_state:
            new_global_state[layer] /= num_clients
        
        global_state = new_global_state

        # Test new model
        global_model.load_state_dict(global_state)
        test(test_dataloader, global_model, loss_fn, device)

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
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    clients = []
    num_clients = 100
    for client_data in random_split(training_data, [len(training_data) // num_clients] * num_clients):
        clients.append(federated_client.FederatedClient(client_data, 1, 60, device))

    batch_size = 64

    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    

    loss_fn = torch.nn.CrossEntropyLoss()

    federated_server(test_dataloader, 50, loss_fn, clients, 0.1, device)
    
    # Probably can use state_dict to average models
    print('done')

if __name__ == '__main__':
    main()
import torch, torch.backends.mps

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
loss_fn = torch.nn.CrossEntropyLoss()
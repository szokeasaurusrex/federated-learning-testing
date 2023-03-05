import torch, torch.backends.mps

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
loss_fn = torch.nn.CrossEntropyLoss()
dynamic_clip_probability_threshold = 0.35
num_clients = 100
adversarial_fraction = 0.25
learning_rate = 1e-1
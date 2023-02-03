import torch, torch.backends.mps

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
loss_fn = torch.nn.CrossEntropyLoss()
dynamic_clip_probability_threshold = 0.35
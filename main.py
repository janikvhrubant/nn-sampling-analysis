import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    mps_device = torch.device("cpu")

model = mps_device
x = x.to(device)
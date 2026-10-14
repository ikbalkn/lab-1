# deneme dali
import torch

if torch.cuda.is_available():
    aygit = torch.device("cuda")
elif torch.backends.mps.is_available():
    aygit = torch.device("mps")
else:
    aygit = torch.device("cpu")

x = torch.randn(1000, 1000, device=aygit)
print(aygit, (x @ x).shape)

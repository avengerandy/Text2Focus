import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use GPU.")
    print("Current GPU: ", torch.cuda.get_device_name(0))
    x = torch.rand(5, 5)
    x = x.to('cuda')
    print("Tensor on GPU:", x)
else:
    print("CUDA is not available. PyTorch cannot use GPU.")

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Move a tensor to GPU
    device = torch.device("cuda")
    x = torch.rand(3, 3).to(device)
    print(f"Tensor on GPU:\n{x}")
else:
    print("No GPU available")
   
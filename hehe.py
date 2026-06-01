import torch
device= "cuda" if torch.cuda.is_available() else "cpu"
print(device)
def check_cuda():
    print(torch.version.cuda)
    cuda_is_ok = torch.cuda.is_available()
    print(f"CUDA Enabled: {cuda_is_ok}")
check_cuda()
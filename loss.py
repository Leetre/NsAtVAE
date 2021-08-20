import torch

def Loss(error_c, energy, error_r):
    # Lambda = torch.tensor([0.01], dtype=torch.float)
    L = 10*error_c - 0.005*energy + 5*error_r
    # L = error_c + error_r
    return L

if __name__ == '__main__':
    print(Loss(0, 0, 0))
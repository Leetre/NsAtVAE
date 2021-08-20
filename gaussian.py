import torch, math
import torch.nn as nn

class GaussianMixture(nn.Module):
    def __init__(self, embedding_size, num_components=8):
        super(GaussianMixture, self).__init__()
        self.num_components = num_components
        self.component_weight = nn.Sequential(
            nn.Linear(embedding_size, num_components),
            nn.Softmax(-1)
        )
    
    def forward(self, x):
        seq_len = x.size()[1]
        PI = torch.tensor([2*math.pi], dtype=float)
        component_weight = self.component_weight(x.squeeze())
        component_weight_ = torch.sum(component_weight, 0).view(1, -1)/torch.tensor([seq_len], dtype=torch.float)
        energy = torch.tensor([0], dtype=torch.float)
        mu_l = []
        sigma_l = []
        for i in range(self.num_components):
            component_k = component_weight[:, i:i+1]
            mu_tmp = component_k*x.squeeze()
            mu_k = torch.sum(mu_tmp, 0).view(1, -1)/torch.sum(component_k, 0).view(1, -1)
            mu_l.append(mu_k)
            sigma_tmp = torch.tensor([0], dtype=torch.float)
            for j in range(seq_len):
                tmp = x.squeeze()[j:j+1]-mu_k
                sigma_tmp = sigma_tmp + component_k[j:j+1]*torch.matmul(tmp.T, tmp)
            sigma_k = sigma_tmp/torch.sum(component_k, 0).view(1, -1)
            sigma_l.append(sigma_k)

        for i in range(seq_len):
            i_result = torch.tensor([0], dtype=torch.float)
            for j in range(self.num_components):
                tmp = x.squeeze()[i:i+1]-mu_l[j]
                k_result = component_weight_[:, j:j+1]*torch.exp(torch.tensor([-0.5], dtype=torch.float)
                *torch.matmul(torch.matmul(tmp,torch.inverse(sigma_l[j])), tmp.T))/torch.abs((torch.det(PI*sigma_l[j]))).sqrt()
                i_result = i_result + k_result
            energy_i = -torch.log(i_result)
            energy = energy + energy_i
        return energy

if __name__ == '__main__':
    model = GaussianMixture(4, 8)
    x_in = torch.randn(1, 3, 4)
    x_out = model(x_in)
    print(x_out.shape, '\n', x_out)
    # print(x_out)
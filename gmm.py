import torch
import torch.nn as nn
import numpy as np

class GMM(nn.Module):
    def __init__(self, embedding_size, num_components=8):
        super(GMM, self).__init__()
        self.num_components = num_components
        self.component_weight = nn.Sequential(
            nn.Linear(embedding_size, num_components),
            nn.Softmax(-1)
        )

    def compute_gmm_params(self, x):
        z = x.squeeze()
        seq_len = z.size()[0]
        component_weight = self.component_weight(z)
        component_weight_ = torch.sum(component_weight, 0).view(1, -1) / torch.tensor([seq_len], dtype=torch.float)
        mu = torch.sum(component_weight.unsqueeze(-1) * z.unsqueeze(1), dim=0) / torch.sum(component_weight, dim=0).unsqueeze(-1)
        z_mu = z.unsqueeze(1)- mu.unsqueeze(0)
        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        # K x D x D
        cov = torch.sum(component_weight.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / torch.sum(component_weight, dim=0).unsqueeze(-1).unsqueeze(-1)
        return component_weight_, mu, cov

    def compute_energy(self, x, phi, mu, cov):
        z = x.squeeze()
        k, D, _ = cov.size()
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-5
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + torch.eye(D)*eps
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            det_cov.append(torch.abs(torch.det(cov_k*(2*np.pi))).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov)
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(torch.sum((phi.unsqueeze(0) * exp_term).squeeze() / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        energy = torch.sum(sample_energy)
        return energy

    def forward(self, x):
        phi, mu, cov = self.compute_gmm_params(x)
        energy = self.compute_energy(x, phi, mu, cov)
        return energy

if __name__ == '__main__':
    model = GMM(4, 8)
    x_in = torch.randn(1, 3, 4)
    x_out = model(x_in)
    print(x_out.shape, '\n', x_out)
    # print(x_out)
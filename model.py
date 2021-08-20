from gmm import GMM
from selfatt import *
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NsAtVAE(nn.Module):
    def __init__(self,
                 embedding_size,
                 drop_out=0.2,
                 num_heads=1,
                 num_components=8):
        super(NsAtVAE, self).__init__()
        self.encoder = SelfAttention(embedding_size, num_heads, drop_out)
        self.decoder = SelfAttention(2*embedding_size, num_heads, drop_out)
        self.mu = nn.Linear(embedding_size, embedding_size)
        self.sigma = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Softplus()
        )
        self.gaussian = GMM(embedding_size, num_components)
        self.out = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.Softmax(-1)
        )
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        last_mu, last_sigma, x1 = x
        seq_len = x1.size()[0]
        embedding_size = x1.size()[1]
        norm_num = seq_len * embedding_size
        x_ = x1.view(1, seq_len, embedding_size)
        # encode
        z_a = self.encoder(self.encoder(x_))
        # differences in data distribution between two time series
        mu = self.mu(z_a)
        sigma = self.sigma(z_a)
        current_d = Normal(mu, sigma)
        last_d = Normal(last_mu, last_sigma)
        z_c = kl_divergence(current_d, last_d)
        error_c = z_c.mean()
        # concatenate hidden vector
        z = torch.cat([z_a, z_c], -1)
        # Gaussian mixture model
        # energy = 0
        energy = self.gaussian(z_a)/torch.tensor([seq_len], dtype=torch.float)
        energy = energy.squeeze()
        # decode
        output = self.out(self.decoder(self.decoder(z)))
        input_norm = self.softmax(x_)
        error_r = torch.norm(output-input_norm)/torch.tensor([norm_num], dtype=torch.float).sqrt()
        error_r = error_r.squeeze()
        score = (torch.sum((output-input_norm)**2, -1)).squeeze().sqrt()

        return error_c, energy, error_r, score, mu, sigma
if __name__ == '__main__':
    model = NsAtVAE(4)
    x_in = torch.randn(3, 4)
    x = tuple([0, 1, x_in])
    x_out = model(x)
    # print(x_out.shape, '\n', x_out)
    # print(x_in)
    print(x_out)
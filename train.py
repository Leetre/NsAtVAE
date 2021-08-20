import argparse, yaml
import numpy as np
from model import *
import loss

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yml')
parser.add_argument('--datapath', type=str, default='data/valve1')
parser.add_argument('--epochs', type=int, default=3)

def load_data(path):
    data = np.load(path+'/'+'data.npy')
    return data

def load_model(config_f):
    with open(config_f, 'r') as config_file:
        config = yaml.load(config_file)
        model_params = config['model']
        model = NsAtVAE(**model_params)
    return model

def train(args):
    datapath = args.datapath
    config_f = args.config
    epochs = args.epochs
    data = load_data(datapath)
    data_num = data.shape[0]
    model = load_model(config_f)
    print(model)

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters())

    # training
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        last_mu = torch.tensor(0, dtype=float)
        last_sigma = torch.tensor(1, dtype=float)
        for num in range(data_num):
            current_data = torch.tensor(data[num], dtype=torch.float)
            # print(current_data)
            error_c, energy, error_r, _, mu, sigma = model(tuple((last_mu, last_sigma, current_data)))
            # print(error_r)
            mu.detach_()
            sigma.detach_()
            current_loss = loss.Loss(error_c, energy, error_r)
            last_mu = mu
            last_sigma = sigma
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

            print(("epoch %d: data %d: current loss is %.5f")%(epoch+1, num, current_loss.item()))

        torch.save(model.state_dict(), 'model/epoch'+str(epoch)+'.pth')
    # print(model)
    # print(data.shape[0])

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

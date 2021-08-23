import argparse, yaml
import numpy as np
from model import *
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yml')
parser.add_argument('--datapath', type=str, default='data/valve1')
parser.add_argument('--labelpath', type=str, default='data/valve1')
parser.add_argument('--modelpath', type=str, default='model/epoch10.pth')

def load_data(path):
    data = np.load(path+'/'+'data.npy')
    return data

def load_label(path):
    label = np.load(path+'/'+'label.npy')
    return label

def load_model(config_f):
    with open(config_f, 'r') as config_file:
        config = yaml.load(config_file)
        model_params = config['model']
        model = NsAtVAE(**model_params)
    return model

def test(args):
    datapath = args.datapath
    labelpath = args.labelpath
    config_f = args.config
    modelpath = args.modelpath
    data = load_data(datapath)
    label = load_label(labelpath)
    data_num = data.shape[0]
    model = load_model(config_f)
    model.load_state_dict(torch.load(modelpath))
    model = model.eval()
    last_mu = torch.tensor(0, dtype=float)
    last_sigma = torch.tensor(1, dtype=float)
    result = []
    for i in range(data_num):
        current_data = torch.tensor(data[i], dtype=torch.float)
        _, _, _, score, mu, sigma = model(tuple((last_mu, last_sigma, current_data)))
        print(score)
        last_mu = mu
        last_sigma = sigma
        result.extend(score.detach_().numpy().tolist())
    result = np.array(result)
    result = result.reshape((-1, 1))
    min_max_scaler = preprocessing.MinMaxScaler()
    result = min_max_scaler.fit_transform(result)
    result = np.squeeze(result)
    result[result>0.5] = 1
    result[result<=0.5] = 0
    tmp = (result == label)
    avg_acc = np.mean(tmp)
    print(avg_acc)


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
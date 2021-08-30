import argparse, yaml
import numpy as np
from model import *
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yml')
parser.add_argument('--datapath', type=str, default='data/valve1')
parser.add_argument('--labelpath', type=str, default='data/valve1')
parser.add_argument('--modelpath', type=str, default='model/epoch10SKAB1.pth')

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
        last_mu = mu
        last_sigma = sigma
        result.extend(score.detach_().numpy().tolist())
    result = np.array(result)
    result = result.reshape((-1, 1))
    min_max_scaler = preprocessing.MinMaxScaler()
    result = min_max_scaler.fit_transform(result)
    result = np.squeeze(result)
    result[result>0.2] = 1
    result[result<=0.2] = 0
    prediction_l = result.tolist()
    label_l = label.tolist()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(label_l)):
        if prediction_l[i] == 1 and label_l[i] == 1:
            TP += 1
        if prediction_l[i] == 1 and label_l[i] == 0:
            FP += 1
        if prediction_l[i] == 0 and label_l[i] == 1:
            FN += 1
        if prediction_l[i] == 0 and label_l[i] == 0:
            TN += 1
    precision_1 = float(float(TP)/(float(TP)+float(FP)))
    recall_1 = float(float(TP)/(float(TP)+float(FN)))
    F1_1 = 2.0*precision_1*recall_1 / (precision_1+recall_1)
    precision_0 = float(float(TN)/(float(TN)+float(FN)))
    recall_0 = float(float(TN)/(float(TN)+float(FP)))
    F1_0 = 2.0*precision_0*recall_0 / (precision_0+recall_0)
    F1 = (F1_0+F1_1) / 2.0
    print(F1_0)
    print(F1_1)
    print(F1)

    # tmp = (result == label)
    # avg_acc = np.mean(tmp)
    # print(avg_acc)


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)

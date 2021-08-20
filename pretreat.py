import csv, os
import numpy as np
from sklearn import preprocessing

path = 'data/valve1'
files = os.listdir(path)
files.sort(key=lambda x: int(x.split('.')[0]))
data = []
label = []
for file in files:
    with open(path+"/"+file, 'r') as f:
        reader = csv.reader(f)
        data_num = 0
        for row in reader:
            if data_num != 0:
                data.append(list(map(float, row[0].split(';')[1:-2])))
                label.append(float(row[0].split(';')[-2]))
            data_num = data_num + 1
data = np.array(data)
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

data = data.reshape((6054, 3, 8))
label = np.array(label)
np.save('data/valve1/data.npy', data)
np.save('data/valve1/label.npy', label)
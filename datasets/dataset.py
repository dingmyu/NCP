import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DataStat:
    def __init__(self, cfg, name='seg'):
        self.data = pickle.load(open(f'data/{name}/trainval.pkl', 'rb'))

        self.input = np.array([item['embedding'] for item in self.data]).astype(np.float32)
        cfg.data.numbers = len(self.data)
        cfg.data.input_mean = self.input.mean(0)
        cfg.data.input_std = self.input.std(0)
        cfg.network.input_dim = self.input.shape[1]
        cfg.network.metrics = [len(item) for item in cfg.data.metrics]
        self.input = (self.input - cfg.data.input_mean) / cfg.data.input_std

        self.metrics = []
        for metric in cfg.data.metrics:
            self.metrics.extend(metric)
        cfg.data.combined_metrics = self.metrics

        self.output = []
        for item in self.data:
            output_item = []
            for metric in self.metrics:
                output_item.append(item[metric])
            self.output.append(output_item)
        self.output = np.array(self.output).astype(np.float32)
        cfg.data.output_mean = self.output.mean(0)
        cfg.data.output_std = self.output.std(0)
        self.output = (self.output - cfg.data.output_mean) / cfg.data.output_std

        self.input_train, self.input_val, self.output_train, self.output_val = \
            train_test_split(self.input, self.output, test_size=0.2)

        pickle.dump({'input_train': self.input_train,
                     'input_val': self.input_val,
                     'output_train': self.output_train,
                     'output_val': self.output_val},
                    open('{}/data_split.pkl'.format(cfg.log_dir), 'wb'))


class BaseDataset(Dataset):
    def __init__(self, data_stat, mode='train'):
        if mode == 'train':
            self.data = [data_stat.input_train, data_stat.output_train]
        if mode == 'val':
            self.data = [data_stat.input_val, data_stat.output_val]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, item):
        return self.data[0][item], self.data[1][item]

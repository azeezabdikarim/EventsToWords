import os
import pandas as pd
import numpy as np
# from tqdm import tqdm
import torch
import pickle as pkl
from torch.utils.data import DataLoader, Dataset

class EventsToWords(Dataset):
    def __init__(self, directory, time_bins = 10, resize_scale = 0.5, size = [640,480], device = 'cpu'):
        self.device = device
        self.time_bins = time_bins
        self.resize_scale = resize_scale
        self.og_size = size
        self.train_set, self.test_set = self._load_data(directory)
        self.train_samples, self.train_targets, self.train_text_labels = self._build_training_set(self.train_set)
        self.test_samples = self._build_test_set(self.test_set)

    def __len__(self):
        return len(self.train_samples)
    
    def __getitem__(self, idx):
        sample = self.train_samples[idx]
        target = self.train_targets[idx]
        text_label = self.train_text_labels[idx]
        return sample, target, text_label

    def _load_csv_smart_header(self, path, columns):
        df = pd.read_csv(path)
        # If the first row contains non-numeric data, assume it's a header and reload without column names
        if 'x' in df.columns:
            return df  # Header is present
        else:
            df = pd.read_csv(path, header=None, names=columns)  # Header is not present, add manually
        return df

    def _load_data(self, directory):
        test_set_data = []
        test_set_paths = []
        train_set_paths = {}
        train_set_data = {}
        columns = ['x', 'y', 'polarity', 'time']
        for dirname, _, filenames in os.walk(directory):
            for filename in filenames:
                if 'smemi309-final-evaluation-challenge-2022/test10/test10' in dirname:
                    p = os.path.join(dirname, filename)
                    test_set_paths.append(p)
                    df = self._load_csv_smart_header(p, columns)
                    test_set_data.append(df)
                if "smemi309-final-evaluation-challenge-2022/train10/train10" in dirname:
                    stem_len = len("smemi309-final-evaluation-challenge-2022/train10/train10")
                    class_label = dirname.split('/')[-1]
                    p = os.path.join(dirname, filename)
                    if class_label in train_set_paths.keys():
                        train_set_paths[class_label].append(p)
                        df = self._load_csv_smart_header(p, columns)
                        train_set_data[class_label].append(df)
                    else:
                        train_set_paths[class_label] = [p]
                        df = self._load_csv_smart_header(p, columns)
                        train_set_data[class_label] = [df]

        print(f"The length of the test set is {len(train_set_data)}")
        print(f"The keys to access path files for the training set are {len(train_set_data)}")
        for k in train_set_data.keys():
            print(f"\t{k} has {len(train_set_data[k])} samples")
        return train_set_data, test_set_data

    def _build_test_set(self, test_set):
        test_samples = []
        for s in test_set:
            test_samples.append(self.bin_sample_vectorized(s, self.time_bins, self.resize_scale, self.og_size))
        return test_samples

    def _build_training_set(self, dict_of_samples):
        samples = []
        text_labels = []
        one_hot_labels = []
        class_list = list(dict_of_samples.keys())
        for i,k in enumerate(class_list):
            for s in dict_of_samples[k]:
                samples.append(self.bin_sample_vectorized(s, self.time_bins, self.resize_scale, self.og_size))
                text_labels.append(k)
                one_hot = np.zeros(len(class_list))
                one_hot[i] = 1
                one_hot_labels.append(one_hot)
        return np.array(samples), np.array(one_hot_labels), np.array(text_labels),

    def bin_sample_vectorized(self, sample, time_bins, resize_scale=0.5, size=[640, 480]):
        temp = sample.copy()
        bin_size = int(sample.time.max() / time_bins) + 1
        temp['time_bin'] = pd.cut(temp['time'], bins=np.arange(-1, sample.time.max() + bin_size, bin_size), labels=False)

        x_dim, y_dim = size
        x_dim_bin = int(x_dim * resize_scale) + 1
        y_dim_bin = int(y_dim * resize_scale) + 1

        temp['xbin'] = pd.cut(temp['x'], bins=np.arange(-1, x_dim, int(x_dim / x_dim_bin) + 1), labels=False)
        temp['ybin'] = pd.cut(temp['y'], bins=np.arange(-1, y_dim, int(y_dim / y_dim_bin) + 1), labels=False)

        bin_group_events = temp.groupby(by=['time_bin', 'xbin', 'ybin'])['polarity'].sum().reset_index()

        # Convert to PyTorch tensors
        indices = torch.tensor(bin_group_events[['time_bin', 'ybin', 'xbin']].values)
        polarities = torch.tensor(bin_group_events['polarity'].values, dtype=torch.float32)

        # Prepare the layers tensor
        layers = torch.zeros((time_bins, y_dim_bin, x_dim_bin))

        # Vectorized update
        layers[indices[:, 0], indices[:, 1], indices[:, 2]] += polarities

        return layers

    def bin_sample(self,sample, time_bins, resize_scale = 0.5, size = [640,480]):
        temp = sample.copy()
        bin_size = int(sample.time.max()/time_bins) + 1
        bin_labels = list(range(time_bins))
        bin_ranges = list(range(-1, sample.time.max() + bin_size, bin_size))
        temp['time_bin'] = pd.cut(temp['time'], bins=bin_ranges, labels=False)
            
        x_dim = size[0]
        y_dim = size[1]
        
        x_dim_bin = int(x_dim*resize_scale) + 1
        y_dim_bin = int(y_dim*resize_scale) + 1
        
        x_bin_size = int(x_dim/x_dim_bin) + 1
        x_bin_labels = list(range(x_dim_bin))
        x_bin_ranges = list(range(-1, x_dim, x_bin_size))
        
        y_bin_size = int(y_dim/y_dim_bin) + 1
        y_bin_labels = list(range(y_dim_bin))
        y_bin_ranges = list(range(-1, y_dim, y_bin_size))
        
        temp['xbin'] = pd.cut(temp['x'], bins=x_bin_ranges, labels=False)
        temp['ybin'] = pd.cut(temp['y'], bins=x_bin_ranges, labels=False)
        
        bin_group_events = temp.groupby(by=['time_bin', 'xbin', 'ybin'])['polarity'].sum().reset_index()
        
    #     return bin_group_events
        #####
        layers = [np.zeros([y_dim_bin, x_dim_bin]) for x in range(time_bins)]
        for s in bin_group_events.iterrows():
            i, x, y, polarity = s[1]
            i, x, y, polarity = int(i), int(x), int(y), int(polarity)
            layers[i][y][x] += polarity
        return layers


if __name__ == "__main__":
    data_dir = '/Users/azeez/Documents/Github_Projects/EventsToWords/smemi309-final-evaluation-challenge-2022'
    dataset = EventsToWords(data_dir)
    print('done')

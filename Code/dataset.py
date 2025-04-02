import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Compose
import transforms as T

class FWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''
    def __init__(self, anno, preload=True, sample_ratio=1, file_size=1000,
                    transform_data=None, transform_label=None):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        if preload: 
            self.data_list, self.label_list = [], []
            for batch in self.batches: 
                data, label = self.load_every(batch)
                self.data_list.append(data)
                self.label_list.append(label)

    def load_every(self, batch):
        batch = batch.split('\t')
        data_path, label_path = batch[0], batch[1][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        label = np.load(label_path)
        data = data.astype('float32')
        label = label.astype('float32')
        return data, label
        
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx]
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx]
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label:
            label = self.transform_label(label)
        return data, label
        
    def __len__(self):
        return len(self.batches) * self.file_size


if __name__ == '__main__':
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(2000, 6000)
    ])
    suffix = '3_357_raw'
    dataset = FWIDataset(f'relevant_files/flat_transform_{suffix}.txt',
                transform_data=transform_data, transform_label=transform_label)
    train_set, valid_set = random_split(dataset, [2000, 1000], generator=torch.Generator().manual_seed(0))
    print('Before saving: ', len(train_set), len(valid_set))
    save = True
    if save:
        print('Saving...')
        torch.save(train_set, f'relevant_files/flat_transform_train_{suffix}.pth')
        torch.save(valid_set, f'relevant_files/flat_transform_valid_{suffix}.pth')
        print('Verifying...')
        train_set_verify = torch.load(f'relevant_files/flat_transform_train_{suffix}.pth')
        valid_set_verify = torch.load(f'relevant_files/flat_transform_valid_{suffix}.pth')
        print('Load saving: ', type(train_set_verify), len(train_set_verify), 
                                type(valid_set_verify), len(valid_set_verify))
        data, label = train_set_verify[0]
        print('Read sample: ', data.shape, label.shape)

import cv2

from torch.utils.data import Dataset
from datasets.data import customized_dataset

# TODO: annotation
class MADFD_dataset(Dataset):
    def __init__(self, dir_root, split='train'):
        '''
        dir_root: directory root of your dataset
        split: train dataset or test dataset
        '''
        self.epoch_n = 0
        self.split = split
        self.dir_root = dir_root
        self.dataset = customized_dataset(self.dir_root, self.split)

    def next(self):
        self.epoch_n += 1

    def __getitem__(self, i):
        data = {}
        filepath, annotation = self.dataset[i]
        data['data'] = cv2.imread(filepath)
        data['anno'] = annotation
        return data

    def __len__(self, ):
        return len(self.dataset)
from abc import ABC, abstractmethod
import importlib
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "mislight.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(f'In {dataset_filename}.py, there should be a subclass of BaseDataset with class name that matches {target_dataset_name} in lowercase.')
    return dataset

class BaseDataset(Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.datadir = opt.datadir
        self.loader = get_loader(opt)
        
    def batch_sampler(self, shuffle=True):
        return None
        
    @staticmethod
    def add_dataset_specific_args(parser):
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    
    
def get_loader(opt):
    rawloader = None
    
    if opt.file_extension.lower() == 'npy':
        def npy_loader(x):
            y = np.load(x, allow_pickle=True)
            try:
                y = y[()]
            except:
                pass
            return y
        rawloader = npy_loader
    
    return lambda x: rawloader(x)


###############################################
## Batch sampler for Semi-supervised Learning

###############################################

class TwoStreamMiniBatchSampler(Sampler):
    """Iterate mini-batch from TwoStreamBatchSampler
    Minibatch_size is the true "batch size" of sampler
    e.g.: if, primary_indices ABC ... primary_batch_size=1 / secondary indices abc ... secondary_batch_size=3 / minibatch=2
        generates -> Aa bc Bd ef ...
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, minibatch_size=1, shuffle=True):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mini = minibatch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices, self.shuffle)
        secondary_iter = iterate_eternally(self.secondary_indices, self.shuffle) 
        sample = (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )
        
        return (x for x in grouper(itertools.chain.from_iterable(sample), self.mini))
        

    def __len__(self):
        return ((len(self.primary_indices) // self.primary_batch_size) * self.batch_size) // self.mini

### Below copied from: https://github.com/HiLab-git/SSL4MIS/blob/master/code/dataloaders/brats2019.py
    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, shuffle=True):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.shuffle = shuffle

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices, self.shuffle)
        secondary_iter = iterate_eternally(self.secondary_indices, self.shuffle)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable, shuffle=True):
    if shuffle:
        return np.random.permutation(iterable)
    else:
        return iterable


def iterate_eternally(indices, shuffle=True):
    def infinite_shuffles():
        while True:
            if shuffle:
                yield np.random.permutation(indices)
            else:
                yield indices
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

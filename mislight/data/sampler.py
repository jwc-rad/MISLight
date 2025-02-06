import itertools
import numpy as np
from typing import Optional

import torch
from torch.utils.data.sampler import Sampler

from hydra.utils import instantiate

import lightning.pytorch as pl


class SamplerDataModule(pl.LightningDataModule):
    """
    DataModule should be instantiated with _recursive_=False, so that dataset and dataloaders are not defined at init. e.g., instantiate(dm_cfg, _recursive_=False)
    """

    def __init__(self, dataset, dataloader):
        super().__init__()
        assert getattr(dataset, "train", None), "dataset must have train attr"
        assert getattr(dataloader, "train", None), "dataloader must have train attr"
        self.dataset = dataset
        self.dataloader = dataloader

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.ds_train = instantiate(self.dataset.train, _recursive_=False)
            print(f"train dataset [{type(self.ds_train).__name__}] was created")

            if getattr(self.dataset, "valid", None) is not None:
                self.ds_valid = instantiate(self.dataset.valid, _recursive_=False)
                print(f"valid dataset [{type(self.ds_valid).__name__}] was created")

        if stage == "test" or stage is None:
            if getattr(self.dataset, "test", None) is not None:
                self.ds_test = instantiate(self.dataset.test, _recursive_=False)
                print(f"test datasets [{type(self.ds_test).__name__}] were created")

        if stage == "valid":
            if getattr(self.dataset, "valid", None) is not None:
                self.ds_valid = instantiate(self.dataset.valid, _recursive_=False)
                print(f"valid dataset [{type(self.ds_valid).__name__}] was created")

    def train_dataloader(self):
        _sampler = (
            self.ds_train._sampler(True) if hasattr(self.ds_train, "_sampler") else None
        )
        if _sampler is None:
            DL = instantiate(self.dataloader.train, dataset=self.ds_train)
        else:
            for k in ["shuffle", "sampler"]:
                if k in self.dataloader.train:
                    self.dataloader.train.__delattr__(k)
            DL = instantiate(
                self.dataloader.train, dataset=self.ds_train, sampler=_sampler
            )
        return DL

    def val_dataloader(self):
        if hasattr(self, "ds_valid") and len(self.ds_valid) > 0:
            _sampler = (
                self.ds_valid._sampler(False)
                if hasattr(self.ds_valid, "_sampler")
                else None
            )
            if _sampler is None:
                DL = instantiate(self.dataloader.valid, dataset=self.ds_valid)
            else:
                for k in ["shuffle", "sampler"]:
                    if k in self.dataloader.valid:
                        self.dataloader.valid.__delattr__(k)
                DL = instantiate(
                    self.dataloader.valid, dataset=self.ds_valid, sampler=_sampler
                )
            return DL
        else:
            return None

    def test_dataloader(self):
        if hasattr(self, "ds_test") and len(self.ds_test) > 0:
            _sampler = (
                self.ds_test._sampler(False)
                if hasattr(self.ds_test, "_sampler")
                else None
            )
            if _sampler is None:
                DL = instantiate(self.dataloader.test, dataset=self.ds_test)
            else:
                for k in ["shuffle", "sampler"]:
                    if k in self.dataloader.test:
                        self.dataloader.test.__delattr__(k)
                DL = instantiate(
                    self.dataloader.test, dataset=self.ds_test, sampler=_sampler
                )
            return DL
        else:
            return None


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


class TwoStreamMiniBatchSampler(Sampler):
    """Iterate mini-batch from TwoStreamBatchSampler
    Minibatch_size is the true "batch size" of sampler
    e.g.: if, primary_indices ABC ... primary_batch_size=1 / secondary indices abc ... secondary_batch_size=3 / minibatch=2
        generates -> Aa bc Bd ef ...
    """

    def __init__(
        self,
        primary_indices,
        secondary_indices,
        batch_size=1,
        secondary_batch_size=1,
        minibatch_size=1,
        shuffle=True,
        drop_last=False,
        len_dataset: int = None,
    ):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mini = minibatch_size
        self.drop_last = drop_last
        self.len_dataset = len_dataset

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        # primary_iter = iterate_once(self.primary_indices, self.shuffle)
        primary_iter = iterate_eternally(self.primary_indices, self.shuffle)
        secondary_iter = iterate_eternally(self.secondary_indices, self.shuffle)
        sample = (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

        return (x for x in grouper(itertools.chain.from_iterable(sample), self.mini))

    def __len__(self):
        if self.len_dataset is None:
            if self.drop_last:
                return (
                    (len(self.primary_indices) // self.primary_batch_size)
                    * self.batch_size
                ) // self.mini
            else:
                return (
                    (
                        (len(self.primary_indices) + self.primary_batch_size - 1)
                        // self.primary_batch_size
                    )
                    * self.batch_size
                    + self.mini
                    - 1
                ) // self.mini
        else:
            if self.drop_last:
                return (
                    (self.len_dataset // self.primary_batch_size) * self.batch_size
                ) // self.mini
            else:
                return (
                    (
                        (self.len_dataset + self.primary_batch_size - 1)
                        // self.primary_batch_size
                    )
                    * self.batch_size
                    + self.mini
                    - 1
                ) // self.mini

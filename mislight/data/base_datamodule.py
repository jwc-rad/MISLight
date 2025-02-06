from typing import Optional

from hydra.utils import instantiate

import lightning.pytorch as pl


class BaseDataModule(pl.LightningDataModule):
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
        DL = instantiate(self.dataloader.train, dataset=self.ds_train)
        return DL

    def val_dataloader(self):
        if hasattr(self, "ds_valid") and len(self.ds_valid) > 0:
            DL = instantiate(self.dataloader.valid, dataset=self.ds_valid)
            return DL
        else:
            return None

    def test_dataloader(self):
        if hasattr(self, "ds_test") and len(self.ds_test) > 0:
            DL = instantiate(self.dataloader.test, dataset=self.ds_test)
            return DL
        else:
            return None


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

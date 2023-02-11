from typing import Optional

from hydra.utils import instantiate

import pytorch_lightning as pl

class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset_train, dataloader_train, 
        dataset_valid=None, dataloader_valid=None,
        dataset_test=None, dataloader_test=None,
    ):
        super().__init__()
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.dataset_test = dataset_test
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.dataloader_test = dataloader_test
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.ds_train = instantiate(self.dataset_train, _recursive_ = False)
            print(f"train dataset [{type(self.ds_train).__name__}] was created")
            
            if getattr(self, 'dataset_valid', None) is not None:      
                self.ds_valid = instantiate(self.dataset_valid, _recursive_ = False)
                print(f"valid dataset [{type(self.ds_valid).__name__}] was created")  

        if stage == "test" or stage is None:  
            if getattr(self, 'dataset_test', None) is not None:             
                self.ds_test = instantiate(self.dataset_test, _recursive_ = False)
                print(f"test datasets [{type(self.ds_test).__name__}] were created")            
            
        if stage == "valid":
            if getattr(self, 'dataset_valid', None) is not None:
                self.ds_valid = instantiate(self.dataset_valid, _recursive_ = False)
                print(f"valid dataset [{type(self.ds_valid).__name__}] was created")              
            
    def train_dataloader(self):
        DL = instantiate(self.dataloader_train, dataset=self.ds_train)
        return DL
    
    def val_dataloader(self):
        if len(self.ds_valid) > 0:
            DL = instantiate(self.dataloader_valid, dataset=self.ds_valid)
            return DL
        else:
            return None
        
    def test_dataloader(self):
        if len(self.ds_test) > 0:
            DL = instantiate(self.dataloader_test, dataset=self.ds_test)
            return DL
        else:
            return None
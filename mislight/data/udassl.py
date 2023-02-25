import random

from torch.utils.data import Dataset

from monai.transforms import Compose

from mislight.utils.hydra import instantiate_list

class UDASSLBase(Dataset):     
    def __init__(self, transform):
        super().__init__()
        self.prepare_transforms(transform)
           
    def __len__(self):        
        return self.imageA_size

    def __getitem__(self, index):
        read_items = self.read_data(index)

        return_items = self.run_transform(read_items)
        
        return return_items
    
    ## override this to define self.keys, paths, and etc.
    def prepare_data(self):
        pass
    
    ## override this to read data by index
    def read_data(self, index):
        imageA_path = self.imageA_paths[index % self.imageA_size]
        read_items = {
            'imageA': imageA_path,
        }     
        metadata = {'imageA_path': imageA_path}   
        
        if hasattr(self, 'labelA_paths'):
            labelA_path = self.labelA_paths[index % self.imageA_size]
            read_items['labelA'] = labelA_path
            metadata['labelA_path'] = labelA_path
        
        if hasattr(self, 'imageB_paths'):
            index_B = random.randint(0, self.imageB_size - 1)
            imageB_path = self.imageB_paths[index_B % self.imageB_size]
            read_items['imageB'] = imageB_path
            metadata['imageB_path'] = imageB_path            
                
        read_items['metadata'] = metadata
        return read_items        
    
    ## override this to define transforms
    def prepare_transforms(self, transform):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
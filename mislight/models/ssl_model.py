from .segmentation_model import SegmentationModel

class SSLModel(SegmentationModel):
    '''Semi-supervised Learning Base Model
    Define steps for labeled, unlabeled, all inputs
    '''
    def set_input(self, batch):
        self.image = batch['image']
        if not self.inference:
            self.label = batch['label']
            self.labeled = batch['labeled']
            
    def _step_S(self, stage):
        loss = self._step_S_labeled(stage) + self._step_S_unlabeled(stage) + self._step_S_all(stage)
        return loss
    
    def _step_S_labeled(self, stage):
        return 0
    
    def _step_S_unlabeled(self, stage):
        return 0
    
    def _step_S_all(self, stage):
        return 0

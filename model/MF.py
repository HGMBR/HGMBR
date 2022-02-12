import torch
import torch.nn.functional as F
from model.base_model import Model


class MF(Model):
    def __init__(self, args, data_dict=None):
        super().__init__(args, data_dict)

    def propagate(self):
        u_embeds = self.u_embeds
        i_embeds = self.i_embeds
        return u_embeds, i_embeds

    @property
    def dataset_format(self):
        return 'SamplingTrainSet'




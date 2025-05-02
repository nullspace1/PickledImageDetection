from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, weight_box,weight_found):
        self.weight_box = weight_box
        self.weight_found = weight_found

    def forward(self,box : Tensor, found : Tensor, box_gt : Tensor, found_gt : Tensor) -> Tensor:
        if (found_gt == 0):
            return F.cross_entropy(found, found_gt)
        box_loss = F.smooth_l1_loss(box, box_gt, reduction='sum')
        found_loss = F.cross_entropy(found, found_gt)
        return self.weight_box * box_loss + self.weight_found * found_loss    
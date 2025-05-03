import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, weight_box,weight_found):
        self.weight_box = weight_box
        self.weight_found = weight_found

    def forward(self,box : Tensor, found : Tensor, box_gt : Tensor, found_gt : Tensor) -> Tensor:
        box_loss = F.smooth_l1_loss(box, box_gt, reduction='sum')
        found_loss = F.binary_cross_entropy(found, found_gt)
        return self.weight_box * box_loss + self.weight_found * found_loss    
    
    def test(self):
        print("Testing loss...")
        input_box = torch.randn(1,4)
        input_found = torch.sigmoid(torch.randn(1,1))
        input_box_gt = torch.randn(1,4)
        input_found_gt = torch.sigmoid(torch.randn(1,1))
        output = self.forward(input_box, input_found, input_box_gt, input_found_gt)
        print('output.shape: ', output.shape)
        print('expected: number')
        assert output.item() > 0
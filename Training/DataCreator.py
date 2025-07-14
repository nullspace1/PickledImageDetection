
import torch


class DataCreator:
    
    def create_data(self, index : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("create_data() must be implemented by subclasses")
    
    def sample_size(self) -> int:
        raise NotImplementedError("sample_size() must be implemented by subclasses")
    
class BasicShapeDataCreator(DataCreator):
    
    def __init__(self, size : int):
        self.size = size
        
    def create_data(self, index : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.randn(3, 1920, 1080)
        template = torch.ones(3, 120, 40)
        
        rand_x, rand_y = torch.randint(0, 1920 - 120, (1,)), torch.randint(0, 1080 - 40, (1,))
        image[ :, rand_x:rand_x+120, rand_y:rand_y+40] = 1
        
        heatmap = torch.zeros(1920, 1080)
        heatmap[rand_x:rand_x+120, rand_y:rand_y+40] = 1
        
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = torch.nn.functional.interpolate(heatmap, size=(self.size, self.size), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze(0).squeeze(0)
        
        return image, template, heatmap
    
    def sample_size(self) -> int:
        return 100
    
    
    
    
    
    
    
    
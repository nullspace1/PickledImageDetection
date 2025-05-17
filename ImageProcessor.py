import torch
import timm

class ImageProcessor(torch.nn.Module):
    
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
    def forward(self, x):
        return self.efficientnet(x)[-1]

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def test(self):
        x = torch.randn(1, 3, 1200, 1200)
        out = self.forward(x)
        print(f"Output shape for Image Processor: {out.shape}")
        print(f"Parameter count for Image Processor: {self.parameter_count():,}")

if __name__ == "__main__":
    image_processor = ImageProcessor()
    image_processor.test()

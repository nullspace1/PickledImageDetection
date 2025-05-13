import torch

class ImageProcessor(torch.nn.Module):
    
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).features

    def forward(self, x):
        return self.vgg(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def test(self):
        x = torch.randn(1, 3, 1920, 1080)
        print(self(x).shape)

if __name__ == "__main__":
    image_processor = ImageProcessor()
    image_processor.test()

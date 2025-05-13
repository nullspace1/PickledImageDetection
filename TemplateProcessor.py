import torch

class TemplateProcessor(torch.nn.Module):
    
    def __init__(self, intermediate_size, reduced_channels=512, final_kernel_channels = 3):
        super(TemplateProcessor, self).__init__()
        self.reduced_channels = reduced_channels
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).features
        self.channel_reduction = torch.nn.Conv2d(512, reduced_channels, kernel_size=1)
        self.linear = torch.nn.Linear(reduced_channels * 32 * 32, intermediate_size)
        self.linear2 = torch.nn.Linear(intermediate_size, 512 * 3 * 3 * final_kernel_channels)
        self.final_kernel_channels = final_kernel_channels
        
    def forward(self, x):
        output = self.vgg(x)
        output = torch.nn.functional.adaptive_avg_pool2d(output, (32, 32))
        output = self.channel_reduction(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        output = self.linear2(output)
        output = output.view(output.size(0), self.final_kernel_channels, 512, 3, 3)
        return output
    
    def test(self):
        x = torch.randn(1, 3, 1200, 1200)
        x = self.forward(x)
        print(x.shape)
        
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    template_processor = TemplateProcessor(1000)
    template_processor.test()
    print(f"Parameter count: {template_processor.parameter_count()}")
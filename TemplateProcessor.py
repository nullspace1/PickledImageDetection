import torch
import timm
import torch.nn as nn
import torch.nn.functional as F

class TemplateProcessor(nn.Module):
    def __init__(self, intermediate_size, final_kernel_channels=3):
        super().__init__()
        self.final_kernel_channels = final_kernel_channels
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)

        # Efficientnet[-1] output is (B, 320, H, W)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))         # (B, 320, 1, 1)
        self.linear1 = nn.Linear(320, intermediate_size)
        self.relu_1 = nn.ReLU()
        self.linear2 = nn.Linear(intermediate_size, final_kernel_channels * 320 * 3 * 3)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        if x.shape[2] < 256 or x.shape[3] < 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        features = self.efficientnet(x)[-1]              # (B, 320, H, W)
        pooled = self.pool(features).view(features.size(0), -1)  # (B, 320)
        x = self.relu_1(self.linear1(pooled))                         # (B, intermediate_size)
        x = self.relu_2(self.linear2(x))                              # (B, C*K*K*F)
        x = x.view(x.size(0), self.final_kernel_channels, 320, 3, 3)
        return x

    def test(self):
        x = torch.randn(1, 3, 1200, 1200)
        out = self.forward(x)
        print(f"Output shape for Template Processor: {out.shape}")
        print(f"Parameter count for Template Processor: {self.parameter_count():,}")

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    TemplateProcessor(300).test()

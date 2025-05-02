import torch
import torch.nn.functional as F

B, C, H, W = 30, 128, 512, 256
kh, kw = 128, 128

# Input: [B, C, H, W]
tensor_1 = torch.randn(B, C, H, W).cuda()
# Kernel: [B, C, kh, kw]
tensor_2 = torch.randn(B, C, kh, kw).cuda()

# Flip for cross-correlation (optional)
tensor_2 = torch.flip(tensor_2, [2, 3])

# Reshape
x = tensor_1.reshape(1, B * C, H, W)
kernel = tensor_2.reshape(B * C, 1, kh, kw)

# Convolve
out = F.conv2d(x, kernel, groups=B * C)
out = out.reshape(B, C, out.shape[-2], out.shape[-1])

print(out.shape)

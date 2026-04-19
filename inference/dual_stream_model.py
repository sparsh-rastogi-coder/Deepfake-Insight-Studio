import torch
import torch.nn as nn

# ==========================================
# 1. CBAM: Channel Attention Module
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        # Shared Multi-Layer Perceptron (MLP)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Combine the distinct features from both pooling methods
        out = avg_out + max_out
        return self.sigmoid(out)

# ==========================================
# 2. CBAM: Spatial Attention Module
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 7x7 convolution looks at a wide spatial area to find artifact boundaries
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compress the channel dimension to highlight spatial regions
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        out = self.conv(x_cat)
        return self.sigmoid(out)

# ==========================================
# 3. The Combined CBAM Block
# ==========================================
class CBAM_Block(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM_Block, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply Channel Attention first, then Spatial Attention
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# ==========================================
# 4. The High-Frequency Noise Extractor
# ==========================================
class HighFrequencyFilter(nn.Module):
    def __init__(self):
        super(HighFrequencyFilter, self).__init__()
        # A standard 3x3 Laplacian filter to extract microscopic edges and noise
        kernel = torch.tensor([[[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]]], dtype=torch.float32)
        
        # Apply the exact same filter to all 3 RGB channels independently
        kernel = kernel.repeat(3, 1, 1, 1) 
        self.filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        self.filter.weight.data = kernel
        
        # WE FREEZE THIS. The network is not allowed to change these weights. 
        # It must act as a pure mathematical noise extractor.
        self.filter.weight.requires_grad = False

    def forward(self, x):
        return self.filter(x)

# ==========================================
# 5. The Dual-Stream CBAM Architecture
# ==========================================
class DualStream_CBAM_CNN(nn.Module):
    def __init__(self):
        super(DualStream_CBAM_CNN, self).__init__()
        
        # --- STREAM A: The RGB Network ---
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), CBAM_Block(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), CBAM_Block(64), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), CBAM_Block(128), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), CBAM_Block(256), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), CBAM_Block(512), nn.MaxPool2d(2, 2)
        )
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)

        # --- STREAM B: The High-Frequency Noise Network ---
        self.noise_extractor = HighFrequencyFilter()
        # Shallower to save VRAM (only goes up to 128 channels)
        self.noise_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1)
        )

        # --- FUSION: The Classification Head ---
        self.fc1 = nn.Linear(512 + 128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Pass image through RGB stream
        out_rgb = self.rgb_pool(self.rgb_stream(x)).view(x.size(0), -1)
        
        # 2. Extract noise, then pass through Noise stream
        noise_img = self.noise_extractor(x)
        out_noise = self.noise_stream(noise_img).view(x.size(0), -1)
        
        # 3. Concatenate both feature vectors [Batch, 640]
        out_fused = torch.cat((out_rgb, out_noise), dim=1)
        
        # 4. Final Classification
        out = self.relu(self.fc1(out_fused))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

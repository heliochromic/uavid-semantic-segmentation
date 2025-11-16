import torch
import torch.nn as nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.conv(x)

        if self.dropout_p > 0.0:
            x = self.dropout(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.down = DoubleConvolution(in_channels, out_channels,dropout_p )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.down(x) 
        pool = self.pool(down)
        return down, pool
    

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.transposed_conv = DoubleConvolution(in_channels, out_channels, dropout_p)

    def forward(self, x, x_res):
        x = self.up(x)
        
        diff_h = x_res.size(2) - x.size(2)
        diff_w = x_res.size(3) - x.size(3)
        
        x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                  diff_h // 2, diff_h - diff_h // 2])
        
        x_concat = self.transposed_conv(torch.cat([x, x_res], 1))
        return x_concat
        


class UNet(nn.Module):
    def __init__(self, classes, dropout_prob: float = 0.2):
        super().__init__()
        
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        
        self.bottleneck = DoubleConvolution(512, 1024, dropout_p=dropout_prob * 1.5)

        self.up1 = Upsample(1024, 512, dropout_p=dropout_prob)
        self.up2 = Upsample(512, 256, dropout_p=dropout_prob * 0.5)
        self.up3 = Upsample(256, 128, dropout_p=0.0)
        self.up4 = Upsample(128, 64, dropout_p=0.0)
        
        self.final = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.down1(x)
        d2, p2 = self.down2(p1)
        d3, p3 = self.down3(p2)
        d4, p4 = self.down4(p3)
        
        b = self.bottleneck(p4)
        
        u1 = self.up1(b, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        
        output = self.final(u4)
        return output
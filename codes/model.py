import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_domains=3, num_pos = 3, init_features=64):
        """
        in_channels: 입력 MRI 채널 (1)
        out_channels: 출력 MRI 채널 (1)
        num_domains: 타겟 도메인 개수 (예: T2, FLAIR... -> 3)
        init_features: 32 (메모리 절약용)
        """
        super(Generator, self).__init__()

        features = init_features

        # 입력 채널 = 이미지(1) + 타겟 라벨(num_domains)
        self.encoder1 = self._block(in_channels + num_domains + num_pos, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x, label, pos):
        # 1. Label 확장 (Vector -> Spatial Map)
        # label: (Batch, num_domains) -> (Batch, num_domains, D, H, W)
        B, _, D, H, W = x.shape
        c = label.view(label.size(0), label.size(1), 1, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3), x.size(4))

        pos_map = pos.view(B, 3, 1, 1, 1).expand(-1, -1, D, H, W)

        # 2. 이미지 + 라벨 합체
        x_input = torch.cat([x, c, pos_map], dim=1)

        # 3. U-Net Pass (패딩 없이 바로 들어갑니다)
        enc1 = self.encoder1(x_input)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = torch.tanh(self.conv(dec1))

        return output

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features, affine=True),
            nn.ReLU(inplace=True),
        )


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_domains=3, init_features=64):
        super(Discriminator, self).__init__()

        features = init_features
        # 입력: Source(1) + Target(1) + Label(num_domains)
        total_in_channels = 1 + 1 + num_domains

        self.model = nn.Sequential(
            nn.Conv3d(total_in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(features * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(features * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(features * 4, features * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm3d(features * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x_source, x_target, label):
        # 1. Label 확장
        c = label.view(label.size(0), label.size(1), 1, 1, 1)
        c = c.repeat(1, 1, x_source.size(2), x_source.size(3), x_source.size(4))

        # 2. 합체 후 판별
        x_input = torch.cat([x_source, x_target, c], dim=1)
        return self.model(x_input)


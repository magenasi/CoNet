# coding=utf-8
"""
Author: xiezhenqing
date: 2022/8/4 13:42
desc:
"""

import torch
import torch.nn as nn


# 卷积块
# Conv2d + BatchNorm2d + LeakyRelu
class BasicConv(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 kernel_size,
				 stride=1):
		super(BasicConv, self).__init__()

		pad = (kernel_size - 1) // 2 if kernel_size else 0
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.activation = nn.LeakyReLU(0.1)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.activation(x)

		return x


class ResidualBlock(nn.Module):
	"""
	文献中使用的残差模块
	"""
	def __init__(self,
	             in_channel):
		super(ResidualBlock, self).__init__()
		self.CONV1 = BasicConv(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
		self.CONV2 = BasicConv(in_channels=in_channel, out_channels=in_channel, kernel_size=3)
		self.act = nn.LeakyReLU(0.1, inplace=True)

	def forward(self, x):
		identity = x
		return self.act(identity + self.CONV2(self.CONV1(x)))


class BasicBlock(nn.Module):
	"""
	ResNet模型中BasicBlock模块
	"""
	expansion = 1

	def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
							   kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channel)
		self.act = nn.LeakyReLU(0.1, inplace=True)
		self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
							   kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channel)
		self.downsample = downsample

	def forward(self, x):
		identity = x
		if self.downsample is not None:
			identity = self.downsample(x)

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += identity
		out = self.act(out)

		return out


class Bottleneck(nn.Module):
	"""
	ResNet模型中Bottleneck模块
	"""
	expansion = 4

	def __init__(self, in_channel, out_channel, stride=1, downsample=None,
				 groups=1, width_per_group=64):
		super(Bottleneck, self).__init__()

		width = int(out_channel * (width_per_group / 64.)) * groups

		self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
							   kernel_size=1, stride=1, bias=False)  # squeeze channels
		self.bn1 = nn.BatchNorm2d(width)
		# -----------------------------------------
		self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
							   kernel_size=3, stride=stride, bias=False, padding=1)
		self.bn2 = nn.BatchNorm2d(width)
		# -----------------------------------------
		self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
							   kernel_size=1, stride=1, bias=False)  # unsqueeze channels
		self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
		self.act = nn.LeakyReLU(0.1, inplace=True)
		self.downsample = downsample

	def forward(self, x):
		identity = x
		if self.downsample is not None:
			identity = self.downsample(x)

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.act(out)

		out = self.conv3(out)
		out = self.bn3(out)

		out += identity
		out = self.act(out)

		return out


class BackboneV1(nn.Module):
	"""
	利用ResNet50模型构建的骨架网络
	"""
	def __init__(self,
				 block,
				 blocks_num,
				 groups=1,
				 width_per_group=64):
		super(BackboneV1, self).__init__()
		self.in_channel = 64

		self.groups = groups
		self.width_per_group = width_per_group

		self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(self.in_channel)
		self.act = nn.LeakyReLU(0.1, inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, blocks_num[0])
		self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

	def _make_layer(self, block, channel, block_num, stride=1):
		downsample = None
		if stride != 1 or self.in_channel != channel * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride,
						  bias=False),
				nn.BatchNorm2d(channel * block.expansion)
			)

		layers = [block(self.in_channel,
						channel,
						downsample=downsample,
						stride=stride,
						groups=self.groups,
						width_per_group=self.width_per_group)]
		self.in_channel = channel * block.expansion

		for _ in range(1, block_num):
			layers.append(block(self.in_channel,
								channel,
								groups=self.groups,
								width_per_group=self.width_per_group))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)

		return x


class BackboneV2(nn.Module):
	"""
	文献中使用的骨干网络
	"""
	def __init__(self):
		super(BackboneV2, self).__init__()
		self.conv1 = BasicConv(3, 32, 3)
		self.conv2 = BasicConv(32, 64, 3, 2)
		self.block1 = ResidualBlock(64)
		self.conv3 = BasicConv(64, 128, 3, 2)
		self.block2 = nn.Sequential(
			ResidualBlock(128),
			ResidualBlock(128)
		)
		self.conv4 = BasicConv(128, 256, 3, 2)
		self.block3 = nn.Sequential(
			ResidualBlock(256),
			ResidualBlock(256),
			ResidualBlock(256),
			ResidualBlock(256)
		)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.block1(x)
		x = self.conv3(x)
		x = self.block2(x)
		x = self.conv4(x)
		x = self.block3(x)
		return x


class DCM(nn.Module):
	def __init__(self,
	             # Backbone选择，可选'BackboneV1'和'BackboneV2'，默认'BackboneV1'
	             backbone='BackboneV1',
	             shared=True,               # Ture表示共享参数，False表示不共享参数
	             # 特征融合算子方式，可选'subfuser'->语义差分算子，或者'convfuser'->卷积融合算子，默认'convfuser'
	             fuser='convfuser'
	             ):
		super(DCM, self).__init__()
		assert backbone in ['BackboneV1', 'BackboneV2']
		assert fuser in ['convfuser', 'subfuser']

		self.shared = shared
		self.fuser = fuser
		self.backbone = backbone

		# 采用语义差分算子
		if fuser == 'subfuser':
			self.subfuser = nn.Sigmoid()

		if shared:
			# 两个骨干分支使用同一个模型
			if backbone == 'BackboneV1':
				self.backbone = BackboneV1(Bottleneck, [3, 4])
				# 采用卷积融合算子
				self.convfuser = nn.Sequential(
					BasicConv(in_channels=1024, out_channels=512, kernel_size=1),
					BasicConv(in_channels=512, out_channels=512, kernel_size=3)
				)
			else:
				self.backbone = BackboneV2()
				# 采用卷积融合算子
				self.convfuser = nn.Sequential(
					BasicConv(in_channels=512, out_channels=256, kernel_size=1),
					BasicConv(in_channels=256, out_channels=256, kernel_size=3)
				)
		else:
			# 两个骨干分支分别使用一个模型
			if backbone == 'BackboneV1':
				self.backbone_1 = BackboneV1(Bottleneck, [3, 4])
				self.backbone_2 = BackboneV1(Bottleneck, [3, 4])
				# 采用卷积融合算子
				self.convfuser = nn.Sequential(
					BasicConv(in_channels=1024, out_channels=512, kernel_size=1),
					BasicConv(in_channels=512, out_channels=512, kernel_size=3)
				)
			else:
				self.backbone_1 = BackboneV2()
				self.backbone_2 = BackboneV2()
				# 采用卷积融合算子
				self.convfuser = nn.Sequential(
					BasicConv(in_channels=512, out_channels=256, kernel_size=1),
					BasicConv(in_channels=256, out_channels=256, kernel_size=3)
				)

	def forward(self, x):
		"""
		x.shape = [2, N, C, H, W], 2表示两个patch
		"""
		# 两个骨干分支的输入
		input_1 = x[0]
		input_2 = x[1]
		if self.shared:
			feature_1 = self.backbone(input_1)
			feature_2 = self.backbone(input_2)

			# test
			# print('features.shape:{}'.format(feature_1.cpu().shape))

			if self.fuser == 'convfuser':
				features = torch.cat((feature_1, feature_2), 1)
				# test
				# print('features.shape:{}'.format(features.cpu().shape))
				return self.convfuser(features), feature_2
			else:
				return self.subfuser(feature_2 - feature_1), feature_2

		else:
			feature_1 = self.backbone_1(input_1)
			feature_2 = self.backbone_2(input_2)

			# test
			# print('features.shape:{}'.format(feature_1.cpu().shape))

			if self.fuser == 'convfuser':
				features = torch.cat((feature_1, feature_2), 1)
				# test
				# print('features.shape:{}'.format(features.cpu().shape))
				return self.convfuser(features), feature_2
			else:
				return self.subfuser(feature_2 - feature_1), feature_2


if __name__ == '__main__':
	model1 = DCM(backbone='BackboneV1', shared=True, fuser='convfuser').eval()
	model2 = DCM(backbone='BackboneV1', shared=False, fuser='convfuser').eval()
	model3 = DCM(backbone='BackboneV1', shared=True, fuser='subfuser').eval()
	model4 = DCM(backbone='BackboneV1', shared=False, fuser='subfuser').eval()
	model5 = DCM(backbone='BackboneV2', shared=True, fuser='convfuser').eval()
	model6 = DCM(backbone='BackboneV2', shared=False, fuser='convfuser').eval()
	model7 = DCM(backbone='BackboneV2', shared=True, fuser='subfuser').eval()
	model8 = DCM(backbone='BackboneV2', shared=False, fuser='subfuser').eval()
	inputs = torch.randn((2, 8, 3, 224, 224), dtype=torch.float32)
	for model in [model1, model2, model3, model4, model5, model6, model7, model8]:
		out1, out2 = model(inputs)
		print(out1.shape, out2.shape)

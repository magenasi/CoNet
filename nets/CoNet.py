# coding=utf-8
"""
Author: xiezhenqing
date: 2022/8/4 13:40
desc: 定义 CoNet 模型，将文献中的目标检测层改为简单的分类任务线性层，可选骨干模型
	  ‘BackboneV1’表示ResNet50下采样至8倍时的模型，‘BackboneV2’表示文献所用模型

	  后续改进点-->
	  特征融合阶段主通道可否用ResNet50后半段代替？
	  用 yolov4 检测头替换线性层实现目标检测
	  将特征融合阶段次通道改为可动态添加或删除，以实现消融实验
"""

import torch
import torch.nn as nn

from DCM import BasicConv, ResidualBlock, DCM


class OutputBlock(nn.Module):
	def __init__(self,
	             in_channels,
	             num_classes
	             ):
		super(OutputBlock, self).__init__()

		self.convsets = nn.Sequential(
			BasicConv(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1),
			BasicConv(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3),
			BasicConv(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1),
			BasicConv(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3),
			BasicConv(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1)
		)

		self.conv_for_input2 = BasicConv(in_channels=in_channels, out_channels=in_channels // 2,
		                                 kernel_size=1)
		self.conv1 = BasicConv(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
		                       stride=1)
		self.avg = nn.AdaptiveAvgPool2d((1, 1))
		self.linear = nn.Linear(in_features=in_channels, out_features=num_classes)

	def forward(self, input1, input2):
		"""
		定义Output Block，连接主通路和次通路，主通路经过self.convsets后进入两个分支：
		一个作为中间值被输入到上采样模块，另一个与次通路的检测特征图拼接，用于预测结果计算
		:param input1: 主通路输入
		:param input2: 次通路输入
		:return: output1: 输入到上采样层
				 output2：与次通路的特征图拼接，用于预测结果计算
		"""
		outputs1 = self.convsets(input1)
		input2 = self.conv_for_input2(input2)
		features = torch.concat([outputs1, input2], 1)
		features = self.conv1(features)
		features = self.avg(features)
		features = torch.flatten(features, 1)
		outputs2 = self.linear(features)

		return outputs1, outputs2


class OutputBlockDet(nn.Module):
	def __init__(self,
	             in_channels,
	             num_classes
	             ):
		super(OutputBlockDet, self).__init__()

		self.convsets = nn.Sequential(
			BasicConv(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1),
			BasicConv(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3),
			BasicConv(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1),
			BasicConv(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3),
			BasicConv(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1)
		)

		self.conv_for_input2 = BasicConv(in_channels=in_channels, out_channels=in_channels // 2,
		                                 kernel_size=1)
		self.conv1 = BasicConv(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
		                       stride=1)
		self.detectlayer = nn.Conv2d(in_channels, 3*(5+num_classes), 1, 1)

	def forward(self, input1, input2):
		"""
		定义Output Block，连接主通路和次通路，主通路经过self.convsets后进入两个分支：
		一个作为中间值被输入到上采样模块，另一个与次通路的检测特征图拼接，用于预测结果计算
		:param input1: 主通路输入
		:param input2: 次通路输入
		:return: output1: 输入到上采样层
				 output2：与次通路的特征图拼接，用于预测结果计算
		"""
		outputs1 = self.convsets(input1)
		input2 = self.conv_for_input2(input2)
		features = torch.concat([outputs1, input2], 1)
		features = self.conv1(features)
		outputs2 = self.detectlayer(features)
		return outputs1, outputs2


class UpsampleConcat(nn.Module):
	def __init__(self, in_channels):
		super(UpsampleConcat, self).__init__()
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
		self.conv_for_concat = BasicConv(in_channels * 2, in_channels, 1, 1)

	def forward(self, input1, input2):
		input1 = self.upsample(input1)
		outputs = torch.concat([input1, input2], 1)
		outputs = self.conv_for_concat(outputs)

		return outputs


class MinorPath(nn.Module):
	def __init__(self, in_channels):
		super(MinorPath, self).__init__()
		self.ResidualBlocks_1 = ResidualBlock(in_channels)
		self.conv1 = BasicConv(in_channels, in_channels * 2, 3, 2)
		self.ResidualBlocks_2 = ResidualBlock(in_channels * 2)
		self.conv2 = BasicConv(in_channels * 2, in_channels * 4, 3, 2)
		self.ResidualBlocks_3 = ResidualBlock(in_channels * 4)

	def forward(self, x):
		# N, 256, 28, 28 -> N, 256, 28, 28
		outs1 = self.ResidualBlocks_1(x)
		# N, 256, 28, 28 -> N, 512, 14, 14
		x = self.conv1(outs1)
		# N, 512, 14, 14 -> N, 512, 14, 14
		outs2 = self.ResidualBlocks_2(x)
		# N, 512, 14, 14 -> N, 1024, 7, 7
		x = self.conv2(outs2)
		# N, 1024, 7, 7 -> N, 1024, 7, 7
		outs3 = self.ResidualBlocks_3(x)

		return outs1, outs2, outs3


class MsCDM(nn.Module):
	def __init__(self, in_channels, num_classses):
		super(MsCDM, self).__init__()
		self.MinorPath = MinorPath(in_channels)
		self.ResidualBlocks_1 = nn.Sequential(
			ResidualBlock(in_channels),
			ResidualBlock(in_channels),
			ResidualBlock(in_channels),
			ResidualBlock(in_channels)
		)
		self.upsampleconcat1 = UpsampleConcat(in_channels)
		self.conv1 = BasicConv(in_channels, in_channels * 2, 3, 2)
		self.ResidualBlocks_2 = nn.Sequential(
			ResidualBlock(in_channels * 2),
			ResidualBlock(in_channels * 2),
			ResidualBlock(in_channels * 2),
			ResidualBlock(in_channels * 2),
			ResidualBlock(in_channels * 2),
			ResidualBlock(in_channels * 2),
			ResidualBlock(in_channels * 2),
			ResidualBlock(in_channels * 2)
		)
		self.upsampleconcat2 = UpsampleConcat(in_channels * 2)
		self.conv2 = BasicConv(in_channels * 2, in_channels * 4, 3, 2)
		self.ResidualBlocks_3 = nn.Sequential(
			ResidualBlock(in_channels * 4),
			ResidualBlock(in_channels * 4),
			ResidualBlock(in_channels * 4),
			ResidualBlock(in_channels * 4)
		)

		# OutputBlock
		self.outputblock3 = OutputBlock(in_channels * 4, num_classses)
		self.outputblock2 = OutputBlock(in_channels * 2, num_classses)
		self.outputblock1 = OutputBlock(in_channels, num_classses)
		# OutputBlockDet
		self.outputblockdet3 = OutputBlockDet(in_channels * 4, num_classses)
		self.outputblockdet2 = OutputBlockDet(in_channels * 2, num_classses)
		self.outputblockdet1 = OutputBlockDet(in_channels * 1, num_classses)

	def forward(self, feature1, feature2):
		# 次通路
		# [N, 256, 28, 28], [N, 512, 14, 14], [N, 1024, 7, 7]
		minor_p1, minor_p2, minor_p3 = self.MinorPath(feature2)

		# 主通路
		# N, 256, 28, 28 -> N, 256, 28, 28
		major_p1 = self.ResidualBlocks_1(feature1)
		# N, 256, 28, 28 -> N, 512, 14, 14
		feature1 = self.conv1(major_p1)
		# N, 512, 14, 14 -> N, 512, 14, 14
		major_p2 = self.ResidualBlocks_2(feature1)
		# N, 512, 14, 14 -> N, 1024, 7, 7
		feature1 = self.conv2(major_p2)
		# N, 1024, 7, 7 -> N, 1024, 7, 7
		major_p3 = self.ResidualBlocks_3(feature1)
		# [N, 512, 7, 7], [N, num_classes]
		major_p3, outs3 = self.outputblockdet3(major_p3, minor_p3)
		# N, 512, 7, 7 -> N, 512, 14, 14
		major_p3_upsample = self.upsampleconcat2(major_p3, major_p2)
		# [N, 256, 14, 14], [N, num_classes]
		major_p2, outs2 = self.outputblockdet2(major_p3_upsample, minor_p2)
		# N, 256, 14, 14 -> N, 256, 28, 28
		major_p2_upsample = self.upsampleconcat1(major_p2, major_p1)
		# [], [N, num_classes]
		_, outs1 = self.outputblockdet1(major_p2_upsample, minor_p1)

		return outs1, outs2, outs3


class CoNet(nn.Module):
	def __init__(self,
	             backbone: str = 'BackboneV1',
	             shared: bool = True,
	             fuser: str = 'convfuser',
	             num_classes: int = 1000):
		super(CoNet, self).__init__()

		assert backbone in ['BackboneV1', 'BackboneV2']
		assert fuser in ['convfuser', 'subfuser']

		self.DCM = DCM(backbone, shared, fuser)
		self.MsCDM = MsCDM(512, num_classes) if backbone == 'BackboneV1' else MsCDM(256,
		                                                                            num_classes)

	def forward(self, x):
		"""
		x.shape = [2, N, C, H, W], 2表示两个patch
		"""

		# ref_features 表示语义关系图， det_features 表示检测特征图
		ref_features, det_features = self.DCM(x)
		outs1, outs2, outs3 = self.MsCDM(ref_features, det_features)

		return outs1, outs2, outs3


if __name__ == '__main__':
	model = CoNet()
	inputs = torch.randn(2, 8, 3, 224, 224)
	out1, out2, out3 = model(inputs)
	print(out1.shape)
	print(out2.shape)
	print(out3.shape)

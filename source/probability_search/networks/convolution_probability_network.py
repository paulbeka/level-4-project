import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProbabilityFinder(nn.Module):

	def __init__(self, batch_size):
		# stick to 3x3 and 1x1 conv at the end
		self.batch_size = batch_size
		super(ProbabilityFinder, self).__init__()
		self.conv1 = nn.Conv2d(1, 3, 3, padding="same")
		self.conv2 = nn.Conv2d(3, 12, 3, padding="same")
		self.conv3 = nn.Conv2d(12, 24, 5, padding="same")
		self.conv4 = nn.Conv2d(24, 48, 5, padding="same")
		self.finalConv = nn.Conv2d(48, 1, 1)

		self.scorePooling = nn.AdaptiveAvgPool2d(100)
		self.scoreFC1 = nn.Linear(100*100, 100)
		self.scoreFC2 = nn.Linear(100, 1)

		self.relu = nn.ReLU()
		self.silu = nn.SiLU()
		

	def forward(self, x):
		x = self.silu(self.conv1(x))
		x = self.silu(self.conv2(x))
		x = self.silu(self.conv3(x))
		x = self.silu(self.conv4(x))
		x = self.finalConv(x)

		score = self.scorePooling(x)
		score = score.flatten()
		score = self.relu(self.scoreFC1(score))
		score = self.scoreFC2(score)

		x = torch.flatten(x)
		x = torch.cat((score, x), 0)
		x = x.unsqueeze(0)

		return x
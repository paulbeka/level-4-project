import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProbabilityFinder(nn.Module):

	def __init__(self, batch_size):
		self.batch_size = batch_size
		super(ProbabilityFinder, self).__init__()
		self.conv1	= nn.Conv2d(1, 3, 3, padding=1)
		self.conv2 = nn.Conv2d(3, 12, 3, padding=1)
		self.conv3 = nn.Conv2d(12, 24, 5, padding=2)
		self.conv4 = nn.Conv2d(24, 48, 5, padding=2)
		self.finalConv = nn.Conv2d(48, 1, 1)
		self.silu = nn.SiLU()
		

	def forward(self, x):
		x = self.silu(self.conv1(x))
		x = self.silu(self.conv2(x))
		x = self.silu(self.conv3(x))
		x = self.silu(self.conv4(x))
		x = self.finalConv(x)
		return x
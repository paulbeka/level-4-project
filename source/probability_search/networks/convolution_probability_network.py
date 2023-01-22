import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProbabilityFinder(nn.Module):

	def __init__(self, batch_size):
		self.batch_size = batch_size
		super(ProbabilityFinder, self).__init__()
		self.conv1	= nn.Conv2d(1, 3, batch_size)
		self.conv2 = nn.Conv2d(3, 5, batch_size)
		self.conv3 = nn.Conv2d(5, 7, batch_size)
		self.finalConv = nn.Conv2d(7, 1, batch_size)
		self.silu = nn.SiLU()
		

	def forward(self, x):
		x = self.silu(self.conv1(x))
		x = self.silu(self.conv2(x))
		x = self.silu(self.conv3(x))
		x = self.finalConv(x)
		return x
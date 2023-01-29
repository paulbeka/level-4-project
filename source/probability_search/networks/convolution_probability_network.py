import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: add a scoring function

class ProbabilityFinder(nn.Module):

	def __init__(self, batch_size):
		# stick to 3x3 and 1x1 conv at the end
		self.batch_size = batch_size
		super(ProbabilityFinder, self).__init__()
		self.conv1 = nn.Conv2d(1, 3, 3, padding="same")
		self.conv2 = nn.Conv2d(3, 12, 3, padding="same")
		self.conv3 = nn.Conv2d(12, 24, 3, padding="same")
		self.conv4 = nn.Conv2d(24, 48, 3, padding="same")
		self.finalConv = nn.Conv2d(48, 1, 1)

		self.scoreFC1 = nn.Linear()
		self.scoreFC2 = nn.Linear()
		self.score = nn.Linear()

		self.silu = nn.SiLU()
		self.relu = nn.ReLU()
		

	def forward(self, x):
		x = self.silu(self.conv1(x))
		x = self.silu(self.conv2(x))
		x = self.silu(self.conv3(x))
		x = self.silu(self.conv4(x))		
		x = self.finalConv(x)
		
		score = self.scoreFC1(x)
		score = self.scoreFC2(score)

		return x
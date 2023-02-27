import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CHANGE THIS NETWORK TYPE COMPLETELY
class ScoreFinder(nn.Module):

	def __init__(self, batch_size):
		super(ScoreFinder, self).__init__()

		# use conv
		self.conv1 = nn.Conv2d(1, 3, 3, padding="same")
		self.conv2 = nn.Conv2d(1, 3, 3, padding="same")

		self.scorePooling = nn.AdaptiveAvgPool2d(15)
		self.scoreFC1 = nn.Linear(15*15, 1)

		self.relu = nn.ReLU()
		self.silu = nn.SiLU()


	def forward(self, x):

		score = self.silu(self.conv1(x))
		score = self.scorePooling(x)
		score = score.flatten()
		score = self.relu(self.scoreFC1(score))

		return score
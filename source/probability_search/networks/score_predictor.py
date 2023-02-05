import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScoreFinder(nn.Module):

	def __init__(self, batch_size):
		super(ScoreFinder, self).__init__()

		self.scorePooling = nn.AdaptiveAvgPool2d(100)
		self.scoreFC1 = nn.Linear(100*100, 100)
		self.scoreFC2 = nn.Linear(100, 1)

		self.relu = nn.ReLU()
		self.silu = nn.SiLU()
		

	def forward(self, x):

		score = self.scorePooling(x)
		score = score.flatten()
		score = self.relu(self.scoreFC1(score))
		score = self.scoreFC2(score)

		return score
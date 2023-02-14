import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CHANGE THIS NETWORK TYPE COMPLETELY
class ScoreFinder(nn.Module):

	def __init__(self, batch_size):
		super(ScoreFinder, self).__init__()

		self.scorePooling = nn.AdaptiveAvgPool2d(100)
		self.scoreFC1 = nn.Linear(100*100, 100)
		self.scoreFC2 = nn.Linear(100, 10)
		self.scoreFC3 = nn.Linear(10, 1)

		self.relu = nn.ReLU()
		

	def forward(self, x):

		score = self.scorePooling(x)
		score = score.flatten()
		score = self.relu(self.scoreFC1(score))
		score = self.relu(self.scoreFC2(score))
		score = self.scoreFC3(score)

		return score
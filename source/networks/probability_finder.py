import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProbabilityFinder(nn.Module):

	def __init__(self, batch_size):
		super(ProbabilityFinder, self).__init__()
		self.conv1	= nn.Conv2d(1, 3, batch_size)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(3, 16, batch_size)
		self.pool2 = nn.AdaptiveAvgPool2d((22, 22))
		self.fc1 = nn.Linear(16*22*22, 100*100)
		self.fc2 = nn.Linear(100*100, 100*100)
		

	def forward(self, x):
		dimx, dimy = x.shape[1], x.shape[2]
		x = torch.tensor(np.expand_dims(x, axis=1))
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = x.reshape(7744)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = x.reshape(1, 100, 100)
		x = torch.nn.AdaptiveAvgPool2d((dimx, dimy))(x)
		return x
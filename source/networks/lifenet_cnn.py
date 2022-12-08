import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LifeNetCNN(nn.Module):

	def __init__(self, num_classes, batch_size):
		super(LifeNetCNN, self).__init__()
		self.conv1	= nn.Conv2d(1, 3, batch_size)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(3, 16, batch_size)
		self.pool2 = nn.AdaptiveAvgPool2d((22, 22))
		self.fc1 = nn.Linear(16*22*22, 100)
		self.fc2 = nn.Linear(100, 84)
		self.fc3 = nn.Linear(84, num_classes)

		self.testFC = nn.Linear(100*100, 100)
		self.testFC2 = nn.Linear(100, 84)
		self.testFC3 = nn.Linear(84, num_classes)


	def forward(self, x):
		x = torch.tensor(np.expand_dims(x, axis=1))
		# x = self.pool(F.relu(self.conv1(x)))
		# x = self.pool2(F.relu(self.conv2(x)))
		x = x.view(-1, 100*100)
		# x = F.relu(self.fc1(x))
		# x = F.relu(self.fc2(x))
		# x = self.fc3(x)
		x = F.relu(self.testFC(x))
		x = F.relu(self.testFC2(x))
		x = self.testFC3(x)
		return x
import torch
import torch.nn as nn
import numpy as np
from game_of_life import Game


# hyperparameters
num_epochs = 5
batch_size = 5
learning_rate = 0.001


# Get all the gliders known and split the dataset
# feed nn a mix of gliders and not gliders
# make it detect when it is a glider or not
# given an input make it output a glider? How?

train_dataset = []
test_dataset = []

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class LifeNet(nn.Module):

	def __init__(self):
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 2)


# fitness = final(complexity) * max(complexity)
def objectiveFunction(object):
	self.conv1 = None


if __name__ == '__main__':
	main()
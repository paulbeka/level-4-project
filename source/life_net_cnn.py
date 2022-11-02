import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game_of_life import Game
from tools.rle_reader import RleReader
from dataloaders.spaceshipid_dataloader import SpaceshipIdentifierDataLoader


### HYPERPARAMETERS ###
num_epochs = 1
batch_size = 5
learning_rate = 0.001

width, height = 100, 100

### LOAD DATA ###

dataloader = SpaceshipIdentifierDataLoader(1000, 0.5, False, "C:\\Workspace\\level-4-project\\source\\data\\spaceship_identification", batch_size=batch_size)
train_loader, test_loader = dataloader.loadSpaceshipIdentifierDataset(width, height)

### NEURAL NET ###

num_classes = 2

class LifeNetCNN(nn.Module):

	def __init__(self, num_classes):
		super(LifeNetCNN, self).__init__()
		self.conv1	= nn.Conv2d(1, 3, batch_size)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(3, 16, batch_size)
		# reaches 9x9
		self.fc1 = nn.Linear(16*22*22, 100)
		self.fc2 = nn.Linear(100, 84)
		self.fc3 = nn.Linear(84, num_classes)


	def forward(self, x):
		x = torch.tensor(np.expand_dims(x, axis=1))
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*22*22)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


model = LifeNetCNN(num_classes).double()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (configs, labels) in enumerate(train_loader):

		outputs = model(configs)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 10 == 0:
			print(f"Loss: {loss.item():.9f}")


### TESTING ###

with torch.no_grad():
	correct, samples = 0, 0
	displayList = []
	for configs, labels in test_loader:
		outputs = model(configs)

		_, predictions = torch.max(outputs, 1)

		displayList += [(configs[i], predictions[i]) for i in range(batch_size)]
		samples += labels.shape[0]
		correct += (predictions == labels).sum().item()

	accuracy = 100 * (correct / samples)
	print(f"Accuracy: {accuracy:.2f}%")

	game = Game(width, height)
	game.renderItemList(displayList)
	game.run()

# run
if __name__ == '__main__':
	pass
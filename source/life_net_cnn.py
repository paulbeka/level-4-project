import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game_of_life import Game
from tools.rle_reader import RleReader


### HYPERPARAMETERS ###
num_epochs = 5
batch_size = 5
learning_rate = 0.001

### LOAD DATA ###
ratio = 5 
codeReader = RleReader("C:\\Workspace\\level-4-project\\source\\data\\30_30_all_spaceships.txt", box=30)

glider_dataset = [(item, 1) for item in codeReader.getFileArray()]
codeReader.fileName = "C:\\Workspace\\level-4-project\\source\\data\\random_rle.txt"
fake_dataset = [(item, 0) for item in codeReader.getFileArray()]

train_dataset = glider_dataset[0:len(glider_dataset) - (len(glider_dataset)//5)]
train_dataset += fake_dataset[0:len(fake_dataset) - (len(fake_dataset)//5)]
test_dataset = glider_dataset[len(glider_dataset) - (len(glider_dataset)//5): len(glider_dataset)]
test_dataset += fake_dataset[len(fake_dataset) - (len(fake_dataset)//5): len(fake_dataset)]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

### NEURAL NET ###

input_size = 30 * 30
hidden_size = 100
num_classes = 2

class LifeNetCNN(nn.Module):

	def __init__(self, input_size, hidden_size, num_classes):
		super(LifeNetCNN, self).__init__()
		self.conv1	= nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		# reaches 9x9
		self.fc1 = nn.Linear(16*5*5, 100)
		self.fc2 = nn.Linear(100, 84)
		self.fc3 = nn.Linear(84, num_classes)


	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


model = LifeNetCNN(input_size, hidden_size, num_classes).double()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (configs, labels) in enumerate(train_loader):
		configs = configs.reshape(-1, 30*30)

		outputs = model(configs)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


### TESTING ###

with torch.no_grad():
	correct, samples = 0, 0
	for configs, labels in test_loader:
		configs = configs.reshape(-1, 30*30)
		outputs = model(configs)

		_, predictions = torch.max(outputs, 1)
		samples += labels.shape[0]
		correct += (predictions == labels).sum().item()

	accuracy = 100 * (correct / samples)
	print(f"Accuracy: {accuracy:.2f}%")


# run
if __name__ == '__main__':
	pass
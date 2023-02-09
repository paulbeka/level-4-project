import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

# from networks.probability_finder import ProbabilityFinder
from networks.score_predictor import ScoreFinder
from dataloaders.score_dataloader import scoreDataloader


if not torch.cuda.is_available():
	print("GPU IS NOT AVAILABLE AND HAS BEEN IMPROPERLY CONFIGURED.")
	print("INSTALL THE NVIDIA DRIVER AND RETRY.")
	print("EXITING.")
	quit()


### HYPERPARAMETERS ###
num_epochs = 10
batch_size = 1
learning_rate = 0.00005

model =  ScoreFinder(batch_size).double()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
# model.to(device)
model.cuda(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


### DATA LOADING ###
train_loader = scoreDataloader(100) #number of fake items per ship
print("Data loaded.")


### NEURAL NET ###
total_steps = len(train_loader)
for epoch in range(num_epochs):
	print(f"Epoch: {epoch+1}/{num_epochs}")
	for i, (configs, labels) in tqdm(enumerate(train_loader), desc="Training: ", total=len(train_loader)):
		# load data into GPU
		# configs, labels = configs.to(device), labels.to(device)
		
		outputs = model(configs)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print(f"Saving epoch {epoch+1}...")

	torch.save(model.state_dict(), f"deconstructScoreOutputFile_{epoch+1}")


### TESTING ###


print("EXECUTION OVER.")

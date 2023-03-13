import os
import random
import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

from neuralnetwork import CNN
from dataloader import get_dataloader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnn = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
valid_losses = []
num_epochs = 30


# Dataloaders
train_loader = get_dataloader(split='train', is_augmentation=True)
valid_loader = get_dataloader(split='valid')
test_loader = get_dataloader(split='test')


for epoch in range(num_epochs):
	losses = []

	# Train
	cnn.train()
	training_loop = tqdm(train_loader)

	for (wav, genre_index) in training_loop:
		wav = wav.to(device)
		genre_index = genre_index.to(device)

		# Forward
		out = cnn(wav)
		loss = loss_function(out, genre_index)

		# Backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
	print('Epoch: [%d/%d], Train loss: %.4f' % (epoch+1, num_epochs, np.mean(losses)))

	# Validation
	cnn.eval()
	y_true = []
	y_pred = []
	losses = []
	for wav, genre_index in valid_loader:
		wav = wav.to(device)
		genre_index = genre_index.to(device)

		# reshape and aggregate chunk-level predictions
		b, c, t = wav.size()
		logits = cnn(wav.view(-1, t))
		logits = logits.view(b, c, -1).mean(dim=1)
		loss = loss_function(logits, genre_index)
		losses.append(loss.item())
		_, pred = torch.max(logits.data, 1)

		# append labels and predictions
		y_true.extend(genre_index.tolist())
		y_pred.extend(pred.tolist())
	accuracy = accuracy_score(y_true, y_pred)
	valid_loss = np.mean(losses)
	print('Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f' % (epoch+1, num_epochs, valid_loss, accuracy))

	# Save model
	valid_losses.append(valid_loss.item())
	if np.argmin(valid_losses) == epoch:
		print('Saving the best model at %d epochs!' % epoch)
		torch.save(cnn.state_dict(), './save/best_model.ckpt')
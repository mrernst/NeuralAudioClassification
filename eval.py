import os
import random
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from neuralnetwork import CNN
from dataloader import get_dataloader, GTZAN_GENRES

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnn = CNN().to(device)

# Load the best model
S = torch.load('./save/best_model.ckpt')
cnn.load_state_dict(S)
print('[INFO]:\tModel loaded!')

# Dataloaders
train_loader = get_dataloader(split='train', is_augmentation=True)
valid_loader = get_dataloader(split='valid')
test_loader = get_dataloader(split='test')


# Run evaluation
cnn.eval()
y_true = []
y_pred = []

with torch.no_grad():
	for wav, genre_index in test_loader:
		wav = wav.to(device)
		genre_index = genre_index.to(device)

		# reshape and aggregate chunk-level predictions
		b, c, t = wav.size()
		logits = cnn(wav.view(-1, t))
		logits = logits.view(b, c, -1).mean(dim=1)
		_, pred = torch.max(logits.data, 1)

		# append labels and predictions
		y_true.extend(genre_index.tolist())
		y_pred.extend(pred.tolist())


accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, xticklabels=GTZAN_GENRES, yticklabels=GTZAN_GENRES, cmap='YlGnBu')
plt.show()
print('Accuracy: %.4f' % accuracy)
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from data.dataset import CustomDataset, readAndPreprocessDataset  
from train.training import train, validate, save_batch_losses_to_csv, save_metrics_to_csv, plot_and_save_loss  # Importazione dal modulo comune
from config.config import MODEL, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS, NUM_CLASSES, LEARNING_RATE
import logging
from sklearn import metrics
from models.bert import BERTClass
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

# Load and preprocess dataset
trainSet, testSet, valSet = readAndPreprocessDataset()

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
               }

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
              }

training_loader = DataLoader(trainSet, **train_params)
testing_loader = DataLoader(testSet, **test_params)
validation_loader = DataLoader(valSet, **test_params)

model = BERTClass(num_classes=NUM_CLASSES) #Change the model here to the one you want to use
model.to(DEVICE)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Create the directory to save the state dicts and plots
os.makedirs(f'./stateDict_{MODEL}', exist_ok=True)

batch_losses = []
epoch_losses = []
epoch_accuracies = []
all_batch_losses = []  # Add this list to store all batch losses
best_val_loss = float('inf')

print("Starting training")

for epoch in range(EPOCHS):
    print(f"Starting Epoch: {epoch}")
    avg_train_loss, batch_losses = train(model, training_loader, optimizer, epoch, logger, DEVICE)
    avg_val_loss, outputs, targets = validate(model, validation_loader, DEVICE)

    epoch_losses.append(avg_train_loss)
    all_batch_losses.extend(batch_losses)  # Collect batch losses
    val_accuracy = metrics.accuracy_score(np.array(targets), np.array(outputs) > 0.5)
    epoch_accuracies.append(val_accuracy)

    if avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), f'./stateDict_{MODEL}/best_model_{MODEL}.pth')
        best_val_loss = avg_val_loss

    logger.info(f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")

save_metrics_to_csv(epoch_losses, epoch_accuracies)
plot_and_save_loss(all_batch_losses)
save_batch_losses_to_csv(all_batch_losses)

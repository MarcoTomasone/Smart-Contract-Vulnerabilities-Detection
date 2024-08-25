# train/training.py
import torch
from sklearn import metrics
from config.config import EPOCHS, MODEL
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train(model, training_loader, optimizer, epoch, logger, DEVICE):
    model.train()
    total_loss = 0.0
    batch_losses = []
    progress_bar = tqdm(enumerate(training_loader), total=len(training_loader))

    for batch_idx, data in progress_bar:
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)
        # Check if token_type_ids exists in the data
        if 'token_type_ids' in data:
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
            # Forward pass including token_type_ids
            outputs = model(ids, mask, token_type_ids)
        else:
            # Forward pass without token_type_ids
            outputs = model(ids, mask)
       
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_losses.append(loss.item())  # Add this line to track batch loss

        if batch_idx % 10 == 0:
            progress_bar.set_description(f'Epoch {epoch}, Loss: {loss.item()}')

    avg_loss = total_loss / len(training_loader)
    logger.info(f'Epoch {epoch}, Average Loss: {avg_loss}')
    torch.save(model.state_dict(), f'./stateDict_{MODEL}/epoch_{epoch}_{MODEL}.pth')

    return avg_loss, batch_losses  # Return batch losses


def validate(model, validation_loader, DEVICE):
    model.eval()
    fin_targets = []
    fin_outputs = []
    total_loss = 0.0

    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.float)

            # Check if token_type_ids exists in the data
            if 'token_type_ids' in data:
                token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
                # Forward pass including token_type_ids
                outputs = model(ids, mask, token_type_ids)
            else:
                # Forward pass without token_type_ids
                outputs = model(ids, mask)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    avg_loss = total_loss / len(validation_loader)
    return avg_loss, fin_outputs, fin_targets

def save_metrics_to_csv(epoch_losses, epoch_accuracies, filename='train-metrics.csv'):
    df = pd.DataFrame({'Epoch': list(range(1, EPOCHS + 1)), 'Loss': epoch_losses, 'Accuracy': epoch_accuracies})
    df.to_csv(filename, index=False)

def plot_and_save_loss(epoch_losses, filename='loss_plot.png'):
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def save_batch_losses_to_csv(batch_losses, filename='batch-losses.csv'):
    df = pd.DataFrame({'Batch': list(range(1, len(batch_losses) + 1)), 'Loss': batch_losses})
    df.to_csv(filename, index=False)
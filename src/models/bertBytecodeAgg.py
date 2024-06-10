import numpy as np
import pandas as pd
import re
import torch
import transformers
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from tqdm import tqdm
import logging
from sklearn import metrics

# Constants
MAX_LEN = 512
CODE_BLOCKS = 2  # Number of 512-token blocks
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
SAFE_IDX = 4
MODEL = 'bertBytecode'
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper Functions
def oneHotEncodeLabel(label):
    one_hot = np.zeros(NUM_CLASSES)
    for elem in label:
        if elem < SAFE_IDX:
            one_hot[elem] = 1
        elif elem > SAFE_IDX:
            one_hot[elem-1] = 1
    return one_hot

def transformData(example):
    bytecode = example["bytecode"]
    if bytecode.startswith("0x"):
        bytecode = bytecode[2:]
    bytecode_tokens = [bytecode[i:i+2] for i in range(0, len(bytecode), 2)]
    bytecode_text = ' '.join(bytecode_tokens)  # Create a string with spaces between tokens
    return {
        "bytecode": bytecode_text,
        "label": oneHotEncodeLabel(example["slither"])
    }

def readAndPreprocessDataset():
    train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir="./cache", ignore_verifications=True).map(transformData)
    test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir="./cache", ignore_verifications=True).map(transformData)
    val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir="./cache", ignore_verifications=True).map(transformData)
    return train_set, test_set, val_set

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN * CODE_BLOCKS):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.bytecode = dataframe["bytecode"]
        self.targets = dataframe["label"]

    def __len__(self):
        return len(self.bytecode)

    def __getitem__(self, index):
        bytecode = str(self.bytecode[index])

        if len(bytecode) == 0:
            raise ValueError(f"Bytecode at index {index} is empty.")

        encoding = self.tokenizer.encode_plus(
            bytecode,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ids = encoding['input_ids'].flatten()
        mask = encoding['attention_mask'].flatten()

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# BERT Model Class
class BERTAggregatedClass(torch.nn.Module):
    def __init__(self, num_classes, aggregation='mean', dropout=0.3):
        super(BERTAggregatedClass, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir="./cache")
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        self.aggregation = aggregation

    def forward(self, input_ids, attention_masks):
        batch_size, seq_len = input_ids.size()

        # Divide input_ids e attention_masks in blocchi di 512 token
        num_chunks = (seq_len + 511) // 512
        input_ids = input_ids[:, :512*num_chunks].view(batch_size * num_chunks, 512)
        attention_masks = attention_masks[:, :512*num_chunks].view(batch_size * num_chunks, 512)

        # Ottieni l'output del modello BERT
        _, last_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_masks, return_dict=False)

        cls_tokens = last_hidden_states[:, 0, :]

        cls_tokens = cls_tokens.view(batch_size, num_chunks, -1)

        if self.aggregation == 'mean':
            aggregated_output = torch.mean(cls_tokens, dim=1)
        elif self.aggregation == 'max':
            aggregated_output, _ = torch.max(cls_tokens, dim=1)
        else:
            raise ValueError("Aggregation must be 'mean' or 'max'")

        aggregated_output = self.dropout(aggregated_output)
        output = self.fc(aggregated_output)
        return output


model = BERTAggregatedClass(num_classes=NUM_CLASSES)
model.to(DEVICE)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Data Loading
trainSet, testSet, valSet = readAndPreprocessDataset()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
training_set = CustomDataset(trainSet, tokenizer, MAX_LEN * CODE_BLOCKS)
testing_set = CustomDataset(testSet, tokenizer, MAX_LEN * CODE_BLOCKS)
validation_set = CustomDataset(valSet, tokenizer, MAX_LEN * CODE_BLOCKS)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **test_params)

# Training Functions
batch_losses = []
epoch_losses = []
epoch_accuracies = []
def train(epoch):
    model.train()
    total_loss = 0.0
    batch_losses = []
    progress_bar = tqdm(enumerate(training_loader), total=len(training_loader))

    for batch_idx, data in progress_bar:
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)

        outputs = model(input_ids=ids, attention_masks=mask)
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


def validate():
    model.eval()
    fin_targets = []
    fin_outputs = []
    total_loss = 0.0

    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.float)

            outputs = model(input_ids=ids, attention_masks=mask)
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

best_val_loss = float('inf')
print("Starting training")

# Create the directory to save the state dicts and plots
import os
os.makedirs(f'./stateDict_{MODEL}', exist_ok=True)

all_batch_losses = []  # Add this list to store all batch losses

for epoch in range(EPOCHS):
    print(f"Starting Epoch: {epoch}")
    avg_train_loss, batch_losses = train(epoch)
    avg_val_loss, outputs, targets = validate()

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

def save_batch_losses_to_csv(batch_losses, filename='batch-losses.csv'):
    df = pd.DataFrame({'Batch': list(range(1, len(batch_losses) + 1)), 'Loss': batch_losses})
    df.to_csv(filename, index=False)

save_batch_losses_to_csv(all_batch_losses)

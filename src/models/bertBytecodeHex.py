import numpy as np
import pandas as pd
import re
import torch
import transformers
from transformers import PreTrainedTokenizer, BertTokenizer, BertModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from tqdm import tqdm
import logging
from sklearn import metrics

# Constants
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
SAFE_IDX = 4
MODEL = 'bertBytecode'
TOKENIZER_TYPE = 'hex'  # Options: 'hex', 'bert'
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HexTokenizer Class
class HexTokenizer(PreTrainedTokenizer):
    def __init__(self, max_length=None):
        self.hex_dict = self._getHexDict()
        self.ids_to_tokens = {i: self.hex_dict[i] for i in range(len(self.hex_dict))}
        self.tokens_to_ids = {v: k for k, v in self.ids_to_tokens.items()}
        self.max_length = max_length
        super().__init__(vocab_size=len(self.hex_dict))

    def _getHexDict(self):
        hex_dict = {i: f"{j:02x}".upper() for i, j in enumerate(range(256))}
        hex_dict[256] = '<UNK>'
        hex_dict[257] = '<PAD>'
        hex_dict[258] = '<INIT>'
        hex_dict[259] = '<END>'
        return hex_dict

    def get_vocab(self):
        return self.tokens_to_ids

    def encode(self, text):
        if self.max_length is not None and len(text) > self.max_length:
            text = text[:self.max_length]

        ids = []
        for i in range(0, len(text), 2):
            code = text[i:i+2]
            if code == '0X':
                ids.append(self.tokens_to_ids["<INIT>"])
            elif code in self.tokens_to_ids:
                ids.append(self.tokens_to_ids[code])
            else:
                ids.append(self.tokens_to_ids["<UNK>"])

        for i in range(self.max_length // 2 - len(ids)):
            ids.append(self.tokens_to_ids["<PAD>"])

        ids[-1] = self.tokens_to_ids["<END>"]
        return ids

    def decode(self, ids):
        return ''.join([self.ids_to_tokens[i] for i in ids])

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
    return {
        "bytecode": example["bytecode"],
        "label": oneHotEncodeLabel(example["slither"])
    }

def readAndPreprocessDataset():
    train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir="./cache", ignore_verifications=True).map(transformData)
    test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir="./cache", ignore_verifications=True).map(transformData)
    val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir="./cache", ignore_verifications=True).map(transformData)
    return train_set, test_set, val_set

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, tokenizer_type):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.bytecode = dataframe["bytecodeOrigin"]
        self.targets = dataframe["label"]
        self.max_len = max_len
        self.tokenizer_type = tokenizer_type

    def __len__(self):
        return len(self.bytecode)

    def __getitem__(self, index):
        bytecode = str(self.bytecode[index])
        # Remove the "0x" prefix if it exists
        if bytecode.startswith("0x"):
            bytecode = bytecode[2:]

        # Split the bytecode into tokens of 2 characters each
        bytecode_tokens = [bytecode[i:i+2] for i in range(0, len(bytecode), 2)]


        if self.tokenizer_type == 'hex':
            ids = self.tokenizer.encode(''.join(bytecode_tokens))
            padding_length = self.max_len - len(ids)
            ids += [self.tokenizer.tokens_to_ids["<PAD>"]] * padding_length
            mask = [1] * len(ids)
        else:
            bytecode_text = ' '.join(bytecode_tokens)  # Create a string with spaces between tokens
            encoding = self.tokenizer.encode_plus(
                bytecode_text,
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
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased', cache_dir="./cache")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, NUM_CLASSES)
    
    def forward(self, input_ids, attention_masks):
        _, output_1 = self.l1(input_ids=input_ids, attention_mask=attention_masks, return_dict=False)
        output_2 = self.l2(output_1)
        return self.l3(output_2)

model = BERTClass()
model.to(DEVICE)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Data Loading
trainSet, testSet, valSet = readAndPreprocessDataset()
tokenizer = HexTokenizer(max_length=MAX_LEN * 2) if TOKENIZER_TYPE == 'hex' else BertTokenizer.from_pretrained('bert-base-uncased')
training_set = CustomDataset(trainSet, tokenizer, MAX_LEN, TOKENIZER_TYPE)
testing_set = CustomDataset(testSet, tokenizer, MAX_LEN, TOKENIZER_TYPE)
validation_set = CustomDataset(valSet, tokenizer, MAX_LEN, TOKENIZER_TYPE)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **test_params)
# Training Functions
epoch_losses = []
epoch_accuracies = []

def train(epoch):
    model.train()
    total_loss = 0.0
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
        if batch_idx % 10 == 0:
            progress_bar.set_description(f'Epoch {epoch}, Loss: {loss.item()}')

    avg_loss = total_loss / len(training_loader)
    logger.info(f'Epoch {epoch}, Average Loss: {avg_loss}')
    torch.save(model.state_dict(), f'./stateDict_{TOKENIZER_TYPE}/epoch_{epoch}_{MODEL}.pth')

    return avg_loss

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

    avg_loss = total_loss / len(testing_loader)
    return avg_loss, fin_outputs, fin_targets

def save_metrics_to_csv(epoch_losses, epoch_accuracies, filename='train-metrics.csv'):
    df = pd.DataFrame({'Epoch': list(range(1, EPOCHS + 1)), 'Loss': epoch_losses, 'Accuracy': epoch_accuracies})
    df.to_csv(filename, index=False)

best_val_loss = float('inf')
print("Starting training")

for epoch in range(EPOCHS):
    print(f"Starting Epoch: {epoch}")
    avg_train_loss = train(epoch)
    avg_val_loss, outputs, targets = validate()

    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Validation Loss = {avg_val_loss}")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    epoch_losses.append(avg_train_loss)
    epoch_accuracies.append(accuracy)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'./stateDict_{TOKENIZER_TYPE}/best_model_{MODEL}.pth')

save_metrics_to_csv(epoch_losses, epoch_accuracies)

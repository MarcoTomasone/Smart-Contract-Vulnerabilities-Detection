import re
import numpy as np
import pandas as pd
import logging
from sklearn import metrics
import transformers
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, RobertaTokenizer
from torch import cuda, nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Start!")
logger.info("Starting script...")

#region constants
MAX_LEN = 512
CODE_BLOCKS = 2
SAFE_IDX = 4
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
AGG = 'mean'
MODE = AGG + CODE_BLOCKS
MODEL = 'codeBertAggMean'
TOKENIZER = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory Usage: Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB, Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
else:
    logger.info("NO GPU AVAILABLE!")
#endregion

#region functions

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
  train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
  test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
  val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
  return train_set, test_set, val_set
#endregion

#region dataset
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

#endregion

#region model
class CodeBERTAggregatedClass(torch.nn.Module):
    def __init__(self, num_classes, aggregation=AGG, dropout=0.3):
        super(CodeBERTAggregatedClass, self).__init__()
        self.codebert = AutoModel.from_pretrained('microsoft/codebert-base', cache_dir="./cache")
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.codebert.config.hidden_size, num_classes)
        self.aggregation = aggregation

    def forward(self, input_ids, attention_masks):
        batch_size, seq_len = input_ids.size()

        # Divide input_ids e attention_masks in blocchi di 512 token
        num_chunks = (seq_len + 511) // 512
        input_ids = input_ids[:, :512*num_chunks].view(batch_size * num_chunks, 512)
        attention_masks = attention_masks[:, :512*num_chunks].view(batch_size * num_chunks, 512)

        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_masks)

        last_hidden_states = outputs.last_hidden_state
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
#endregion

#region train
losses = []

def train(epoch):
    model.train()
    total_loss = 0
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
        losses.append(loss.item())

        if batch_idx % 10 == 0:
            progress_bar.set_description(f'Epoch {epoch}, Loss: {loss.item()}')

    avg_loss = total_loss / len(training_loader)
    logger.info(f'Epoch {epoch}, Average Loss: {avg_loss}')
    torch.save(model.state_dict(), f'./stateDictBC_{MODE}/epoch_{epoch}_{MODEL}.pth')
    return avg_loss

#endregion

#region validate
def validate(epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.float)

            outputs = model(input_ids=ids, attention_masks=mask)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets
#endregion

#region execution
import os

train_set, test_set, val_set = readAndPreprocessDataset()
training_set = CustomDataset(train_set, TOKENIZER)
validation_set = CustomDataset(val_set, TOKENIZER)
testing_set = CustomDataset(test_set, TOKENIZER)

training_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
validation_loader = DataLoader(validation_set, batch_size=VALID_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
testing_loader = DataLoader(testing_set, batch_size=VALID_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

model = CodeBERTAggregatedClass(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()

best_f1 = 0
best_accuracy = 0

# Create directory if it doesn't exist
os.makedirs(f'./stateDictBC_{MODE}', exist_ok=True)

for epoch in range(EPOCHS):
    avg_train_loss = train(epoch)
    fin_outputs, fin_targets = validate(epoch)
    accuracy = metrics.accuracy_score(np.array(fin_targets), np.array(fin_outputs) > 0.5)
    f1 = metrics.f1_score(np.array(fin_targets), np.array(fin_outputs) > 0.5, average='macro')
    logger.info(f"Epoch {epoch}, Accuracy: {accuracy}, F1: {f1}")

    if f1 > best_f1:
        torch.save(model.state_dict(), f'./stateDictBC_{MODE}/best_model_{MODEL}.pth')
        best_f1 = f1
        best_accuracy = accuracy
        logger.info(f"New best model saved with F1: {f1} and Accuracy: {accuracy}")

# Save losses to CSV
loss_df = pd.DataFrame(losses, columns=["loss"])
loss_df.to_csv(f'./stateDictBC_{MODE}/training_losses.csv', index=False)

# Generate loss plot
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Batch Number")
plt.ylabel("Loss")
plt.title("Training Loss over batches")
plt.legend()
plt.savefig(f'./stateDictBC_{MODE}/training_loss_plot.png')
plt.close()

#endregion

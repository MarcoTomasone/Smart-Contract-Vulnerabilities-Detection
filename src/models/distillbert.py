# Importing stock ml libraries
import re
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from torch import cuda

#region constants
# Defining some key variables that will be used later on in the training
MAX_LEN = 512
SAFE_IDX = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
#Additional Info when using cuda
if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
#endregion

#region functions 
def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group is not None, then we have captured a real comment string.
        if match.group(2) is not None:
            return ""
        else: # otherwise, we will return the 1st group
            return match.group(1)
    return regex.sub(_replacer, string)


def deleteNewlineAndGetters(string):
    string = string.replace('\n', '') #delete newlines
    # define regex to match all the getter functions with just a return statement
    regex_getter = r'function\s+(_get|get)[a-zA-Z0-9_]+\([^\)]*\)\s(external|view|override|virtual|private|returns|public|\s)*\(([^)]*)\)\s*\{(\r\n|\r|\n|\s)*return\s*[^\}]*\}'
    # delete all the getter functions matched 
    string = re.sub(regex_getter, '', string)
    return string


def oneHotEncodeLabel(label):
    one_hot = np.zeros(NUM_CLASSES)
    for elem in label:
        if elem < SAFE_IDX:
            one_hot[elem] = 1
        elif elem > SAFE_IDX:
            one_hot[elem-1] = 1
    return one_hot


def transformData(example):
   data= {}
   data["source_code"] = deleteNewlineAndGetters(remove_comments(example["source_code"])) 
   data["bytecodeOrigin"] = example["bytecode"]
   #data["bytecode"] = bytecode_tokenizer.encode(example["bytecode"])
   data["label"] = oneHotEncodeLabel(example["slither"])
   return data

def readAndPreprocessDataset():
  train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir="./cache", ignore_verifications=True).map(transformData)
  test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir="./cache",  ignore_verifications=True).map(transformData)
  val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir="./cache",  ignore_verifications=True).map(transformData)
  return train_set, test_set, val_set
#endregion
#region dataset
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.sourceCode = dataframe["source_code"]
        self.targets =  dataframe["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_code = str(self.sourceCode[index])
        source_code = " ".join(source_code.split())

        encoding = self.tokenizer.encode_plus(
            source_code,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        target = torch.tensor(self.targets[index], dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': target
        }
#endregion
    
#region model 

class DistilBERTClass(torch.nn.Module):
    def __init__(self, NUM_CLASSES):
        super(DistilBERTClass, self).__init__()
        self.num_classes = NUM_CLASSES
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768, NUM_CLASSES)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        output = self.fc(pooled_output)
        return output

model = DistilBERTClass(NUM_CLASSES)
model.to(DEVICE)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

#endregion
trainSet, testSet, valSet = readAndPreprocessDataset()

training_set = CustomDataset(trainSet, TOKENIZER, MAX_LEN)
testing_set = CustomDataset(testSet, TOKENIZER, MAX_LEN)
validation_set = CustomDataset(valSet, TOKENIZER, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

#region training
def train(epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(training_loader, 0):
        input_ids = data['ids'].to(DEVICE)
        attention_mask = data['mask'].to(DEVICE)
        targets = data['targets'].to(DEVICE)

        outputs = model(input_ids, attention_mask)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

        if batch_idx % 500 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(training_loader)
    print(f'Epoch: {epoch}, Average Loss: {avg_loss}')

    torch.save(model.state_dict(), f'./stateDict/epoch_{epoch}_model.pth')

    return avg_loss

def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            input_ids = data['ids'].to(DEVICE)
            attention_mask = data['mask'].to(DEVICE)
            targets = data['targets'].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

print("Starting training")

for epoch in range(EPOCHS):
    print(f"Starting Epoch: {epoch}")
    train(epoch) 
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
#endregion

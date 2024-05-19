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

logging.basicConfig(level=logging.INFO)

#region constants
MAX_LEN = 512
CODE_BLOCKS = 4
SAFE_IDX = 4
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
MODEL = 'codeBertAgg'
TOKENIZER = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
#endregion

#region functions 
def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)
    return regex.sub(_replacer, string)

def deleteNewlineAndGetters(string):
    string = string.replace('\n', '')
    regex_getter = r'function\s+(_get|get)[a-zA-Z0-9_]+\([^\)]*\)\s(external|view|override|virtual|private|returns|public|\s)*\(([^)]*)\)\s*\{(\r\n|\r|\n|\s)*return\s*[^\}]*\}'
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
   data["label"] = oneHotEncodeLabel(example["slither"])
   return data

def readAndPreprocessDataset():
  train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir="./cache", ignore_verifications=True).map(transformData)
  test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir="./cache", ignore_verifications=True).map(transformData)
  val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir="./cache", ignore_verifications=True).map(transformData)
  return train_set, test_set, val_set
#endregion

#region dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN * CODE_BLOCKS):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.sourceCode = dataframe["source_code"]
        self.targets = dataframe["label"]

    def __len__(self):
        return len(self.sourceCode)

    def __getitem__(self, index):
        sourceCode = str(self.sourceCode[index])
        sourceCode = " ".join(sourceCode.split())

        if len(sourceCode) == 0:
            raise ValueError(f"Source code at index {index} is empty.")

        inputs = self.tokenizer.encode_plus(
            sourceCode,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

#endregion

#region model
class CodeBERTAggregatedClass(torch.nn.Module):
    def __init__(self, num_classes, aggregation='mean', dropout=0.3):
        super(CodeBERTAggregatedClass, self).__init__()
        self.codebert = AutoModel.from_pretrained('microsoft/codebert-base', cache_dir="./cache")
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.codebert.config.hidden_size, num_classes)
        self.aggregation = aggregation

    def forward(self, input_ids, attention_masks, token_type_ids):
        batch_size, seq_len = input_ids.size()

        # Divide input_ids e attention_masks in blocchi di 512 token
        num_chunks = (seq_len + 511) // 512
        input_ids = input_ids[:, :512*num_chunks].view(batch_size * num_chunks, 512)
        attention_masks = attention_masks[:, :512*num_chunks].view(batch_size * num_chunks, 512)
        token_type_ids = token_type_ids[:, :512*num_chunks].view(batch_size * num_chunks, 512)

        with torch.no_grad():
            outputs = self.codebert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)

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
def train(epoch):
    model.train()
    for batch_idx, data in enumerate(training_loader, 0):
        print(f"Epoch: {epoch}")
        print(f"Batch: {batch_idx}")
        #print size of data 
        print(data['ids'].size())
        print(data['mask'].size())
        print(data['token_type_ids'].size())

        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)

        outputs = model(input_ids=ids, attention_masks=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
#endregion

#region validate
def validate(epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.float)

            outputs = model(input_ids=ids, attention_masks=mask, token_type_ids=token_type_ids)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
#endregion

#region execution
train_set, test_set, val_set = readAndPreprocessDataset()
training_set = CustomDataset(train_set, TOKENIZER)
validation_set = CustomDataset(val_set, TOKENIZER)
testing_set = CustomDataset(test_set, TOKENIZER)

training_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=VALID_BATCH_SIZE, shuffle=False)
testing_loader = DataLoader(testing_set, batch_size=VALID_BATCH_SIZE, shuffle=False)

model = CodeBERTAggregatedClass(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    print(f"Starting Epoch: {epoch}")
    train(epoch) 
    outputs, targets = validate(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
#endregion

# Importing stock ml libraries
import re
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from pyevmasm import disassemble_hex
#region constants
# Defining some key variables that will be used later on in the training
MAX_LEN = 512
SAFE_IDX = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MODEL = 'bert-bytecode'
#Additional Info when using cuda
if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
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

def normalize_bytecode(bytecode):
    opcode_list = disassemble_hex(bytecode).split('\n')
    new_opcodes = []

    for opcode in opcode_list:
        if 'PUSH' in opcode:
            opcode = opcode.split(' ')[0]  # Prendi solo il comando, escludendo i parametri
        new_opcodes.append(opcode)

    return new_opcodes

def transformData(example):
   data= {}
   #data["source_code"] = deleteNewlineAndGetters(remove_comments(example["source_code"])) 
   data["bytecode"] = normalize_bytecode(example["bytecode"])
   #data["bytecode"] = bytecode_tokenizer.encode(example["bytecode"])
   data["label"] = oneHotEncodeLabel(example["slither"])
   return data

def readAndPreprocessDataset():
  train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData, remove_columns=['address',  'source_code', 'slither'])
  test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir="./cache",  ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData, remove_columns=['address',  'source_code', 'slither'])
  val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir="./cache",  ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData, remove_columns=['address',  'source_code', 'slither'])
  return train_set, test_set, val_set
#endregion
#region dataset
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.bytecode = dataframe["bytecode"]
        self.targets =  dataframe["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.sourceCode)

    def __getitem__(self, index):
        bytecode = str(self.bytecode[index])
        bytecode = " ".join(bytecode.split())

        inputs = self.tokenizer.encode_plus(
            bytecode,
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
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', cache_dir="./cache")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, NUM_CLASSES)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass()
model.to(DEVICE)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
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
    model.train() #Set the model to training mode, for dropout and batchnorm
    total_loss = 0.0
    for batch_idx, data in enumerate(training_loader, 0):
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()  # Aggiunta della perdita al totale

        if batch_idx % 500 == 0:  # Modificato il log ogni 500 iterazioni
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        loss.backward()
        optimizer.step()

    # Calcolo della perdita media per epoca
    avg_loss = total_loss / len(training_loader)
    print(f'Epoch: {epoch}, Average Loss: {avg_loss}')

    # Salvataggio dei pesi del modello alla fine di ogni epoca
    torch.save(model.state_dict(), f'epoch_{epoch}_{MODEL}.pth')

    return avg_loss  # Restituzione della perdita media per monitorare l'andamento

def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(DEVICE, dtype = torch.long)
            mask = data['mask'].to(DEVICE, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype = torch.long)
            targets = data['targets'].to(DEVICE, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
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

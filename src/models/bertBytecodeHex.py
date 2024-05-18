
import numpy as np
import pandas as pd
from sklearn import metrics
import re
import transformers
from transformers import PreTrainedTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda

#region constants
# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
SAFE_IDX = 4
MODEL = 'bertBytecode'
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
#endregion
#region Tokenizer
class HexTokenizer(PreTrainedTokenizer):
    def __init__(self,  max_length=None):
        self.hex_dict = self._getHexDict()
        self.ids_to_tokens = {i: self.hex_dict[i] for i in range(len(self.hex_dict))}
        self.tokens_to_ids = {v: k for k, v in self.ids_to_tokens.items()}
        self.max_length = max_length
        tokenizer_kwargs = {"vocab_size": len(self.hex_dict)}
        super().__init__(**tokenizer_kwargs)

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

        # truncate the input with the maximum length
        if self.max_length is not None and len(text) > self.max_length:
            text = text[:self.max_length]


        # convert pair of nibbles in a token
        ids = []
        for i in range(0, len(text), 2):
            code = text[i:i+2]
            if code == '0X':
                ids.append(self.tokens_to_ids["<INIT>"])
            elif code in self.tokens_to_ids:
                ids.append(self.tokens_to_ids[code])
            else:
                # if the byte isn't in the vocabulary
                ids.append(self.tokens_to_ids["<UNK>"])

        # add padding
        for i in range(self.max_length//2 - len(ids)):
            ids.append(self.tokens_to_ids["<PAD>"])

        # replace the last token with the end token
        ids[-1] = self.tokens_to_ids["<END>"]
        return ids

    def decode(self, ids):
        text = ''.join([self.ids_to_tokens[i] for i in ids])
        return text
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
   data["source_code"] = remove_comments(example["source_code"])
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
        return len(self.sourceCode)

    def __getitem__(self, index):
        sourceCode = str(self.sourceCode[index])
        sourceCode = " ".join(sourceCode.split())

        # Utilizzo il metodo `encode` di HexTokenizer al posto di `encode_plus` di Hugging Face Tokenizer
        ids = self.tokenizer.encode(sourceCode)

        # Padding fino alla lunghezza massima
        padding_length = self.max_len - len(ids)
        ids += [self.tokenizer.hex_dict["<PAD>"]] * padding_length

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
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
    
    def forward(self, ids):
        _, output_1= self.l1(ids,  return_dict=False)
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
tokenizer = HexTokenizer(max_length=MAX_LEN * 2)
training_set = CustomDataset(trainSet, tokenizer, MAX_LEN)
testing_set = CustomDataset(testSet, tokenizer, MAX_LEN)
validation_set = CustomDataset(valSet, tokenizer, MAX_LEN)

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
# Lists to store epoch loss and accuracy
epoch_losses = []
epoch_accuracies = []

#region training
def train(epoch):
    model.train() #Set the model to training mode, for dropout and batchnorm
    total_loss = 0.0
    for batch_idx, data in enumerate(training_loader, 0):
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)

        outputs = model(ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()  # Aggiunta della perdita al totale

        if batch_idx % 500 == 0:  
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(training_loader)
    print(f'Epoch: {epoch}, Average Loss: {avg_loss}')

    torch.save(model.state_dict(), f'./stateDict/epoch_{epoch}_{MODEL}.pth')

    epoch_losses.append(avg_loss)

    return avg_loss

def validation(epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(DEVICE, dtype = torch.long)
            targets = data['targets'].to(DEVICE, dtype = torch.float)
            outputs = model(ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

def save_metrics_to_csv(epoch_losses, epoch_accuracies, filename='train-metrics.csv'):
    df = pd.DataFrame({
        'Epoch': list(range(1, EPOCHS + 1)),
        'Loss': epoch_losses,
        'Accuracy': epoch_accuracies
    })
    df.to_csv(filename, index=False)

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
    epoch_accuracies.append(accuracy)

# Save metrics to CSV
save_metrics_to_csv(epoch_losses, epoch_accuracies)
#endregion
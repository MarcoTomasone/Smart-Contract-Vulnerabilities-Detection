# Importing stock ml libraries
import re
import numpy as np
import pandas as pd
import transformers
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AutoModel, AutoTokenizer
from torch import cuda
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

  
#region constants
# Defining some key variables that will be used later on in the training
MAX_LEN = 512
SAFE_IDX = 4
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
MODEL_NAME = 'CodeBERT'
TOKENIZER = AutoTokenizer.from_pretrained('microsoft/codebert-base')
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

def readAndPreprocessTrainSet():
  train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir="./cache", ignore_verifications=True).map(transformData)
  return train_set

def readAndPreprocessTestSet():
  test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir="./cache",  ignore_verifications=True).map(transformData)
  return test_set

def readAndPreprocessValSet():
  val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir="./cache",  ignore_verifications=True).map(transformData)
  return  val_set

def readAndPreprocessDataset():
    train_set = readAndPreprocessTrainSet()
    test_set = readAndPreprocessTestSet()
    val_set = readAndPreprocessValSet()
    return train_set, test_set, val_set


def save_confusion_matrix_plot(y_true, y_pred, classes, file_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix {MODEL_NAME}')
    plt.savefig(file_path, format='png')  # Salva l'immagine come file PNG
    plt.close()  # Chiudi la figura per liberare la memoria


def evaluate_model(predictions, ground_truth):
    print("Ground Truth:\n", ground_truth)
    print("\nPredictions:\n", predictions)
    
    threshold = 0.5
    y_pred = []
    
    for sample in predictions:
        y_pred.append([1 if i >= threshold else 0 for i in sample])
        
    y_pred = np.array(y_pred)

    label_names = ['access-control', 'arithmetic', 'other', 'reentrancy', 'safe', 'unchecked-calls']

    print("Classification Report:")
    print(classification_report(ground_truth, y_pred, target_names=label_names))
    
    print("Precision of each label:")
    print("None:", precision_score(ground_truth, y_pred, average=None))
    print("micro:", precision_score(ground_truth, y_pred, average='micro'))
    print("macro:", precision_score(ground_truth, y_pred, average='macro'))
    print("weighted:", precision_score(ground_truth, y_pred, average='weighted'))
    print("samples:", precision_score(ground_truth, y_pred, average='samples'))
    
    print("Recall of each label:")
    print("None:", recall_score(ground_truth, y_pred, average=None))
    print("micro:", recall_score(ground_truth, y_pred, average='micro'))
    print("macro:", recall_score(ground_truth, y_pred, average='macro'))
    print("weighted:", recall_score(ground_truth, y_pred, average='weighted'))
    print("samples:", recall_score(ground_truth, y_pred, average='samples'))
    
    print("F1-score of each label:")
    print("None:", f1_score(ground_truth, y_pred, average=None))
    print("micro:", f1_score(ground_truth, y_pred, average='micro'))
    print("macro:", f1_score(ground_truth, y_pred, average='macro'))
    print("weighted:", f1_score(ground_truth, y_pred, average='weighted'))
    print("samples:", f1_score(ground_truth, y_pred, average='samples'))

    save_confusion_matrix_plot(ground_truth, y_pred, label_names, f'./confusion_matrix/{MODEL_NAME}_confusion_matrix.png')

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
class CodeBERTClass(torch.nn.Module):
    def __init__(self, NUM_CLASSES):
        super(CodeBERTClass, self).__init__()
        self.num_classes = NUM_CLASSES
        self.codebert = AutoModel.from_pretrained('microsoft/codebert-base', cache_dir="./cache")
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768, NUM_CLASSES)
    
    def forward(self, ids, mask, token_type_ids):
        outputs = self.codebert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        output = self.fc(pooled_output)
        return output

model = CodeBERTClass(NUM_CLASSES)
model.to(DEVICE)
model.load_state_dict(torch.load('./stateDict/epoch_19_model.pth'))


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
#endregion
#trainSet, testSet, valSet = readAndPreprocessDataset()

#training_set = CustomDataset(trainSet, TOKENIZER, MAX_LEN)

testSet = readAndPreprocessTestSet()
valSet = readAndPreprocessValSet()
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

#training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **test_params)
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
    torch.save(model.state_dict(), f'./stateDict/epoch_{epoch}_{MODEL_NAME}.pth')

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
"""
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
"""
#validate test set and call evaluate model 
print("Validating test set")
outputs, targets = validation(0)
evaluate_model(outputs, targets)

#validate validation set and call evaluate model
print("Validating validation set")
outputs, targets = validation(0)
evaluate_model(outputs, targets)

#endregion

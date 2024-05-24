import os
import re
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import transformers
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, RobertaTokenizer
from torch import cuda, nn
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Start!")
logger.info("Starting script...")

#region constants
MAX_LEN = 512
CODE_BLOCKS = 2
SAFE_IDX = 4
VALID_BATCH_SIZE = 16
NUM_CLASSES = 5
MODEL = 'codeBertAggMean'
MODE = "Mean2"
TOKENIZER = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory Usage: Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB, Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
else:
    logger.info("NO GPU AVAILABLE!")
#endregion

#region class mapping
class_mapping = {
    0: 'access-control',
    1: 'arithmetic',
    2: 'other',
    3: 'reentrancy',
    4: 'unchecked-calls'
}
class_names = [class_mapping[i] for i in range(NUM_CLASSES)]
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
    data = {}
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
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
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

#region evaluation
def plot_and_save_confusion_matrix(confusion_matrix, class_names, dataset_type, output_dir):
    for i, matrix in enumerate(confusion_matrix):
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{dataset_type} - Confusion Matrix for Class {class_names[i]}')
        plt.savefig(os.path.join(output_dir, f'{dataset_type}_confusion_matrix_class_{class_names[i]}.png'))
        plt.close()

def evaluate_and_save(loader, dataset_type, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader), total=len(loader)):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.float)

            outputs = model(input_ids=ids, attention_masks=mask)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    predictions = np.array(fin_outputs) > 0.5
    targets = np.array(fin_targets)

    accuracy = metrics.accuracy_score(targets, predictions)
    precision_micro = metrics.precision_score(targets, predictions, average='micro')
    precision_macro = metrics.precision_score(targets, predictions, average='macro')
    precision_weighted = metrics.precision_score(targets, predictions, average='weighted')
    recall_micro = metrics.recall_score(targets, predictions, average='micro')
    recall_macro = metrics.recall_score(targets, predictions, average='macro')
    recall_weighted = metrics.recall_score(targets, predictions, average='weighted')
    f1_micro = metrics.f1_score(targets, predictions, average='micro')
    f1_macro = metrics.f1_score(targets, predictions, average='macro')
    f1_weighted = metrics.f1_score(targets, predictions, average='weighted')
    confusion = metrics.multilabel_confusion_matrix(targets, predictions)
    classification_report = metrics.classification_report(targets, predictions, target_names=class_names, output_dict=True)

    logger.info(f'{dataset_type} Accuracy: {accuracy}')
    logger.info(f'{dataset_type} Precision (Micro): {precision_micro}')
    logger.info(f'{dataset_type} Precision (Macro): {precision_macro}')
    logger.info(f'{dataset_type} Precision (Weighted): {precision_weighted}')
    logger.info(f'{dataset_type} Recall (Micro): {recall_micro}')
    logger.info(f'{dataset_type} Recall (Macro): {recall_macro}')
    logger.info(f'{dataset_type} Recall (Weighted): {recall_weighted}')
    logger.info(f'{dataset_type} F1 Score (Micro): {f1_micro}')
    logger.info(f'{dataset_type} F1 Score (Macro): {f1_macro}')
    logger.info(f'{dataset_type} F1 Score (Weighted): {f1_weighted}')

    with open(os.path.join(output_dir, f'{dataset_type}_metrics.txt'), 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision (Micro): {precision_micro}\n')
        f.write(f'Precision (Macro): {precision_macro}\n')
        f.write(f'Precision (Weighted): {precision_weighted}\n')
        f.write(f'Recall (Micro): {recall_micro}\n')
        f.write(f'Recall (Macro): {recall_macro}\n')
        f.write(f'Recall (Weighted): {recall_weighted}\n')
        f.write(f'F1 Score (Micro): {f1_micro}\n')
        f.write(f'F1 Score (Macro): {f1_macro}\n')
        f.write(f'F1 Score (Weighted): {f1_weighted}\n')
        f.write('\nClassification Report:\n')
        f.write(metrics.classification_report(targets, predictions, target_names=class_names))

    predictions_df = pd.DataFrame(fin_outputs, columns=class_names)
    predictions_df.to_csv(os.path.join(output_dir, f'{dataset_type}_predictions.csv'), index=False)


    metrics_df = pd.DataFrame(classification_report).transpose()
    metrics_df.to_csv(os.path.join(output_dir, f'{dataset_type}_classification_report.csv'))

    plot_and_save_confusion_matrix(confusion, class_names, dataset_type, output_dir)

    return accuracy, precision_micro, precision_macro, precision_weighted, recall_micro, recall_macro, recall_weighted, f1_micro, f1_macro, f1_weighted

#endregion

#region execution
train_set, test_set, val_set = readAndPreprocessDataset()

training_set = CustomDataset(train_set, TOKENIZER)
validation_set = CustomDataset(val_set, TOKENIZER)
testing_set = CustomDataset(test_set, TOKENIZER)

# Definisci i DataLoader
training_loader = DataLoader(training_set, batch_size=VALID_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
validation_loader = DataLoader(validation_set, batch_size=VALID_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
testing_loader = DataLoader(testing_set, batch_size=VALID_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

# Carica il modello pre-addestrato
model = CodeBERTAggregatedClass(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(f'./stateDictMean/best_model_{MODEL}.pth'))

# Esegui l'evaluation su train, validation, e test set
logger.info("Evaluating on training set...")
evaluate_and_save(training_loader, 'train', './results_{MODE}/train')

logger.info("Evaluating on validation set...")
evaluate_and_save(validation_loader, 'validation', './results_{MODE}/validation')

logger.info("Evaluating on testing set...")
evaluate_and_save(testing_loader, 'test', './results_{MODE}/test')

logger.info("Evaluation completed.")
#endregion

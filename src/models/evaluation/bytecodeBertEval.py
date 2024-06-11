import numpy as np
import pandas as pd
import re
import torch
import transformers
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from tqdm import tqdm
import logging
from sklearn import metrics
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Constants
MAX_LEN = 512
VALID_BATCH_SIZE = 16
NUM_CLASSES = 5
SAFE_IDX = 4
MODEL = 'bertBytecode'
MODE = "bytecode"
TOKENIZER_TYPE = 'bert'  # Use standard BERT tokenizer
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')


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
    return {
        "bytecode": example["bytecode"],
        "label": oneHotEncodeLabel(example["slither"])
    }

def readAndPreprocessDataset():
    train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
    test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
    val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
    return train_set, test_set, val_set

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.bytecode = dataframe["bytecode"]
        self.targets = dataframe["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.bytecode)

    def __getitem__(self, index):
        bytecode = str(self.bytecode[index])
        # Remove the "0x" prefix if it exists
        if bytecode.startswith("0x"):
            bytecode = bytecode[2:]

        # Split the bytecode into tokens of 2 characters each
        bytecode_tokens = [bytecode[i:i+2] for i in range(0, len(bytecode), 2)]
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


def plot_and_save_multilabel_confusion_matrices(y_true, y_pred, label_names, dataset_type, output_dir):
    """
    Plots and saves confusion matrices for each label in a multilabel setting.

    Parameters:
    - y_true: numpy array of true labels, shape (n_samples, n_labels)
    - y_pred: numpy array of predicted labels, shape (n_samples, n_labels)
    - label_names: list of strings, names of the labels
    - dataset_type: string, type of dataset (e.g., 'train', 'validation', 'test')
    - output_dir: string, directory to save the plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_col in range(len(label_names)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]

        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true_label, y_pred_label, ax=ax, cmap=plt.cm.Blues, normalize='true', colorbar=True
        )
        ax.set_title(f"Confusion Matrix for {label_names[label_col]} ({dataset_type})")

        output_path = os.path.join(output_dir, f'{dataset_type}_confusion_matrix_label_{label_names[label_col]}.png')
        plt.savefig(output_path)
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

    plot_and_save_multilabel_confusion_matrices(targets, predictions, class_names, dataset_type, output_dir)

    return accuracy, precision_micro, precision_macro, precision_weighted, recall_micro, recall_macro, recall_weighted, f1_micro, f1_macro, f1_weighted

trainSet, testSet, valSet = readAndPreprocessDataset()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

training_set = CustomDataset(trainSet, tokenizer, MAX_LEN)
testing_set = CustomDataset(testSet, tokenizer, MAX_LEN)
validation_set = CustomDataset(valSet, tokenizer, MAX_LEN)

train_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}


model = BERTClass()
model.to(DEVICE)
# Load the pretrained model
model.load_state_dict(torch.load(f'./stateDict_{TOKENIZER_TYPE}/best_model_{MODEL}.pth'))

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **test_params)


# Esegui l'evaluation su train, validation, e test set
logger.info("Evaluating on training set...")
evaluate_and_save(training_loader, f'train', f'./results_{MODE}/train')

logger.info("Evaluating on validation set...")
evaluate_and_save(validation_loader, 'validation', f'./results_{MODE}/validation')

logger.info("Evaluating on testing set...")
evaluate_and_save(testing_loader, 'test', f'./results_{MODE}/test')

logger.info("Evaluation completed.")
#endregion

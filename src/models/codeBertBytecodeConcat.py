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
MODE = "concat" + str(CODE_BLOCKS)
MODEL = 'codeBertConcat'
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
class CodeBERTConcatenatedClass(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(CodeBERTConcatenatedClass, self).__init__()
        self.codebert = AutoModel.from_pretrained('microsoft/codebert-base', cache_dir="./cache")
        self.dropout = torch.nn.Dropout(dropout)
        # Multiply hidden_size by the number of chunks you're concatenating
        self.fc = torch.nn.Linear(self.codebert.config.hidden_size * CODE_BLOCKS, num_classes)

    def forward(self, input_ids, attention_masks):
        batch_size, seq_len = input_ids.size()

        # Divide input_ids and attention_masks into chunks of 512 tokens
        num_chunks = (seq_len + 511) // 512
        input_ids = input_ids[:, :512*num_chunks].reshape(batch_size * num_chunks, 512)
        attention_masks = attention_masks[:, :512*num_chunks].reshape(batch_size * num_chunks, 512)

        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_masks)

        last_hidden_states = outputs.last_hidden_state
        cls_tokens = last_hidden_states[:, 0, :]

        # Concatenate the CLS tokens of the various chunks
        cls_tokens = cls_tokens.reshape(batch_size, num_chunks, -1)
        concatenated_output = cls_tokens.reshape(batch_size, -1)

        concatenated_output = self.dropout(concatenated_output)
        output = self.fc(concatenated_output)
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

model = CodeBERTConcatenatedClass(num_classes=NUM_CLASSES).to(DEVICE)
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

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Calcolo della media mobile con una finestra di dimensione 10
window_size = 10
smoothed_losses = moving_average(losses, window_size)

plt.figure(figsize=(10, 5))
plt.plot(range(len(smoothed_losses)), smoothed_losses, label="Training Loss (Smoothed)")
plt.xlabel("Batch Number")
plt.ylabel("Loss")
plt.title("Training Loss over Batches")
plt.legend()
plt.grid(True)
plt.savefig(f'./stateDictBC_{MODE}/training_loss_plot.png')
plt.close()

#region evaluation
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

#endregion
training_loader = DataLoader(training_set, batch_size=VALID_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

# Carica il modello pre-addestrato
model.load_state_dict(torch.load(f'./stateDictBC_{MODE}/best_model_{MODEL}.pth'))

# Esegui l'evaluation su train, validation, e test set
logger.info("Evaluating on training set...")
evaluate_and_save(training_loader, f'train', f'./results_{MODE}/train')

logger.info("Evaluating on validation set...")
evaluate_and_save(validation_loader, 'validation', f'./results_{MODE}/validation')

logger.info("Evaluating on testing set...")
evaluate_and_save(testing_loader, 'test', f'./results_{MODE}/test')

logger.info("Evaluation completed.")
#endregion

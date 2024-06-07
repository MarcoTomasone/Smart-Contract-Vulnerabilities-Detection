import os
import random
import functools
import re
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    RobertaTokenizer,
    BitsAndBytesConfig
)
import matplotlib.pyplot as plt

# Constants
MAX_LEN = 512 * 3
NUM_CLASSES = 5
MODEL_NAME = 'mistralai/Mistral-7B-v0.1'
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

# Logging GPU info if available
if DEVICE == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage: Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB, Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
else:
    print("NO GPU AVAILABLE!")

# Functions
def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
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
        if elem < 4:
            one_hot[elem] = 1
        elif elem > 4:
            one_hot[elem-1] = 1
    return one_hot

def transformData(example):
    data = {}
    data["source_code"] = deleteNewlineAndGetters(remove_comments(example["source_code"]))
    data["bytecodeOrigin"] = example["bytecode"]
    data["label"] = oneHotEncodeLabel(example["slither"])
    return data

def readAndPreprocessDataset():
    train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multileabel', split='train', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
    test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multileabel', split='test', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
    val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multileabel', split='validation', cache_dir="./cache", ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
    return train_set, test_set, val_set

# Load and preprocess dataset
train_set, test_set, val_set = readAndPreprocessDataset()
ds = DatasetDict({'train': train_set, 'val': val_set, 'test': test_set})

# Tokenize dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['source_code'], truncation=True, padding=True, max_length=MAX_LEN)
    tokenized_inputs['labels'] = examples['label']
    return tokenized_inputs

tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
tokenized_ds.set_format('torch')

# Quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='SEQ_CLS'
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    num_labels=NUM_CLASSES
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id

# Custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d

# Metrics for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.sigmoid(torch.tensor(predictions))  # Apply sigmoid to logits
    f1_micro = f1_score(labels, predictions > 0.5, average='micro')
    f1_macro = f1_score(labels, predictions > 0.5, average='macro')
    f1_weighted = f1_score(labels, predictions > 0.5, average='weighted')
    accuracy = accuracy_score(labels, predictions > 0.5)
    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# Custom Trainer class
class CustomTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss

# Training args
training_args = TrainingArguments(
    output_dir='multilabel_classification',
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['val'],
    tokenizer=tokenizer,
    data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    label_weights=torch.tensor([1.0] * NUM_CLASSES, device=model.device)  # Modify as needed
)

# Training
train_result = trainer.train()

# Save model and tokenizer
peft_model_id = 'multilabel_mistral'
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

# Plot training loss
def plot_loss(training_stats):
    epochs = range(1, len(training_stats['loss']) + 1)
    plt.plot(epochs, training_stats['loss'], 'b-o', label='Training loss')
    plt.title('Training loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

plot_loss(trainer.state.log_history)

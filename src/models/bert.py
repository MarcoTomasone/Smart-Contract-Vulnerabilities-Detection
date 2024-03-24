import re
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import random
import numpy as np
from transformers import PreTrainedTokenizer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from tqdm import tqdm
import matplotlib.ticker as mtick
from datasets import Dataset
from sklearn.metrics import classification_report
import os
from sklearn.metrics import precision_recall_fscore_support
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# create the dict from 00 to FF (even if not all the bytecodes will be used)
hex_dict = {i: f"{j:02x}".upper() for i, j in enumerate(range(256))}
hex_dict[256] = '<UNK>'
hex_dict[257] = '<PAD>'
hex_dict[258] = '<INIT>'
hex_dict[259] = '<END>'


class HexTokenizer(PreTrainedTokenizer):
    def __init__(self, hex_dict, max_length=None):
        self.hex_dict = hex_dict
        self.ids_to_tokens = {i: hex_dict[i] for i in range(len(hex_dict))}
        self.tokens_to_ids = {v: k for k, v in self.ids_to_tokens.items()}
        self.max_length = max_length
        tokenizer_kwargs = {"vocab_size": len(hex_dict)}
        super().__init__(**tokenizer_kwargs)

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

tokenizer = HexTokenizer(hex_dict, 800)


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

def oneHotEncodeLabel(example):
   hottedLabels = np.zeros(6, dtype=int)
   hottedLabels[example] = 1
   return hottedLabels

def transformData(example):
   data= {}
   #data["source_code"] = remove_comments(example["source_code"])
   data["bytecode"] = tokenizer.encode(example["bytecode"])
   data["label"] = oneHotEncodeLabel(example["slither"])
   return data

def readAndPreprocessDataset():
  train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
  test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
  val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', ignore_verifications=True).filter(lambda elem: elem['bytecode'] != '0x').map(transformData)
  return train_set, test_set, val_set


def evaluate_model(y_stats, val_labels):
  print("Actual \n", val_labels)
  print("\nPredicted \n",y_stats)
  threshold = 0.5
  y_pred=[]
  for sample in  y_stats:
    y_pred.append([1 if i>=0.5 else 0 for i in sample ] )
  y_pred = np.array(y_pred)

  from sklearn.metrics import accuracy_score
  accuracy_score(val_labels, y_pred)
  from sklearn.metrics import multilabel_confusion_matrix
  multilabel_confusion_matrix(val_labels, y_pred)
  from sklearn.metrics import classification_report

  label_names = ['label A', 'label B', 'label C', 'safe', 'label5', 'label6']

  print(classification_report(val_labels, y_pred,target_names=label_names))
  from sklearn import metrics
  print("None ", metrics.precision_score(val_labels, y_pred, average=None))
  print("micro: {:.2f}".format(metrics.precision_score(val_labels, y_pred, average='micro')))
  print("macro: {:.2f} ".format( metrics.precision_score(val_labels, y_pred, average='macro')))
  print("weighted: {:.2f} ".format( metrics.precision_score(val_labels, y_pred, average='weighted')))
  print("samples: {:.2f} ".format( metrics.precision_score(val_labels, y_pred, average='samples')))
  print("Recall of each label: {}".format(metrics.recall_score(val_labels, y_pred, average=None)))
  print("micro: {:.2f}".format(metrics.recall_score(val_labels, y_pred, average='micro')))
  print("macro: {:.2f} ".format( metrics.recall_score(val_labels, y_pred, average='macro')))
  print("weighted: {:.2f} ".format( metrics.recall_score(val_labels, y_pred, average='weighted')))
  print("samples: {:.2f} ".format( metrics.recall_score(val_labels, y_pred, average='samples')))
  print("F1 of each label: {}".format(metrics.f1_score(val_labels, y_pred, average=None)))
  print("micro: {:.2f}".format(metrics.f1_score(val_labels, y_pred, average='micro')))
  print("macro: {:.2f} ".format( metrics.f1_score(val_labels, y_pred, average='macro')))
  print("weighted: {:.2f} ".format( metrics.f1_score(val_labels, y_pred, average='weighted')))
  print("samples: {:.2f} ".format( metrics.f1_score(val_labels, y_pred, average='samples')))

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

CLASSES = 6

dropout_rates = [0.2, 0.5]
epochs = [10, 20]
max_seq_lengths = [ 250, 300, 350, 400, 450]
hyperparameter_combinations = list(itertools.product(max_seq_lengths, dropout_rates, epochs))


def main():
    results = pd.DataFrame(columns=['Dataset', 'num_rows', 'Macro F1-score test', 'Macro F1-score train', 'Accuracy test', 'Accuracy train', 'Loss test', 'Loss train'])
    itertools_results = pd.DataFrame(columns=['max_seq_length', 'conv_filter', 'dropout_rate', 'num_epochs', 'Macro F1-score test', 'Macro F1-score train', 'Accuracy test', 'Accuracy train', 'Loss test', 'Loss train'])
    trainSet, testSet, valSet = readAndPreprocessDataset()
    #lyrics, moods = prepare_dataset(datasets[0])
    colonnes_da_rimuovere = ['address', 'slither']
    trainSet = trainSet.remove_columns(colonnes_da_rimuovere)
    valSet = valSet.remove_columns(colonnes_da_rimuovere)
    # Seleziona un sottoinsieme di 200 campioni
    #trainSet = trainSet.select(range(1,100))
    #valSet = valSet.select(range(1,100))


    for params in hyperparameter_combinations:
        max_seq_length, dropout_rate, num_epochs = params
        print(f"Training with params: max_seq_length={max_seq_length}, dropout_rate={dropout_rate}, num_epochs={num_epochs}")

        max_seq_length = 400
        # Inizializza il tokenizer BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Tokenizzazione e padding dei testi
        input_ids = []
        attention_masks = []

        # Creazione del modello BERT
        input_layer = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
        bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        pooler_output = bert_model(input_layer)[1]  # Estrai il secondo output, che è il pooler_output

        # Aggiungi un layer di Dropout
        dropout_layer = Dropout(dropout_rate)(pooler_output)
        output_layer = Dense(6, activation='sigmoid')(dropout_layer)

        train_input = np.array(trainSet["bytecode"])
        train_labels = np.array(trainSet["label"])

        #train_input = np.squeeze(train_input, axis=1)
        #train_mask = np.squeeze(train_mask, axis=1)

        print( tf.shape(train_input))

        model = Model(inputs=input_layer, outputs=output_layer)
        project_path = "/../../data/bytecode/"
        checkpoint_path = "training_bytecode"  + str(max_seq_length) + "/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(project_path):
           os.makedirs(project_path)
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        # Compilazione e addestramento del modello
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print(type(train_input))
        #if  os.path.exists(project_path):
        #  print("PreTrainedModel, start from checkpoint")
         # num_epochs = 1
        #  model.load_weights(checkpoint_path)
        val_input = np.array(valSet["bytecode"])
        val_labels = np.array(valSet["label"])

        #val_input = np.squeeze(val_input, axis=1)
        #val_mask = np.squeeze(val_mask, axis=1)


        # Converte la lista di liste in un array NumPy
        train_labels = np.array(train_labels)

        model.fit(train_input, train_labels, epochs=num_epochs, batch_size=16, validation_data=(val_input, val_labels), callbacks=[cp_callback])


        # Converte la lista di liste in un array NumPy
        # Valutazione del modello
        loss_test, accuracy_test = model.evaluate(val_input, val_labels)
        print(f"Loss_Test: {loss_test}, Accuracy_Test: {accuracy_test}")

        loss_train, accuracy_train = model.evaluate(train_input, train_labels)
        print(f"Loss_Train: {loss_train}, Accuracy_Train: {accuracy_train}")

        # predict and calculate f1_score on val set
        y_pred_test = model.predict(val_input)

        # predict and calculate f1_score on train set
        y_pred_train = model.predict(train_input)
        print("RESULT TRAIN")
        evaluate_model(y_pred_train, train_labels)

        print("RESULT TEST")
        evaluate_model(y_pred_test, val_labels)

        # Classification Report (including precision, recall, and f1 for each class)
        class_report = classification_report(true_labels_flat, binary_predictions_flat, target_names=["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"])
        print("Classification Report:\n", class_report)
        #itertools_results = pd.concat([itertools_results, pd.DataFrame([[max_seq_length, dropout_rate, num_epochs, macro_avg_f1_score_test, macro_avg_f1_score_train, accuracy_test, accuracy_train, loss_test, loss_train]], columns=['max_seq_length', 'dropout_rate', 'num_epochs', 'Macro F1-score test', 'Macro F1-score train', 'Accuracy test', 'Accuracy train', 'Loss test', 'Loss train'])], ignore_index=True)

        # add results to a csv file with concat
        # results = pd.concat([results, pd.DataFrame([[dataset, len(lyrics), macro_avg_f1_score_test, macro_avg_f1_score_train, accuracy_test, accuracy_train, loss_test, loss_train]], columns=['Dataset', 'num_rows', 'Macro F1-score test', 'Macro F1-score train', 'Accuracy test', 'Accuracy train', 'Loss test', 'Loss train'])], ignore_index=True)

    # results.to_csv('/content/drive/MyDrive/Colab Notebooks/results_bert_pretrained.csv', index=False)
    itertools_results.to_csv('/content/drive/MyDrive/Colab Notebooks/results_bert_pretrained_itertools.csv', index=False)

if __name__ == '__main__':
    main()
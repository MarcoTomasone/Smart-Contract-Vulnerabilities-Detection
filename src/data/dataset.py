import torch
from config.config import CACHE_DIR, MODE, MODEL, MAX_LEN, CODE_BLOCKS, TOKENIZER
from datasets import load_dataset
from data.preprocessing import remove_comments, delete_newline_and_getters, one_hot_encode_label
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN * CODE_BLOCKS):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.code = dataframe["code"]
        self.targets = dataframe["label"]

    def __len__(self):
        return len(self.code)

    def __getitem__(self, index):
        code = str(self.code[index])
        code = " ".join(code.split())  # Clean up any excessive whitespace
        
        if len(code) == 0:
            raise ValueError(f"Code at index {index} is empty.")
        
        # Tokenize and encode the input text
        inputs = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids = (MODEL == 'bert')  # Include token_type_ids only if MODE is 'bert'
        )

        # Extract required tensors
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Initialize the output dictionary
        output = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

        # Conditionally add token_type_ids if MODE is 'bert'
        if MODEL == 'bert':
            output['token_type_ids'] = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        return output
    
def transformData(example):
    if MODE == "bytecode":
        bytecode = example["bytecode"]
        if bytecode.startswith("0x"):
            bytecode = bytecode[2:]
        bytecode_tokens = [bytecode[i:i+2] for i in range(0, len(bytecode), 2)]
        bytecode_text = ' '.join(bytecode_tokens)  # Create a string with spaces between tokens
        return {
            "code": bytecode_text,
            "label": one_hot_encode_label(example["slither"])
        }
    else:
        return {
            "code": delete_newline_and_getters(remove_comments(example["source_code"])),
            "label": one_hot_encode_label(example["slither"])
        }
        

def readAndPreprocessDataset():
    train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', cache_dir=CACHE_DIR, ignore_verifications=True).map(transformData)
    test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', cache_dir=CACHE_DIR, ignore_verifications=True).map(transformData)
    val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', cache_dir=CACHE_DIR, ignore_verifications=True).map(transformData)
    
    training_set = CustomDataset(train_set, TOKENIZER, MAX_LEN)
    testing_set = CustomDataset(test_set, TOKENIZER, MAX_LEN)
    validation_set = CustomDataset(val_set, TOKENIZER, MAX_LEN)
    
    return training_set, testing_set, validation_set

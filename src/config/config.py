from transformers import AutoTokenizer

MAX_LEN = 512
CODE_BLOCKS = 1  # Number of 512-token blocks needed for aggregate models, 1 if not needed
SAFE_IDX = 4
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05
NUM_CLASSES = 5
TOKENIZER = AutoTokenizer.from_pretrained('microsoft/codebert-base')
MODE = "bytecode" #or "source_code"
MODEL = "codeBert" #name of the model needed for printing logs
CACHE_DIR = "./cache"
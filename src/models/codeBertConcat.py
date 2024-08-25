import torch
from transformers import AutoModel
from config.config import CODE_BLOCKS

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

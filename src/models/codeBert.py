import torch
from transformers import  AutoModel

class CodeBERTClass(torch.nn.Module):
    def __init__(self, NUM_CLASSES):
        super(CodeBERTClass, self).__init__()
        self.num_classes = NUM_CLASSES
        self.codebert = AutoModel.from_pretrained('microsoft/codebert-base', cache_dir="./cache")
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(768, NUM_CLASSES)
    
    def forward(self, ids, mask, token_type_ids):
        outputs = self.codebert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        output = self.fc(pooled_output)
        return output
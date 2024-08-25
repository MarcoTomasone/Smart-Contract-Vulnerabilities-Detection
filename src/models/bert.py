import torch
from transformers import BertModel

class BERTClass(torch.nn.Module):
    def __init__(self, num_classes):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir="./cache")
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768, num_classes)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.dropout(output_1)
        output = self.fc(output_2)
        return output
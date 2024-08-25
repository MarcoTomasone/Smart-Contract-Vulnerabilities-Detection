import torch
from transformers import BertModel
# BERT Model Class
class BERTAggregatedClass(torch.nn.Module):
    def __init__(self, num_classes, aggregation='mean', dropout=0.3):
        super(BERTAggregatedClass, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir="./cache")
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        self.aggregation = aggregation

    def forward(self, input_ids, attention_masks):
        batch_size, seq_len = input_ids.size()

        # Divide input_ids e attention_masks in blocchi di 512 token
        num_chunks = (seq_len + 511) // 512
        input_ids = input_ids[:, :512*num_chunks].view(batch_size * num_chunks, 512)
        attention_masks = attention_masks[:, :512*num_chunks].view(batch_size * num_chunks, 512)

        # Ottieni l'output del modello BERT
        _, last_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_masks, return_dict=False)

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
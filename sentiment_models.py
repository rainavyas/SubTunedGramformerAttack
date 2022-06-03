from transformers import BertModel
import torch
import torch.nn as nn
from transformers import BertTokenizer


class BertSequenceClassifier(nn.Module):
    '''
    Sentence Level classification using Bert for encoding sentence
    '''
    def __init__(self, hidden_size=768, classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(hidden_size, classes)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask=None):
        '''
        input_ids = [N x L], containing sequence of ids of words after tokenization
        attention_mask = [N x L], mask for attention

        N = batch size
        L = maximum sentence length
        '''

        output = self.bert(input_ids, attention_mask)
        sentence_embedding = output.pooler_output # 1st token followed by linear layer and tanh
        logits = self.classifier(sentence_embedding)
        return logits
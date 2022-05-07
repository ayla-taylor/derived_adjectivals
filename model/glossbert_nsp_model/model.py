from typing import Any

from datasets import load_dataset, load_metric, ClassLabel, Value
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW
import numpy as np
import torch
from tqdm import tqdm

LABELS = ['incorrect', 'correct']
CLASSLABEL = ClassLabel(num_classes=len(LABELS), names=list(LABELS))


class DerivedAdjDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)



def make_datasets(data: tuple[str, Any], tokenizer: BertTokenizer) -> None: #TODO: properly type annotate the output (prob dataset?)

    def encode(examples):
        return tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length', max_length=128)

    name, dataset = data
    inputs = dataset.map(encode, batched=True)
    inputs = inputs.map(lambda examples: {'labels': examples['label']}, batched=True)
    inputs.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(inputs, batch_size=32)
    return dataloader

from typing import Any

from datasets import load_dataset, load_metric, ClassLabel, Value
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW
import numpy as np
import torch
from tqdm import tqdm

LABELS = ['incorrect', 'correct']
CLASSLABEL = ClassLabel(num_classes=len(LABELS), names=list(LABELS))


def make_datasets(data: tuple[str, Any], tokenizer: BertTokenizer) -> DataLoader:
    def encode(examples):
        return tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length', max_length=128)

    name, dataset = data
    inputs = dataset.map(encode, batched=True)
    inputs = inputs.map(lambda examples: {'labels': examples['label']}, batched=True)
    inputs.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # dataloader = torch.utils.data.DataLoader(inputs, batch_size=32)
    return inputs


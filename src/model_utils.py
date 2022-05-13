from typing import Any

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric, ClassLabel, Value
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW, TrainingArguments, Trainer, \
    BertForSequenceClassification

# LABELS = ['incorrect', 'correct']
# CLASSLABEL = ClassLabel(num_classes=len(LABELS), names=list(LABELS))
F1_METRIC = load_metric('f1')

MODEL_INFO = {'baseline': {'base_model': 'bert-base-uncased',
                           'model_dir': 'baseline_model'},
              'derived_embed': {'base_model': 'bert-base-uncased',
                                'model_dir': 'derived_embed'},
              'full_model': {'base_model': ['../model/baseline_model/checkpoint-',
                                            '../model/derived_embed/checkpoint-'],
                             'model_dir': 'full_model'}
              }


class DenseModel(nn.Module):

    def __init__(self, embed_size, num_classes):
        super().__init__()
        self.dense = torch.nn.Linear(embed_size, num_classes)

    def forward(self, x):
        return self.dense(x)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return F1_METRIC.compute(predictions=predictions, references=labels, average='micro')


def make_datasets(data: tuple[str, Any], tokenizer: Any, max_len: int, model: str) -> Any:
    def encode(examples):
        if model != 'full_model':
            return tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length', max_length=max_len)
        else:
            if model_group == 'full_model_base':
                return tokenizer[0](examples['text1'], examples['text2'], truncation=True, padding=True, return_tensors='pt')
            elif model_group == 'full_model_embed':
                return tokenizer[1](examples['derived_pair'], truncation=True, padding=True, return_tensors='pt')

    name, dataset = data
    if model == 'full_model':
        model_group = 'full_model_base'
        inputs_base = dataset.map(encode)
        model_group = 'full_model_embed'
        inputs_embeds = dataset.map(encode)
        inputs_base = inputs_base.remove_columns(['text1', 'text2', 'derived_pair'])
        inputs_base = inputs_base.rename_column('label', 'labels')
        inputs_base = inputs_base.with_format('torch')

        inputs_embeds = inputs_embeds.remove_columns(['text1', 'text2', 'derived_pair'])
        inputs_embeds = inputs_embeds.rename_column('label', 'labels')
        inputs_embeds = inputs_embeds.with_format('torch')
        return inputs_base, inputs_embeds
    inputs = dataset.map(encode, batched=True)
    inputs = inputs.remove_columns(['text1', 'text2'])
    inputs = inputs.rename_column('label', 'labels')
    inputs = inputs.with_format('torch')
    return inputs

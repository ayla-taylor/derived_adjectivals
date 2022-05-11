from typing import Any

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric, ClassLabel, Value
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW, TrainingArguments, Trainer, \
    BertForSequenceClassification
from model_train import args

LABELS = ['incorrect', 'correct']
CLASSLABEL = ClassLabel(num_classes=len(LABELS), names=list(LABELS))
F1_METRIC = load_metric('f1')

MODEL_INFO = {'baseline': {'tokenizer': BertTokenizer.from_pretrained('distilbert-base-uncased'),
                           'model': BertForNextSentencePrediction.from_pretrained('distilbert-base-uncased'),
                           'model_dir': 'baseline_model'},
              'derived_embed': {'tokenizer': BertTokenizer.from_pretrained('distilbert-base-uncased'),
                                'model': BertForSequenceClassification.from_pretrained('distilbert-base-uncased'),
                                'model_dir': 'derived_embed'},
              'full_model': {'tokenizer': [BertTokenizer.from_pretrained('../model/baseline_model'),
                                           BertTokenizer.from_pretrained('../model/derived_embed')],
                             'model': [BertForNextSentencePrediction.from_pretrained('distilbert-base-uncased'),
                                       BertForSequenceClassification.from_pretrained('../model/derived_embed')],
                             'model_dir': 'full_model'}
              }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return F1_METRIC.compute(predictions=predictions, references=labels, average='micro')


def make_datasets(data: tuple[str, Any], tokenizer: BertTokenizer) -> Any:
    def encode(examples):
        return tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length',
                         max_length=args.max_len)

    name, dataset = data
    inputs = dataset.map(encode, batched=True)
    inputs = inputs.remove_columns(['text1', 'text2'])
    inputs = inputs.rename_column('label', 'labels')
    inputs = inputs.with_format('torch')
    return inputs

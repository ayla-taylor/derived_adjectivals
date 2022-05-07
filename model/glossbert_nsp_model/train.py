from typing import Any

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric, ClassLabel, Value
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW

LABELS = ['incorrect', 'correct']
CLASSLABEL = ClassLabel(num_classes=len(LABELS), names=list(LABELS))


def make_datasets(data: tuple[str, Any], tokenizer: BertTokenizer) -> DataLoader:
    def encode(examples):
        return tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length', max_length=128)

    name, dataset = data
    inputs = dataset.map(encode, batched=True)
    inputs = inputs.map(lambda examples: {'labels': examples['label']}, batched=True)
    inputs.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(inputs, batch_size=32)
    return dataloader


def train(dataloader: DataLoader, model: Any):
    print("Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)
    optim = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    for epoch in range(3):
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 10 == 0:
                print(f"loss: {loss}")


def main():
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('distilbert-base-uncased')
    filepath = '../../data/data/glossbert/'
    dataset = load_dataset('csv', data_files={'train': filepath + 'train.csv',
                                              'dev': filepath + 'dev.csv',
                                              'test': filepath + 'test.csv'})
    # print(dataset)
    dataloaders = {}
    for name, data in dataset.items():
        dataloaders[name] = make_datasets((name, data), tokenizer)
    train(dataloaders['train'], model)


if __name__ == '__main__':
    main()
